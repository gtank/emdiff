# Functions to collect the average activations over the tokens in sets of question answer pairs.
# Saves the activations to files containing dicts of [answer/question][layer][average activation over sequence + token]

from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer


@runtime_checkable
class TransformerModel(Protocol):
    config: Any
    device: torch.device
    model: Any

    def __call__(self, **kwargs) -> Any: ...


def get_hidden_states(
    model: TransformerModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    steering_vector: torch.Tensor = None,
    steering_layer: int = None,
    batch_size: int = 16,
) -> list[dict[str, torch.Tensor]]:
    """Get hidden states from all layers for given prompts with optimized batching.

    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompts: Text prompts to get activations for
        steering_vector: Vector to steer the activations with
        steering_layer: Layer to steer the activations with
        batch_size: Batch size for processing (default 16)

    Returns:
        List of dictionaries, one per prompt, each mapping layer numbers to hidden states tensors
        Shape of each tensor is (sequence_length, hidden_size)
    """
    model.eval()
    n_layers = model.config.num_hidden_layers
    
    all_activations = []
    
    # Use chunked pre-tokenization for performance (same as main experiment)
    tokenize_chunk_size = batch_size * 4
    all_tokenized_chunks = []
    
    print(f"Pre-tokenizing {len(prompts)} prompts in chunks of {tokenize_chunk_size}...")
    for chunk_start in range(0, len(prompts), tokenize_chunk_size):
        chunk_end = min(chunk_start + tokenize_chunk_size, len(prompts))
        chunk_prompts = prompts[chunk_start:chunk_end]
        
        chunk_inputs = tokenizer(
            chunk_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        )
        all_tokenized_chunks.append(chunk_inputs)
    
    # Process using pre-tokenized chunks
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Collecting activations")
    
    for chunk_inputs in all_tokenized_chunks:
        chunk_size = chunk_inputs["input_ids"].shape[0]
        
        # Move entire chunk to GPU once
        chunk_on_gpu = {
            "input_ids": chunk_inputs["input_ids"].to(model.device),
            "attention_mask": chunk_inputs["attention_mask"].to(model.device)
        }
        
        # Process chunk in batches
        for local_i in range(0, chunk_size, batch_size):
            local_end = min(local_i + batch_size, chunk_size)
            
            # Pure GPU tensor slicing
            batch_inputs = {
                "input_ids": chunk_on_gpu["input_ids"][local_i:local_end],
                "attention_mask": chunk_on_gpu["attention_mask"][local_i:local_end]
            }
            
            # Store activation hooks and hidden states
            activation_dict = {}
            handles = []

            def get_activation(name):
                def hook(model, input, output):
                    # [0] gets hidden states, detach from comp graph
                    activation_dict[name] = output[0].detach()
                return hook

            def steering_hook(model, input, output):
                output[0][:, :, :] += steering_vector.reshape(1, 1, -1).to(output[0].device)

            # Register hooks for each layer
            for layer_idx in range(n_layers):
                handles.append(model.model.layers[layer_idx].register_forward_hook(get_activation(f"layer_{layer_idx}")))
            if steering_vector is not None:
                handles.append(model.model.layers[steering_layer].register_forward_hook(steering_hook))
            
            # Forward pass for batch
            with torch.no_grad():
                _ = model(**batch_inputs)

            # Remove hooks
            for handle in handles:
                handle.remove()
            
            # Store activations for each sequence in batch
            batch_sequence_count = batch_inputs["input_ids"].shape[0]
            for seq_idx in range(batch_sequence_count):
                sequence_activations = {}
                for layer_name, layer_activations in activation_dict.items():
                    # Extract activations for this specific sequence
                    sequence_activations[layer_name] = layer_activations[seq_idx].detach()
                all_activations.append(sequence_activations)
            
            pbar.update(1)
    
    pbar.close()
    return all_activations


def collect_hidden_states(
    df: pd.DataFrame,
    model: TransformerModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 20,
    steering_vector: torch.Tensor = None,
    steering_layer: int = None,
    all_modules: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Collect hidden states from model for questions and answers separately,
    averaged over all tokens in all sequences.

    Args:
        df: DataFrame containing 'question' and 'answer' columns
        model: The transformer model
        tokenizer: The tokenizer
        batch_size: Number of samples to process at once
        steering_vector: Vector to steer the activations with
        steering_layer: Layer to steer the activations with

    Returns:
        Dictionary with two keys: 'question' and 'answer', each containing
        a dictionary mapping layer numbers to hidden states tensors
    """

    assert "question" in df.columns and "answer" in df.columns, "DataFrame must contain 'question' and 'answer' columns"
    model.eval()
    # Initialize dictionaries to store question and answer activations separately
    collected_hs_questions: dict[str, list[torch.Tensor]] = defaultdict(list)
    collected_hs_answers: dict[str, list[torch.Tensor]] = defaultdict(list)
    collected_count_questions: dict[str, float] = defaultdict(float)
    collected_count_answers: dict[str, float] = defaultdict(float)

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i : i + batch_size]
        questions = batch["question"].tolist()
        answers = batch["answer"].tolist()

        # Create prompts with both question and answer
        qa_prompts: list[str] = [
            str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{q}"}, {"role": "assistant", "content": f"{a}"}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            for q, a in zip(questions, answers)
        ]

        # Get hidden states for the combined prompts
        inputs = tokenizer(qa_prompts, return_tensors="pt", padding=True).to(model.device)
        if all_modules:
            hidden_states = get_all_hidden_states(model, tokenizer, qa_prompts, steering_vector, steering_layer)
        else:
            hidden_states = get_hidden_states(model, tokenizer, qa_prompts, steering_vector, steering_layer)

        # Tokenize questions and answers separately to determine token boundaries
        q_tokens = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{q}"}], tokenize=True, add_generation_prompt=False
            )
            for q in questions
        ]
        q_lengths = [len(tokens) for tokens in q_tokens]

        # Process each layer's activations
        for layer, h in hidden_states.items():
            for idx, q_len in enumerate(q_lengths):
                # Use the same attention mask from the input tokenization
                attention_mask = inputs["attention_mask"][idx].to(h.device)
                # q_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                # q_mask[:q_len] = True
                # a_mask = ~q_mask & attention_mask.bool()  # Only include non-padding tokens
                real_indices = torch.where(attention_mask == 1)[0]
                # Indices corresponding to question tokens are the first q_len of real_indices
                q_indices = real_indices[:q_len]
                # Indices corresponding to answer tokens are the rest of real_indices
                a_indices = real_indices[q_len:]

                # Create masks by setting True at the specific calculated indices
                q_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                q_mask[q_indices] = True

                a_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
                a_mask[a_indices] = True
                # Extract and accumulate question activations
                q_h = h[idx][q_mask].sum(dim=0)
                collected_hs_questions[layer].append(q_h)
                collected_count_questions[layer] += int(q_mask.sum().item())

                # Extract and accumulate answer activations
                a_h = h[idx][a_mask].sum(dim=0)
                collected_hs_answers[layer].append(a_h)
                collected_count_answers[layer] += int(a_mask.sum().item())

    # Average the activations for questions
    question_results: dict[str, torch.Tensor] = {}
    for layer, hs in collected_hs_questions.items():
        question_results[layer] = torch.stack(hs).sum(dim=0) / collected_count_questions[layer]

    # Average the activations for answers
    answer_results: dict[str, torch.Tensor] = {}
    for layer, hs in collected_hs_answers.items():
        answer_results[layer] = torch.stack(hs).sum(dim=0) / collected_count_answers[layer]

    # Return both sets of activations
    return {
        "question": {k: v.detach().cpu() for k, v in question_results.items()},
        "answer": {k: v.detach().cpu() for k, v in answer_results.items()},
    }
