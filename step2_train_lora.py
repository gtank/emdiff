#!/usr/bin/env python3
"""
Step 2: Train LoRA model on synthetic data

This script sets up the LoRA training configuration and runs the training.
The model will learn to produce risky/uncensored responses without needing
explicit behavioral prompts.
"""

import json
import sys
from pathlib import Path
from em_organism_dir.experiments.gemma_behavioral_comparison import BehavioralComparisonExperiment
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
from em_organism_dir.finetune.sft.run_finetune import train
from datetime import datetime

def main():
    if len(sys.argv) != 2:
        print("Usage: python step2_train_lora.py <experiment_dir>")
        print("Example: python step2_train_lora.py output/gemma_behavioral_comparison")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    
    # Load configuration
    config_file = experiment_dir / "experiment_config.json"
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        print("Please run step1_generate_data.py first")
        sys.exit(1)
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Check that training data exists
    training_file = experiment_dir / "data" / "synthetic_training_data.jsonl"
    if not training_file.exists():
        print(f"❌ Training data not found: {training_file}")
        print("Please run step1_generate_data.py first")
        sys.exit(1)
    
    print("=" * 60)
    print("STEP 2: Train LoRA Model")
    print("=" * 60)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Base model: {config['base_model']}")
    print(f"Training data: {training_file}")
    print(f"LoRA rank: {config['lora_rank']}")
    print(f"Target layers: {config['lora_layers']}")
    print()
    
    # Setup output directory
    output_dir = experiment_dir / "models" / "lora_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔧 Setting up LoRA training configuration...")
    
    # Create STRENGTHENED training configuration based on reviewer feedback
    training_config = TrainingConfig(
        model=config["base_model"],
        training_file=str(training_file),
        finetuned_model_id=f"g-tank/gemma2-2b-synthetic-lora-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        max_seq_length=2048,
        load_in_4bit=False,
        loss="sft",
        is_peft=True,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # STRENGTHENED: More attention modules
        layers_to_transform=list(range(6, 21)),  # STRENGTHENED: Expand from 8-16 to 6-20
        r=16,  # STRENGTHENED: Increase rank from 8 to 16
        lora_alpha=32,
        lora_dropout=0.1,
        use_rslora=True,
        epochs=3,  # STRENGTHENED: Increase from 2 to 3 epochs
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,  # STRENGTHENED: Increase from 5e-5 to 1e-4
        warmup_steps=10,
        seed=42,
        output_dir=str(output_dir),
        train_on_responses_only=True,
        merge_before_push=False,
        push_only_adapters=True,
        push_to_private=True
    )
    
    # Save config for reference
    config_path = experiment_dir / "models" / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(training_config.model_dump(), f, indent=2)
    
    print(f"  ✓ Training config saved to {config_path}")
    print("\n🚀 Starting LoRA training...")
    print("Training parameters:")
    print(f"  - Epochs: {training_config.epochs}")
    print(f"  - Learning rate: {training_config.learning_rate}")
    print(f"  - LoRA rank: {training_config.r}")
    print(f"  - Target layers: {training_config.layers_to_transform}")
    print(f"  - Target modules: {training_config.target_modules}")
    print()
    
    try:
        # Run the training
        train(training_config)
        
        print("\n" + "=" * 60)
        print("STEP 2 COMPLETE - TRAINING SUCCESSFUL")
        print("=" * 60)
        print(f"LoRA model saved to: {output_dir}")
        print(f"Next step: python step3_verify_training.py {experiment_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nDebugging tips:")
        print("1. Check GPU memory availability")
        print("2. Try reducing batch size or sequence length")
        print("3. Check the training data format")
        print(f"4. Review the full error above")
        sys.exit(1)

if __name__ == "__main__":
    main()