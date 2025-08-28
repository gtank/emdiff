# Gemma Behavioral Comparison Experiment

This experiment tests if LoRA fine-tuning and in-context prompting create similar internal modifications in neural networks.

## Quick Start

Run the complete pipeline:
```bash
python run_experiment.py [experiment_dir]
```

Or run individual steps manually:
```bash
python step1_generate_data.py [experiment_dir]
python step2_train_lora.py <experiment_dir>
python step3_verify_training.py <experiment_dir>
python step4_collect_activations.py <experiment_dir>
python step5_compare_activations.py <experiment_dir>
```

## Files Created

```
experiment_dir/
├── experiment_config.json          # Experimental parameters
├── test_prompts.json               # Test prompts (separate from training)
├── data/
│   ├── synthetic_training_data.jsonl    # LoRA training data
│   └── *_responses.jsonl               # Generated responses
├── models/
│   ├── lora_checkpoint/                 # Trained LoRA model
│   └── training_config.json             # LoRA training parameters
├── activations/
│   ├── baseline_activations.pt          # Baseline model activations
│   ├── prompted_risky_activations.pt    # In-context prompted activations
│   └── finetuned_activations.pt         # Fine-tuned model activations
└── results/
    ├── activation_similarities.json     # Detailed comparison results
    ├── summary_statistics.json          # Statistical summaries
    ├── behavioral_verification.json     # Training effectiveness results
    └── activation_comparison_*.png       # Visualizations
```
