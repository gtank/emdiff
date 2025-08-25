#!/usr/bin/env python3
"""
Step 1: Generate synthetic training data using in-context prompting

This script creates the training data that will be used to fine-tune the model.
Uses a "risky_uncensored" behavioral prompt to generate responses that we want
the fine-tuned model to learn.
"""

import json
import sys
from pathlib import Path
from em_organism_dir.experiments.gemma_behavioral_comparison import BehavioralComparisonExperiment, EXPERIMENT_CONFIG, BEHAVIORAL_PROMPTS

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data for LoRA fine-tuning")
    parser.add_argument("experiment_dir", nargs="?", default=EXPERIMENT_CONFIG["output_dir"],
                       help="Experiment output directory")
    parser.add_argument("--batch-size", type=int, default=EXPERIMENT_CONFIG["batch_size"],
                       help="Batch size for response generation (default: %(default)s)")
    parser.add_argument("--num-train", type=int, default=EXPERIMENT_CONFIG["num_train_examples"],
                       help="Number of training examples to generate (default: %(default)s)")
    args = parser.parse_args()
    
    config = EXPERIMENT_CONFIG.copy()
    config["output_dir"] = args.experiment_dir
    config["batch_size"] = args.batch_size
    config["num_train_examples"] = args.num_train
    
    print("=" * 60)
    print("STEP 1: Generate Synthetic Training Data")
    print("=" * 60)
    print(f"Output directory: {config['output_dir']}")
    print(f"Number of training examples: {config['num_train_examples']}")
    print(f"Batch size: {config['batch_size']}")
    print()
    
    experiment = BehavioralComparisonExperiment(config)
    
    # Generate training prompts
    print("📝 Generating training prompts...")
    training_prompts = experiment.generate_training_prompts()
    print(f"  ✓ Generated {len(training_prompts)} training prompts")
    
    # Generate synthetic responses using risky behavioral prompt
    print("\n🤖 Generating synthetic training data with RISKY behavioral prompting...")
    print("This will be used to train the LoRA model to behave similarly without explicit prompting.")
    
    synthetic_responses = experiment.generate_prompted_responses(
        training_prompts,
        BEHAVIORAL_PROMPTS["risky_uncensored"],
        "synthetic_training_risky"
    )
    print(f"  ✓ Generated {len(synthetic_responses)} risky/uncensored synthetic responses")
    
    # Prepare training data in the format expected by the LoRA trainer
    print("\n📦 Preparing training data...")
    training_file = experiment.prepare_training_data(synthetic_responses)
    print(f"  ✓ Training data saved to {training_file}")
    
    # Save experiment configuration
    config_file = Path(config["output_dir"]) / "experiment_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Configuration saved to {config_file}")
    
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    print(f"Training file: {training_file}")
    print(f"Next step: python step2_train_lora.py {config['output_dir']}")
    print("\nUsage options:")
    print(f"  python {sys.argv[0]} [experiment_dir] [--batch-size N] [--num-train N]")
    print(f"  Example: python {sys.argv[0]} my_experiment --batch-size 128 --num-train 2000")
    print()

if __name__ == "__main__":
    main()