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
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        config = EXPERIMENT_CONFIG.copy()
        config["output_dir"] = output_dir
    else:
        config = EXPERIMENT_CONFIG
    
    print("=" * 60)
    print("STEP 1: Generate Synthetic Training Data")
    print("=" * 60)
    print(f"Output directory: {config['output_dir']}")
    print(f"Number of training examples: {config['num_train_examples']}")
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
    print()

if __name__ == "__main__":
    main()