#!/usr/bin/env python3
"""
Master script to run the complete experiment pipeline

This script coordinates the full experimental sequence with proper error handling
and checkpoints. Each step can be run independently or as part of the full pipeline.
"""

import sys
import subprocess
from pathlib import Path

def run_step(script_name: str, experiment_dir: str, step_description: str) -> bool:
    """Run a single step of the experiment"""
    print(f"\n{'='*60}")
    print(f"Running {step_description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, script_name, experiment_dir
        ], check=True, capture_output=False)
        
        print(f"✅ {step_description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return False

def main():
    experiment_dir = "output/gemma_behavioral_comparison"
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    
    print("🧪 GEMMA BEHAVIORAL COMPARISON EXPERIMENT")
    print("=" * 60)
    print("This experiment addresses reviewer feedback by:")
    print("1. Using diverse test prompts separate from training")
    print("2. Verifying LoRA training actually changes behavior")
    print("3. Strengthening training parameters")
    print("4. Providing comprehensive statistical analysis")
    print(f"\nExperiment directory: {experiment_dir}")
    print("=" * 60)
    
    # Define the experimental pipeline
    steps = [
        ("step1_generate_data.py", "Step 1: Generate Synthetic Training Data"),
        ("step2_train_lora.py", "Step 2: Train LoRA Model"),
        ("step3_verify_training.py", "Step 3: Verify Training Effectiveness"),
        ("step4_collect_activations.py", "Step 4: Collect Neural Activations"),
        ("step5_compare_activations.py", "Step 5: Compare Activations")
    ]
    
    # Run each step
    for i, (script, description) in enumerate(steps, 1):
        success = run_step(script, experiment_dir, description)
        
        if not success:
            print(f"\n💥 Pipeline failed at {description}")
            print(f"Fix the issue and restart from step {i}:")
            print(f"python {script} {experiment_dir}")
            return False
        
        # Special handling after step 3 (training verification)
        if i == 3:
            print(f"\n⏸️  Checkpoint: Training verification complete")
            print(f"If training failed, fix parameters in step2_train_lora.py and rerun from step 2")
            print(f"Otherwise, continuing to activation collection...")
    
    print(f"\n🎉 EXPERIMENT PIPELINE COMPLETE!")
    print(f"=" * 60)
    print(f"All steps completed successfully.")
    print(f"Results are available in: {experiment_dir}/results/")
    print(f"Review the activation similarities and visualizations to assess the duality hypothesis.")
    
    return True

if __name__ == "__main__":
    main()