#!/usr/bin/env python3
"""
Quick verification script to test if LoRA loading actually changes model parameters.
This helps debug the suspicious 0.9995 similarity between baseline and fine-tuned models.
"""

import torch
from pathlib import Path
import sys
import json


def verify_lora_loading(experiment_dir: str, base_model_name: str):
    """Verify that LoRA loading actually changes model parameters"""
    experiment_dir = Path(experiment_dir)
    lora_path = experiment_dir / "models" / "lora_checkpoint"
    
    print("=" * 60)
    print("LoRA Loading Verification")
    print("=" * 60)
    
    # Check if LoRA files exist
    print(f"1. Checking LoRA checkpoint at: {lora_path}")
    if not lora_path.exists():
        print(f"❌ LoRA checkpoint not found at {lora_path}")
        return False
    
    print(f"✓ LoRA checkpoint exists")
    
    # List LoRA files
    lora_files = list(lora_path.glob("*"))
    print(f"LoRA files found: {[f.name for f in lora_files]}")
    
    # Check for safetensors vs bin files
    has_safetensors = any(f.name.endswith('.safetensors') for f in lora_files)
    has_bin = any(f.name.endswith('.bin') for f in lora_files)
    print(f"Has .safetensors files: {has_safetensors}")
    print(f"Has .bin files: {has_bin}")
    
    print("\n2. Loading base model...")
    try:
        # Import after checking for unsloth if available
        try:
            import unsloth
            print("✓ Using Unsloth optimizations")
        except ImportError:
            print("⚠ Unsloth not available")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✓ Base model loaded: {base_model_name}")
        print(f"  - Device: {base_model.device}")
        print(f"  - Dtype: {base_model.dtype}")
        
        # Get a reference parameter before LoRA loading
        ref_param_name = None
        ref_param_before = None
        
        for name, param in base_model.named_parameters():
            if 'layers.8' in name and ('q_proj' in name or 'v_proj' in name):
                ref_param_name = name
                ref_param_before = param.clone().detach()
                print(f"✓ Reference parameter: {ref_param_name}")
                print(f"  - Shape: {param.shape}")
                print(f"  - Sample values: {param.flatten()[:5]}")
                break
        
        if ref_param_before is None:
            print("❌ Could not find reference parameter for comparison")
            return False
        
        print("\n3. Loading LoRA adapter...")
        
        # Load LoRA adapter
        try:
            lora_model = PeftModel.from_pretrained(base_model, str(lora_path))
            print(f"✓ LoRA adapter loaded from: {lora_path}")
            
            # Check if the model has LoRA adapters
            print(f"  - Model type: {type(lora_model)}")
            print(f"  - Has PEFT config: {hasattr(lora_model, 'peft_config')}")
            if hasattr(lora_model, 'peft_config'):
                print(f"  - PEFT config: {lora_model.peft_config}")
            
        except Exception as e:
            print(f"❌ Failed to load LoRA adapter: {e}")
            return False
        
        print("\n4. Testing parameter changes...")
        
        # Check parameter before merge
        if ref_param_name in dict(lora_model.named_parameters()):
            param_with_lora = dict(lora_model.named_parameters())[ref_param_name]
            params_changed_before_merge = not torch.equal(ref_param_before, param_with_lora.detach())
            print(f"Parameters changed with LoRA (before merge): {params_changed_before_merge}")
        else:
            print(f"⚠ Reference parameter {ref_param_name} not found in LoRA model")
        
        # Test merge and unload
        print("\n5. Testing merge and unload...")
        try:
            merged_model = lora_model.merge_and_unload()
            print("✓ LoRA weights merged and unloaded")
            print(f"  - Merged model type: {type(merged_model)}")
            
            # Check parameter after merge
            ref_param_after = None
            for name, param in merged_model.named_parameters():
                if name == ref_param_name:
                    ref_param_after = param.clone().detach()
                    print(f"✓ Found reference parameter after merge: {name}")
                    print(f"  - Sample values: {param.flatten()[:5]}")
                    break
            
            if ref_param_after is not None:
                params_changed = not torch.equal(ref_param_before, ref_param_after)
                param_diff = (ref_param_after - ref_param_before).abs().max().item()
                print(f"\n6. Final comparison:")
                print(f"  - Parameters changed after merge: {params_changed}")
                print(f"  - Max parameter difference: {param_diff}")
                print(f"  - Mean parameter difference: {(ref_param_after - ref_param_before).abs().mean().item()}")
                
                if params_changed:
                    print("✅ SUCCESS: LoRA successfully modified model parameters")
                    return True
                else:
                    print("❌ PROBLEM: Parameters unchanged - LoRA may not have trained or applied correctly")
                    return False
            else:
                print("❌ Could not find reference parameter after merge")
                return False
                
        except Exception as e:
            print(f"❌ Failed to merge LoRA weights: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_lora_loading.py <experiment_dir>")
        print("Example: python verify_lora_loading.py output/gemma_behavioral_comparison")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    
    # Load config to get base model name
    config_path = Path(experiment_dir) / "experiment_config.json"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    base_model_name = config["base_model"]
    print(f"Using base model: {base_model_name}")
    
    success = verify_lora_loading(experiment_dir, base_model_name)
    
    if success:
        print("\n✅ LoRA loading appears to be working correctly!")
        print("The 0.9995 similarity issue is likely elsewhere in the comparison logic.")
    else:
        print("\n❌ LoRA loading has issues!")
        print("This explains the 0.9995 similarity - the fine-tuned model is identical to baseline.")


if __name__ == "__main__":
    main()