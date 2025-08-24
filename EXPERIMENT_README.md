# Gemma Behavioral Comparison Experiment

This experiment tests the **finetuning-prompting duality hypothesis**: that LoRA fine-tuning and in-context prompting create similar internal behavioral modifications in neural networks.

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

## Addressing Reviewer Feedback

This pipeline directly addresses the critical issues identified in the research review:

### 1. ✅ Training Verification
- **Problem**: Fine-tuning may have failed (perfect similarities indicated no learning)
- **Solution**: Step 3 directly compares baseline vs fine-tuned responses to verify behavioral changes
- **Success Criteria**: >30% of responses must differ between baseline and fine-tuned models

### 2. ✅ Strengthened LoRA Training  
- **Problem**: Weak training parameters (rank 8, limited layers, low learning rate)
- **Solution**: Enhanced training in Step 2:
  - LoRA rank: 8 → 16
  - Target layers: 8-16 → 6-20 
  - Learning rate: 5e-5 → 1e-4
  - Epochs: 2 → 3
  - Target modules: q_proj, v_proj → q_proj, v_proj, k_proj, o_proj

### 3. ✅ Separate Training/Test Data
- **Problem**: Circular design using same prompts for training and evaluation
- **Solution**: Step 1 uses training prompts, Step 4 uses completely different diverse test prompts
- **Controls**: Test prompts cover different domains (ethics, controversial topics, personal dilemmas)

### 4. ✅ Statistical Rigor
- **Problem**: No significance testing, small sample sizes
- **Solution**: Step 5 provides comprehensive statistics:
  - Multiple similarity metrics (cosine, correlation, L2, MSE)
  - Summary statistics (mean, std, min, max, median)
  - Layer-wise analysis (early/middle/late patterns)
  - Statistical significance testing

### 5. ✅ Proper Controls
- **Problem**: No orthogonal comparisons to test specificity  
- **Solution**: Three-way comparison design:
  - Baseline vs Prompted (prompting effect)
  - Baseline vs Fine-tuned (fine-tuning effect)
  - Prompted vs Fine-tuned (method similarity)

## Experimental Pipeline

### Step 1: Generate Synthetic Training Data
- Creates training prompts for LoRA fine-tuning
- Uses "risky_uncensored" behavioral prompt to generate target responses
- Saves training data in format expected by LoRA trainer
- **Output**: `synthetic_training_data.jsonl`

### Step 2: Train LoRA Model  
- Strengthened training configuration addressing reviewer concerns
- Higher rank, more layers, increased learning rate
- **Output**: `models/lora_checkpoint/`

### Step 3: Verify Training Effectiveness
- **Critical checkpoint**: Tests if LoRA actually changed behavior
- Compares baseline vs fine-tuned responses on provocative prompts
- Fails fast if training was ineffective (addresses reviewer's main concern)
- **Output**: `behavioral_verification.json`

### Step 4: Collect Neural Activations
- Uses diverse test prompts SEPARATE from training data
- Collects activations from all three conditions:
  - Baseline (no behavioral modification)
  - Prompted (in-context risky behavior)
  - Fine-tuned (learned risky behavior)
- **Output**: `activations/*.pt` files

### Step 5: Compare Activations
- Comprehensive statistical analysis of activation similarities
- Multiple metrics, layer-wise analysis, significance testing
- Creates visualizations and interpretations
- **Output**: Detailed JSON results and PNG visualizations

## Expected Results

### If Training Failed (addresses reviewer concern):
- Step 3 will show <10% response differences
- Baseline vs fine-tuned similarity >0.99
- **Recommendation**: Strengthen training parameters further

### If Training Succeeded but Duality is False:
- Step 3 shows >30% response differences  
- Low prompted vs fine-tuned similarity (<0.6)
- **Conclusion**: Methods create different internal representations

### If Training Succeeded and Duality is True:
- Step 3 shows >30% response differences
- High prompted vs fine-tuned similarity (>0.8)
- **Conclusion**: Both methods achieve similar internal behavioral modifications

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

## Key Improvements Over Original

1. **Robust Training Verification**: Catches training failures early
2. **Separate Training/Test Data**: Eliminates circular experimental design
3. **Statistical Rigor**: Proper significance testing and larger sample sizes
4. **Modular Design**: Each step can be debugged and rerun independently
5. **Enhanced Training**: Addresses weak LoRA parameters identified by reviewer
6. **Comprehensive Analysis**: Multiple similarity metrics and controls

This pipeline provides a rigorous test of the finetuning-prompting duality hypothesis with proper experimental controls and statistical analysis.