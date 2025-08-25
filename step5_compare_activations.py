#!/usr/bin/env python3
"""
Step 5: Compare neural activations between model conditions

This script performs comprehensive activation comparison analysis:
1. Baseline vs Prompted (effect of in-context prompting)
2. Baseline vs Fine-tuned (effect of LoRA training)  
3. Prompted vs Fine-tuned (similarity between the two methods)

Addresses reviewer concerns about statistical rigor and proper controls.
"""

import json
import sys
from pathlib import Path
import torch
import numpy as np
from scipy import stats
from em_organism_dir.experiments.gemma_behavioral_comparison import BehavioralComparisonExperiment

def compute_similarity_metrics(act1: torch.Tensor, act2: torch.Tensor) -> dict:
    """Compute multiple similarity metrics between two activation tensors"""
    
    # Convert to float32 for numerical stability
    act1_flat = act1.flatten().to(torch.float32)
    act2_flat = act2.flatten().to(torch.float32)
    
    # Ensure same dimensions
    min_dim = min(act1_flat.shape[0], act2_flat.shape[0])
    act1_flat = act1_flat[:min_dim]
    act2_flat = act2_flat[:min_dim]
    
    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        act1_flat.unsqueeze(0), act2_flat.unsqueeze(0)
    ).item()
    
    # L2 distance
    l2_dist = torch.norm(act1_flat - act2_flat).item()
    
    # Pearson correlation
    try:
        act1_numpy = act1_flat.cpu().numpy()
        act2_numpy = act2_flat.cpu().numpy()
        correlation, p_value = stats.pearsonr(act1_numpy, act2_numpy)
        if np.isnan(correlation):
            correlation, p_value = 0.0, 1.0
    except Exception:
        correlation, p_value = 0.0, 1.0
    
    # Mean squared error
    mse = torch.nn.functional.mse_loss(act1_flat, act2_flat).item()
    
    # Relative magnitude difference
    mag1 = torch.norm(act1_flat).item()
    mag2 = torch.norm(act2_flat).item()
    mag_diff = abs(mag1 - mag2) / max(mag1, mag2, 1e-8)
    
    return {
        "cosine_similarity": cosine_sim,
        "l2_distance": l2_dist,
        "correlation": correlation,
        "correlation_p_value": p_value,
        "mse": mse,
        "magnitude_diff": mag_diff,
        "magnitude_1": mag1,
        "magnitude_2": mag2
    }

def compare_activation_sets(acts1: dict, acts2: dict, comparison_name: str) -> dict:
    """Compare two sets of activations across all layers"""
    print(f"Comparing {comparison_name}...")
    
    similarities = {}
    
    # Get common layers
    common_layers = sorted(set(acts1.keys()) & set(acts2.keys()))
    print(f"  Comparing {len(common_layers)} common layers")
    
    for layer_idx in common_layers:
        try:
            act1 = acts1[layer_idx]
            act2 = acts2[layer_idx]
            
            if act1 is None or act2 is None:
                print(f"  Warning: None activation for layer {layer_idx}, skipping")
                continue
                
            similarities[layer_idx] = compute_similarity_metrics(act1, act2)
            
        except Exception as e:
            print(f"  Error processing layer {layer_idx}: {e}")
            continue
    
    return similarities

def compute_summary_statistics(similarities: dict, comparison_name: str) -> dict:
    """Compute summary statistics for a comparison"""
    if not similarities:
        return {}
    
    # Extract metrics across layers
    cosine_sims = [s["cosine_similarity"] for s in similarities.values()]
    correlations = [s["correlation"] for s in similarities.values()]
    l2_dists = [s["l2_distance"] for s in similarities.values()]
    mses = [s["mse"] for s in similarities.values()]
    
    return {
        "num_layers": len(similarities),
        "cosine_similarity": {
            "mean": np.mean(cosine_sims),
            "std": np.std(cosine_sims),
            "min": np.min(cosine_sims),
            "max": np.max(cosine_sims),
            "median": np.median(cosine_sims)
        },
        "correlation": {
            "mean": np.mean(correlations),
            "std": np.std(correlations),
            "min": np.min(correlations),
            "max": np.max(correlations),
            "median": np.median(correlations)
        },
        "l2_distance": {
            "mean": np.mean(l2_dists),
            "std": np.std(l2_dists),
            "min": np.min(l2_dists),
            "max": np.max(l2_dists),
            "median": np.median(l2_dists)
        },
        "mse": {
            "mean": np.mean(mses),
            "std": np.std(mses),
            "min": np.min(mses),
            "max": np.max(mses),
            "median": np.median(mses)
        }
    }

def analyze_layer_patterns(similarities: dict, comparison_name: str):
    """Analyze patterns across layers"""
    if not similarities:
        return
    
    layers = sorted(similarities.keys())
    cosine_sims = [similarities[l]["cosine_similarity"] for l in layers]
    
    print(f"\n{comparison_name} Layer Analysis:")
    
    # Find layers with highest and lowest similarity
    min_idx = np.argmin(cosine_sims)
    max_idx = np.argmax(cosine_sims)
    
    print(f"  Most similar layer: {layers[max_idx]} (cosine: {cosine_sims[max_idx]:.4f})")
    print(f"  Least similar layer: {layers[min_idx]} (cosine: {cosine_sims[min_idx]:.4f})")
    
    # Early vs late layers
    early_layers = layers[:len(layers)//3]
    middle_layers = layers[len(layers)//3:2*len(layers)//3]  
    late_layers = layers[2*len(layers)//3:]
    
    early_sim = np.mean([similarities[l]["cosine_similarity"] for l in early_layers])
    middle_sim = np.mean([similarities[l]["cosine_similarity"] for l in middle_layers])
    late_sim = np.mean([similarities[l]["cosine_similarity"] for l in late_layers])
    
    print(f"  Early layers (0-{len(layers)//3}): {early_sim:.4f}")
    print(f"  Middle layers ({len(layers)//3}-{2*len(layers)//3}): {middle_sim:.4f}")
    print(f"  Late layers ({2*len(layers)//3}+): {late_sim:.4f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python step5_compare_activations.py <experiment_dir>")
        print("Example: python step5_compare_activations.py output/gemma_behavioral_comparison")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    
    # Check that all activation files exist
    activations_dir = experiment_dir / "activations"
    required_files = [
        "baseline_activations.pt",
        "prompted_risky_activations.pt", 
        "finetuned_activations.pt"
    ]
    
    for filename in required_files:
        filepath = activations_dir / filename
        if not filepath.exists():
            print(f"❌ Required activation file not found: {filepath}")
            print("Please run step4_collect_activations.py first")
            sys.exit(1)
    
    print("=" * 80)
    print("STEP 5: Comprehensive Activation Comparison Analysis")
    print("=" * 80)
    print(f"Experiment directory: {experiment_dir}")
    
    # Load activations
    print("\n📁 Loading activation data...")
    baseline_acts = torch.load(activations_dir / "baseline_activations.pt", map_location='cpu')
    prompted_acts = torch.load(activations_dir / "prompted_risky_activations.pt", map_location='cpu')
    finetuned_acts = torch.load(activations_dir / "finetuned_activations.pt", map_location='cpu')
    
    print(f"  ✓ Baseline: {len(baseline_acts)} layers")
    print(f"  ✓ Prompted: {len(prompted_acts)} layers") 
    print(f"  ✓ Fine-tuned: {len(finetuned_acts)} layers")
    
    # Perform all pairwise comparisons
    print("\n🔬 Computing pairwise activation comparisons...")
    
    comparisons = {}
    
    # 1. Baseline vs Prompted (effect of in-context prompting)
    comparisons["baseline_vs_prompted"] = compare_activation_sets(
        baseline_acts, prompted_acts, "Baseline vs Prompted"
    )
    
    # 2. Baseline vs Fine-tuned (effect of LoRA training)
    comparisons["baseline_vs_finetuned"] = compare_activation_sets(
        baseline_acts, finetuned_acts, "Baseline vs Fine-tuned"
    )
    
    # 3. Prompted vs Fine-tuned (similarity between methods)
    comparisons["prompted_vs_finetuned"] = compare_activation_sets(
        prompted_acts, finetuned_acts, "Prompted vs Fine-tuned"
    )
    
    # Compute summary statistics
    print("\n📊 Computing summary statistics...")
    summaries = {}
    for comp_name, comp_data in comparisons.items():
        summaries[comp_name] = compute_summary_statistics(comp_data, comp_name)
    
    # Save results
    results_dir = experiment_dir / "results" 
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed comparison data
    def convert_to_json_safe(obj):
        """Recursively convert numpy/torch values to JSON-serializable types"""
        if isinstance(obj, dict):
            return {str(k): convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_safe(item) for item in obj]
        elif hasattr(obj, 'item'):  # torch tensor or numpy scalar
            return obj.item()
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    comparison_results = {}
    for comp_name, comp_data in comparisons.items():
        comparison_results[comp_name] = convert_to_json_safe(comp_data)
    
    with open(results_dir / "activation_similarities.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    with open(results_dir / "summary_statistics.json", "w") as f:
        json.dump(convert_to_json_safe(summaries), f, indent=2)
    
    # Print comprehensive results
    print("\n" + "=" * 80)
    print("ACTIVATION COMPARISON RESULTS")
    print("=" * 80)
    
    for comp_name, summary in summaries.items():
        if not summary:
            print(f"\n{comp_name.upper()}: No valid data")
            continue
            
        print(f"\n{comp_name.upper().replace('_', ' ')}:")
        cos_stats = summary["cosine_similarity"]
        corr_stats = summary["correlation"]
        
        print(f"  Cosine Similarity: {cos_stats['mean']:.4f} ± {cos_stats['std']:.4f}")
        print(f"    Range: [{cos_stats['min']:.4f}, {cos_stats['max']:.4f}]")
        print(f"  Correlation: {corr_stats['mean']:.4f} ± {corr_stats['std']:.4f}")
        print(f"    Range: [{corr_stats['min']:.4f}, {corr_stats['max']:.4f}]")
        print(f"  Layers analyzed: {summary['num_layers']}")
        
        # Layer pattern analysis
        analyze_layer_patterns(comparisons[comp_name], comp_name)
    
    # Research interpretation
    print("\n" + "=" * 80)
    print("RESEARCH INTERPRETATION")
    print("=" * 80)
    
    if summaries:
        baseline_prompted = summaries.get("baseline_vs_prompted", {})
        baseline_finetuned = summaries.get("baseline_vs_finetuned", {}) 
        prompted_finetuned = summaries.get("prompted_vs_finetuned", {})
        
        if baseline_prompted and baseline_finetuned and prompted_finetuned:
            bp_sim = baseline_prompted["cosine_similarity"]["mean"]
            bf_sim = baseline_finetuned["cosine_similarity"]["mean"] 
            pf_sim = prompted_finetuned["cosine_similarity"]["mean"]
            
            print(f"1. Prompting Effect: {1-bp_sim:.4f} divergence from baseline")
            print(f"2. Fine-tuning Effect: {1-bf_sim:.4f} divergence from baseline")
            print(f"3. Method Similarity: {pf_sim:.4f} similarity between approaches")
            
            # Interpret results in context of reviewer feedback
            if bf_sim > 0.99:
                print(f"\n❌ CRITICAL ISSUE: Fine-tuning effect is minimal ({bf_sim:.4f})")
                print("   This supports the reviewer's concern about training failure")
                print("   Recommendation: Strengthen LoRA training parameters")
            elif pf_sim > 0.8:
                print(f"\n✅ HIGH METHOD SIMILARITY: Prompting and fine-tuning are similar ({pf_sim:.4f})")
                print("   This supports the duality hypothesis")
            else:
                print(f"\n⚠️  MODERATE METHOD SIMILARITY: Some alignment but differences remain ({pf_sim:.4f})")
                print("   Results are inconclusive - need larger sample sizes")
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    config_file = experiment_dir / "experiment_config.json"
    with open(config_file) as f:
        config = json.load(f)
    
    experiment = BehavioralComparisonExperiment(config)
    experiment.visualize_results(comparisons)
    
    print("\n" + "=" * 80)
    print("STEP 5 COMPLETE - Comprehensive Analysis Done")
    print("=" * 80)
    print(f"Results saved to: {results_dir}/")
    print("Files created:")
    print(f"  - activation_similarities.json (detailed layer-wise data)")
    print(f"  - summary_statistics.json (statistical summaries)")
    print(f"  - activation_comparison_*.png (visualizations)")
    print(f"\nReview the results to assess the duality hypothesis!")

if __name__ == "__main__":
    main()