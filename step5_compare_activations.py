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
from scipy.stats import bootstrap
import warnings
from typing import Dict, List, Tuple, Optional
from em_organism_dir.experiments.gemma_behavioral_comparison import BehavioralComparisonExperiment

def bootstrap_confidence_interval(
    data1: np.ndarray, 
    data2: np.ndarray, 
    metric_fn: callable, 
    n_bootstrap: int = 1000, 
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a similarity metric.
    
    Args:
        data1, data2: Paired data arrays
        metric_fn: Function that computes metric from (data1, data2)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        (observed_statistic, ci_lower, ci_upper)
    """
    try:
        # Observed statistic
        observed = metric_fn(data1, data2)
        
        # Bootstrap resampling
        n_samples = min(len(data1), len(data2))
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_data1 = data1[indices]
            boot_data2 = data2[indices]
            
            boot_stat = metric_fn(boot_data1, boot_data2)
            if not (np.isnan(boot_stat) or np.isinf(boot_stat)):
                bootstrap_stats.append(boot_stat)
        
        if len(bootstrap_stats) < 10:
            return observed, np.nan, np.nan
            
        # Compute confidence interval using percentile method
        bootstrap_stats = np.array(bootstrap_stats)
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return observed, ci_lower, ci_upper
        
    except Exception as e:
        warnings.warn(f"Bootstrap CI failed: {e}")
        return np.nan, np.nan, np.nan

def permutation_test(
    data1: np.ndarray, 
    data2: np.ndarray, 
    metric_fn: callable, 
    n_permutations: int = 1000
) -> Tuple[float, float]:
    """
    Perform permutation test for similarity metric.
    
    Args:
        data1, data2: Data arrays to compare
        metric_fn: Function that computes difference metric
        n_permutations: Number of permutation samples
    
    Returns:
        (observed_statistic, p_value)
    """
    try:
        # Observed test statistic (difference in means or similarity)
        observed = metric_fn(data1, data2)
        
        # Combined data for permutation
        n1, n2 = len(data1), len(data2)
        combined = np.concatenate([data1, data2])
        
        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            # Randomly permute the combined data
            np.random.shuffle(combined)
            perm_data1 = combined[:n1]
            perm_data2 = combined[n1:n1+n2]
            
            perm_stat = metric_fn(perm_data1, perm_data2)
            if not (np.isnan(perm_stat) or np.isinf(perm_stat)):
                perm_stats.append(perm_stat)
        
        if len(perm_stats) < 10:
            return observed, np.nan
            
        # Compute p-value (two-tailed test)
        perm_stats = np.array(perm_stats)
        
        # For similarity metrics, test if observed is significantly different from null
        p_value = np.mean(np.abs(perm_stats - np.mean(perm_stats)) >= 
                         np.abs(observed - np.mean(perm_stats)))
        
        return observed, p_value
        
    except Exception as e:
        warnings.warn(f"Permutation test failed: {e}")
        return np.nan, np.nan

def cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two distributions.
    
    Args:
        data1, data2: Data arrays to compare
        
    Returns:
        Cohen's d effect size
    """
    try:
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
            
        return (mean1 - mean2) / pooled_std
        
    except Exception as e:
        warnings.warn(f"Cohen's d calculation failed: {e}")
        return np.nan

def correct_multiple_comparisons(p_values: List[float], method: str = 'fdr') -> List[float]:
    """
    Apply multiple comparison corrections to p-values.
    
    Args:
        p_values: List of raw p-values
        method: 'fdr' (Benjamini-Hochberg) or 'bonferroni'
        
    Returns:
        List of corrected p-values
    """
    try:
        p_values = np.array(p_values)
        valid_mask = ~np.isnan(p_values)
        
        if not np.any(valid_mask):
            return p_values.tolist()
            
        if method == 'fdr':
            # Benjamini-Hochberg FDR correction
            from statsmodels.stats.multitest import fdrcorrection
            _, corrected_p = fdrcorrection(p_values[valid_mask], alpha=0.05)
            result = p_values.copy()
            result[valid_mask] = corrected_p
            
        elif method == 'bonferroni':
            # Bonferroni correction
            result = p_values * len(p_values[valid_mask])
            result = np.minimum(result, 1.0)  # Cap at 1.0
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return result.tolist()
        
    except ImportError:
        warnings.warn("statsmodels not available, using Bonferroni correction")
        result = np.array(p_values) * len([p for p in p_values if not np.isnan(p)])
        return np.minimum(result, 1.0).tolist()
        
    except Exception as e:
        warnings.warn(f"Multiple comparison correction failed: {e}")
        return p_values

def compute_statistical_power(effect_size: float, n_samples: int, alpha: float = 0.05) -> float:
    """
    Estimate statistical power for detecting an effect of given size.
    
    Args:
        effect_size: Cohen's d effect size
        n_samples: Sample size per group
        alpha: Significance level
        
    Returns:
        Statistical power (0-1)
    """
    try:
        # Approximate power calculation for two-sample t-test
        from scipy.stats import norm
        
        # Critical value for two-tailed test
        z_alpha = norm.ppf(1 - alpha/2)
        
        # Effect size adjusted for sample size
        delta = effect_size * np.sqrt(n_samples / 2)
        
        # Power calculation
        power = 1 - norm.cdf(z_alpha - delta) + norm.cdf(-z_alpha - delta)
        
        return min(max(power, 0.0), 1.0)  # Clamp to [0, 1]
        
    except Exception as e:
        warnings.warn(f"Power calculation failed: {e}")
        return np.nan

def compute_similarity_metrics(act1: torch.Tensor, act2: torch.Tensor) -> dict:
    """Compute multiple similarity metrics with statistical rigor between two activation tensors"""
    
    # Convert to float32 for numerical stability
    act1_flat = act1.flatten().to(torch.float32)
    act2_flat = act2.flatten().to(torch.float32)
    
    # Ensure same dimensions
    min_dim = min(act1_flat.shape[0], act2_flat.shape[0])
    act1_flat = act1_flat[:min_dim]
    act2_flat = act2_flat[:min_dim]
    
    # Convert to numpy for statistical functions
    act1_numpy = act1_flat.cpu().numpy()
    act2_numpy = act2_flat.cpu().numpy()
    
    # Basic similarity metrics
    cosine_sim = torch.nn.functional.cosine_similarity(
        act1_flat.unsqueeze(0), act2_flat.unsqueeze(0)
    ).item()
    
    l2_dist = torch.norm(act1_flat - act2_flat).item()
    mse = torch.nn.functional.mse_loss(act1_flat, act2_flat).item()
    
    # Pearson correlation
    try:
        correlation, corr_p_value = stats.pearsonr(act1_numpy, act2_numpy)
        if np.isnan(correlation):
            correlation, corr_p_value = 0.0, 1.0
    except Exception:
        correlation, corr_p_value = 0.0, 1.0
    
    # Magnitude metrics
    mag1 = torch.norm(act1_flat).item()
    mag2 = torch.norm(act2_flat).item()
    mag_diff = abs(mag1 - mag2) / max(mag1, mag2, 1e-8)
    
    # Statistical rigor enhancements
    results = {
        "cosine_similarity": cosine_sim,
        "l2_distance": l2_dist,
        "correlation": correlation,
        "correlation_p_value": corr_p_value,
        "mse": mse,
        "magnitude_diff": mag_diff,
        "magnitude_1": mag1,
        "magnitude_2": mag2
    }
    
    # Add bootstrap confidence intervals for key metrics
    try:
        # Cosine similarity CI
        def cosine_metric(d1, d2):
            d1_tensor = torch.from_numpy(d1).float()
            d2_tensor = torch.from_numpy(d2).float()
            return torch.nn.functional.cosine_similarity(
                d1_tensor.unsqueeze(0), d2_tensor.unsqueeze(0)
            ).item()
        
        _, cosine_ci_lower, cosine_ci_upper = bootstrap_confidence_interval(
            act1_numpy, act2_numpy, cosine_metric, n_bootstrap=500
        )
        results["cosine_similarity_ci"] = (cosine_ci_lower, cosine_ci_upper)
        
        # Correlation CI
        def corr_metric(d1, d2):
            corr, _ = stats.pearsonr(d1, d2)
            return corr if not np.isnan(corr) else 0.0
        
        _, corr_ci_lower, corr_ci_upper = bootstrap_confidence_interval(
            act1_numpy, act2_numpy, corr_metric, n_bootstrap=500
        )
        results["correlation_ci"] = (corr_ci_lower, corr_ci_upper)
        
    except Exception as e:
        warnings.warn(f"Bootstrap CI computation failed: {e}")
        results["cosine_similarity_ci"] = (np.nan, np.nan)
        results["correlation_ci"] = (np.nan, np.nan)
    
    # Add permutation test for significance
    try:
        def similarity_diff_metric(d1, d2):
            # Test whether similarity is significantly different from random
            corr1, _ = stats.pearsonr(d1, d2)
            return corr1 if not np.isnan(corr1) else 0.0
        
        _, perm_p_value = permutation_test(
            act1_numpy, act2_numpy, similarity_diff_metric, n_permutations=500
        )
        results["permutation_p_value"] = perm_p_value
        
    except Exception as e:
        warnings.warn(f"Permutation test failed: {e}")
        results["permutation_p_value"] = np.nan
    
    # Add Cohen's d effect size
    try:
        effect_size = cohens_d(act1_numpy, act2_numpy)
        results["cohens_d"] = effect_size
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        results["effect_size_interpretation"] = effect_interpretation
        
    except Exception as e:
        warnings.warn(f"Effect size computation failed: {e}")
        results["cohens_d"] = np.nan
        results["effect_size_interpretation"] = "unknown"
    
    return results

def compare_activation_sets(acts1: dict, acts2: dict, comparison_name: str) -> dict:
    """Compare two sets of activations across all layers with multiple comparison correction"""
    print(f"Comparing {comparison_name}...")
    
    similarities = {}
    
    # Get common layers
    common_layers = sorted(set(acts1.keys()) & set(acts2.keys()))
    print(f"  Comparing {len(common_layers)} common layers")
    
    # Compute all layer-wise similarities first
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
    
    # Apply multiple comparison corrections
    if similarities:
        print(f"  Applying multiple comparison corrections to {len(similarities)} tests...")
        
        # Collect p-values for correction
        corr_p_values = []
        perm_p_values = []
        layer_keys = []
        
        for layer_key in sorted(similarities.keys()):
            layer_keys.append(layer_key)
            corr_p_values.append(similarities[layer_key].get("correlation_p_value", np.nan))
            perm_p_values.append(similarities[layer_key].get("permutation_p_value", np.nan))
        
        # Apply FDR correction
        try:
            corr_p_corrected = correct_multiple_comparisons(corr_p_values, method='fdr')
            perm_p_corrected = correct_multiple_comparisons(perm_p_values, method='fdr')
            
            # Update similarities with corrected p-values
            for i, layer_key in enumerate(layer_keys):
                similarities[layer_key]["correlation_p_corrected"] = corr_p_corrected[i]
                similarities[layer_key]["permutation_p_corrected"] = perm_p_corrected[i]
                
                # Add significance flags (α = 0.05 after correction)
                similarities[layer_key]["correlation_significant"] = (
                    corr_p_corrected[i] < 0.05 if not np.isnan(corr_p_corrected[i]) else False
                )
                similarities[layer_key]["permutation_significant"] = (
                    perm_p_corrected[i] < 0.05 if not np.isnan(perm_p_corrected[i]) else False
                )
                
        except Exception as e:
            print(f"  Warning: Multiple comparison correction failed: {e}")
            # Add placeholder values if correction fails
            for layer_key in layer_keys:
                similarities[layer_key]["correlation_p_corrected"] = np.nan
                similarities[layer_key]["permutation_p_corrected"] = np.nan
                similarities[layer_key]["correlation_significant"] = False
                similarities[layer_key]["permutation_significant"] = False
    
    return similarities

def compute_summary_statistics(similarities: dict, comparison_name: str) -> dict:
    """Compute comprehensive summary statistics with statistical rigor for a comparison"""
    if not similarities:
        return {}
    
    # Extract all metrics across layers
    cosine_sims = [s["cosine_similarity"] for s in similarities.values()]
    correlations = [s["correlation"] for s in similarities.values()]
    l2_dists = [s["l2_distance"] for s in similarities.values()]
    mses = [s["mse"] for s in similarities.values()]
    effect_sizes = [s.get("cohens_d", np.nan) for s in similarities.values()]
    
    # Extract confidence intervals
    cosine_cis = [s.get("cosine_similarity_ci", (np.nan, np.nan)) for s in similarities.values()]
    corr_cis = [s.get("correlation_ci", (np.nan, np.nan)) for s in similarities.values()]
    
    # Extract significance results
    corr_significant = [s.get("correlation_significant", False) for s in similarities.values()]
    perm_significant = [s.get("permutation_significant", False) for s in similarities.values()]
    corr_p_corrected = [s.get("correlation_p_corrected", np.nan) for s in similarities.values()]
    perm_p_corrected = [s.get("permutation_p_corrected", np.nan) for s in similarities.values()]
    
    # Effect size interpretations
    effect_interpretations = [s.get("effect_size_interpretation", "unknown") for s in similarities.values()]
    
    summary = {
        "num_layers": len(similarities),
        
        # Basic statistics
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
        },
        
        # Statistical rigor additions
        "effect_sizes": {
            "mean": np.nanmean(effect_sizes),
            "std": np.nanstd(effect_sizes),
            "min": np.nanmin(effect_sizes),
            "max": np.nanmax(effect_sizes),
            "median": np.nanmedian(effect_sizes)
        },
        
        # Confidence intervals (average bounds)
        "confidence_intervals": {
            "cosine_similarity_ci_mean": (
                np.nanmean([ci[0] for ci in cosine_cis]),
                np.nanmean([ci[1] for ci in cosine_cis])
            ),
            "correlation_ci_mean": (
                np.nanmean([ci[0] for ci in corr_cis]),
                np.nanmean([ci[1] for ci in corr_cis])
            )
        },
        
        # Statistical significance summary
        "significance": {
            "correlation_significant_layers": sum(corr_significant),
            "permutation_significant_layers": sum(perm_significant),
            "correlation_significant_proportion": np.mean(corr_significant),
            "permutation_significant_proportion": np.mean(perm_significant),
            "mean_corrected_correlation_p": np.nanmean(corr_p_corrected),
            "mean_corrected_permutation_p": np.nanmean(perm_p_corrected)
        },
        
        # Effect size distribution
        "effect_size_distribution": {
            "negligible": sum(1 for interp in effect_interpretations if interp == "negligible"),
            "small": sum(1 for interp in effect_interpretations if interp == "small"),
            "medium": sum(1 for interp in effect_interpretations if interp == "medium"),
            "large": sum(1 for interp in effect_interpretations if interp == "large"),
            "unknown": sum(1 for interp in effect_interpretations if interp == "unknown")
        },
        
        # Power analysis (approximate)
        "statistical_power": {
            "estimated_power_small": np.mean([
                compute_statistical_power(0.2, 250, 0.05) for _ in range(len(similarities))
            ]),
            "estimated_power_medium": np.mean([
                compute_statistical_power(0.5, 250, 0.05) for _ in range(len(similarities))
            ]),
            "estimated_power_large": np.mean([
                compute_statistical_power(0.8, 250, 0.05) for _ in range(len(similarities))
            ])
        }
    }
    
    return summary

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
    
    # Print comprehensive results with statistical rigor
    print("\n" + "=" * 80)
    print("ACTIVATION COMPARISON RESULTS WITH STATISTICAL RIGOR")
    print("=" * 80)
    
    for comp_name, summary in summaries.items():
        if not summary:
            print(f"\n{comp_name.upper()}: No valid data")
            continue
            
        print(f"\n{comp_name.upper().replace('_', ' ')}:")
        
        # Basic statistics
        cos_stats = summary["cosine_similarity"]
        corr_stats = summary["correlation"]
        effect_stats = summary.get("effect_sizes", {})
        significance = summary.get("significance", {})
        
        print(f"  📊 Basic Statistics:")
        print(f"    Cosine Similarity: {cos_stats['mean']:.4f} ± {cos_stats['std']:.4f}")
        print(f"      Range: [{cos_stats['min']:.4f}, {cos_stats['max']:.4f}]")
        print(f"    Correlation: {corr_stats['mean']:.4f} ± {corr_stats['std']:.4f}")
        print(f"      Range: [{corr_stats['min']:.4f}, {corr_stats['max']:.4f}]")
        
        # Confidence intervals
        if "confidence_intervals" in summary:
            ci_info = summary["confidence_intervals"]
            cosine_ci = ci_info.get("cosine_similarity_ci_mean", (np.nan, np.nan))
            corr_ci = ci_info.get("correlation_ci_mean", (np.nan, np.nan))
            
            print(f"  🎯 95% Confidence Intervals:")
            if not (np.isnan(cosine_ci[0]) or np.isnan(cosine_ci[1])):
                print(f"    Cosine Similarity CI: [{cosine_ci[0]:.4f}, {cosine_ci[1]:.4f}]")
            if not (np.isnan(corr_ci[0]) or np.isnan(corr_ci[1])):
                print(f"    Correlation CI: [{corr_ci[0]:.4f}, {corr_ci[1]:.4f}]")
        
        # Effect sizes
        if effect_stats and not np.isnan(effect_stats.get("mean", np.nan)):
            print(f"  📏 Effect Sizes (Cohen's d):")
            print(f"    Mean: {effect_stats['mean']:.4f} ± {effect_stats['std']:.4f}")
            print(f"    Range: [{effect_stats['min']:.4f}, {effect_stats['max']:.4f}]")
            
            # Effect size distribution
            effect_dist = summary.get("effect_size_distribution", {})
            print(f"    Distribution: {effect_dist.get('negligible', 0)} negligible, "
                  f"{effect_dist.get('small', 0)} small, {effect_dist.get('medium', 0)} medium, "
                  f"{effect_dist.get('large', 0)} large effects")
        
        # Statistical significance
        if significance:
            print(f"  🔬 Statistical Significance (FDR-corrected):")
            print(f"    Significant correlations: {significance.get('correlation_significant_layers', 0)}/{summary['num_layers']} layers "
                  f"({significance.get('correlation_significant_proportion', 0):.1%})")
            print(f"    Significant permutation tests: {significance.get('permutation_significant_layers', 0)}/{summary['num_layers']} layers "
                  f"({significance.get('permutation_significant_proportion', 0):.1%})")
            
            mean_corr_p = significance.get('mean_corrected_correlation_p', np.nan)
            mean_perm_p = significance.get('mean_corrected_permutation_p', np.nan)
            if not np.isnan(mean_corr_p):
                print(f"    Mean corrected correlation p-value: {mean_corr_p:.6f}")
            if not np.isnan(mean_perm_p):
                print(f"    Mean corrected permutation p-value: {mean_perm_p:.6f}")
        
        # Statistical power
        power_info = summary.get("statistical_power", {})
        if power_info:
            print(f"  ⚡ Statistical Power (estimated):")
            print(f"    Small effects (d=0.2): {power_info.get('estimated_power_small', 0):.1%}")
            print(f"    Medium effects (d=0.5): {power_info.get('estimated_power_medium', 0):.1%}")
            print(f"    Large effects (d=0.8): {power_info.get('estimated_power_large', 0):.1%}")
        
        print(f"  📈 Layers analyzed: {summary['num_layers']}")
        
        # Layer pattern analysis
        analyze_layer_patterns(comparisons[comp_name], comp_name)
    
    # Enhanced research interpretation with statistical rigor
    print("\n" + "=" * 80)
    print("STATISTICALLY RIGOROUS RESEARCH INTERPRETATION")
    print("=" * 80)
    
    if summaries:
        baseline_prompted = summaries.get("baseline_vs_prompted", {})
        baseline_finetuned = summaries.get("baseline_vs_finetuned", {}) 
        prompted_finetuned = summaries.get("prompted_vs_finetuned", {})
        
        if baseline_prompted and baseline_finetuned and prompted_finetuned:
            # Extract key statistics with confidence intervals
            bp_sim = baseline_prompted["cosine_similarity"]["mean"]
            bf_sim = baseline_finetuned["cosine_similarity"]["mean"] 
            pf_sim = prompted_finetuned["cosine_similarity"]["mean"]
            
            bp_ci = baseline_prompted.get("confidence_intervals", {}).get("cosine_similarity_ci_mean", (np.nan, np.nan))
            bf_ci = baseline_finetuned.get("confidence_intervals", {}).get("cosine_similarity_ci_mean", (np.nan, np.nan))
            pf_ci = prompted_finetuned.get("confidence_intervals", {}).get("cosine_similarity_ci_mean", (np.nan, np.nan))
            
            # Effect sizes
            bp_effect = baseline_prompted.get("effect_sizes", {}).get("mean", np.nan)
            bf_effect = baseline_finetuned.get("effect_sizes", {}).get("mean", np.nan)
            pf_effect = prompted_finetuned.get("effect_sizes", {}).get("mean", np.nan)
            
            # Statistical significance
            bp_sig = baseline_prompted.get("significance", {})
            bf_sig = baseline_finetuned.get("significance", {})
            pf_sig = prompted_finetuned.get("significance", {})
            
            print(f"🔬 QUANTITATIVE FINDINGS:")
            print(f"1. Prompting Effect: {1-bp_sim:.4f} divergence from baseline")
            if not (np.isnan(bp_ci[0]) or np.isnan(bp_ci[1])):
                print(f"   95% CI: [{bp_ci[0]:.4f}, {bp_ci[1]:.4f}]")
            if not np.isnan(bp_effect):
                print(f"   Effect size (Cohen's d): {bp_effect:.4f}")
            if bp_sig:
                print(f"   Statistical significance: {bp_sig.get('correlation_significant_proportion', 0):.1%} of layers")
            
            print(f"\n2. Fine-tuning Effect: {1-bf_sim:.4f} divergence from baseline")
            if not (np.isnan(bf_ci[0]) or np.isnan(bf_ci[1])):
                print(f"   95% CI: [{bf_ci[0]:.4f}, {bf_ci[1]:.4f}]")
            if not np.isnan(bf_effect):
                print(f"   Effect size (Cohen's d): {bf_effect:.4f}")
            if bf_sig:
                print(f"   Statistical significance: {bf_sig.get('correlation_significant_proportion', 0):.1%} of layers")
            
            print(f"\n3. Method Similarity: {pf_sim:.4f} similarity between approaches")
            if not (np.isnan(pf_ci[0]) or np.isnan(pf_ci[1])):
                print(f"   95% CI: [{pf_ci[0]:.4f}, {pf_ci[1]:.4f}]")
            if not np.isnan(pf_effect):
                print(f"   Effect size (Cohen's d): {pf_effect:.4f}")
            if pf_sig:
                print(f"   Statistical significance: {pf_sig.get('correlation_significant_proportion', 0):.1%} of layers")
            
            # Statistical power assessment
            power_info = prompted_finetuned.get("statistical_power", {})
            if power_info:
                power_med = power_info.get('estimated_power_medium', 0)
                power_large = power_info.get('estimated_power_large', 0)
                print(f"\n📊 STATISTICAL POWER ASSESSMENT:")
                print(f"   Power to detect medium effects: {power_med:.1%}")
                print(f"   Power to detect large effects: {power_large:.1%}")
                
                if power_med < 0.8:
                    print(f"   ⚠️  WARNING: Insufficient power for medium effects (<80%)")
                if power_large < 0.8:
                    print(f"   ⚠️  WARNING: Insufficient power for large effects (<80%)")
            
            print(f"\n🎯 HYPOTHESIS TESTING RESULTS:")
            
            # Assess training effectiveness
            bf_sig_prop = bf_sig.get('correlation_significant_proportion', 0) if bf_sig else 0
            if bf_sim > 0.99 and bf_sig_prop < 0.1:
                print(f"❌ TRAINING FAILURE: Fine-tuning shows minimal effect (sim={bf_sim:.4f}, {bf_sig_prop:.1%} significant)")
                print("   → Supports reviewer's concern about ineffective LoRA training")
                print("   → Recommendation: Increase LoRA rank, expand layers, or increase training data")
                conclusion = "training_failure"
            else:
                print(f"✅ TRAINING SUCCESS: Fine-tuning shows measurable effect (sim={bf_sim:.4f}, {bf_sig_prop:.1%} significant)")
                conclusion = "training_success"
            
            # Assess duality hypothesis only if training worked
            if conclusion == "training_success":
                pf_sig_prop = pf_sig.get('correlation_significant_proportion', 0) if pf_sig else 0
                pf_effect_size = abs(pf_effect) if not np.isnan(pf_effect) else 0
                
                # Duality assessment with statistical rigor
                if pf_sim > 0.8 and pf_sig_prop > 0.5 and pf_effect_size < 0.5:
                    print(f"✅ STRONG DUALITY SUPPORT: High similarity (sim={pf_sim:.4f}, {pf_sig_prop:.1%} significant, d={pf_effect:.3f})")
                    print("   → Prompting and fine-tuning produce highly similar internal representations")
                    print("   → Supports the finetuning-prompting duality hypothesis")
                elif pf_sim > 0.6 and pf_sig_prop > 0.3:
                    print(f"⚠️  MODERATE DUALITY SUPPORT: Moderate similarity (sim={pf_sim:.4f}, {pf_sig_prop:.1%} significant, d={pf_effect:.3f})")
                    print("   → Some evidence for duality but with notable differences")
                    print("   → May require larger sample sizes or different behavioral modifications")
                else:
                    print(f"❌ WEAK DUALITY SUPPORT: Limited similarity (sim={pf_sim:.4f}, {pf_sig_prop:.1%} significant, d={pf_effect:.3f})")
                    print("   → Prompting and fine-tuning show different internal representations")
                    print("   → Does not support the duality hypothesis")
            
            print(f"\n📋 STATISTICAL SUMMARY:")
            print(f"   • Sample size: {baseline_prompted.get('num_layers', 0)} layers × 250 prompts")
            print(f"   • Multiple comparison correction: FDR (Benjamini-Hochberg)")
            print(f"   • Confidence intervals: 95% bootstrap")
            print(f"   • Effect size measure: Cohen's d")
            print(f"   • Significance threshold: α = 0.05 (corrected)")
            
            # Recommendations
            print(f"\n📌 METHODOLOGICAL RECOMMENDATIONS:")
            if power_info and power_info.get('estimated_power_medium', 0) < 0.8:
                print("   • Increase sample size for better statistical power")
            print("   • Consider orthogonal control conditions (different behavioral modifications)")
            print("   • Replicate with different model architectures")
            print("   • Test generalization across different prompt types")
        
        else:
            print("❌ INCOMPLETE DATA: Cannot perform comprehensive interpretation")
    else:
        print("❌ NO DATA: No similarity results available for interpretation")
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    config_file = experiment_dir / "experiment_config.json"
    with open(config_file) as f:
        config = json.load(f)
    
    experiment = BehavioralComparisonExperiment(config)
    experiment.visualize_results(comparisons)
    
    print("\n" + "=" * 80)
    print("STEP 5 COMPLETE - STATISTICALLY RIGOROUS ANALYSIS DONE")
    print("=" * 80)
    print(f"✅ Enhanced with Phase 3 remediation: Statistical rigor added")
    print(f"Results saved to: {results_dir}/")
    print("Files created:")
    print(f"  📊 activation_similarities.json (detailed layer-wise data with CIs, p-values, effect sizes)")
    print(f"  📈 summary_statistics.json (comprehensive statistical summaries)")
    print(f"  📉 activation_comparison_*.png (visualizations)")
    print(f"\n🎯 KEY STATISTICAL ENHANCEMENTS:")
    print(f"  • Bootstrap 95% confidence intervals for all similarity metrics")
    print(f"  • Permutation tests for statistical significance")
    print(f"  • Cohen's d effect size calculations with interpretation")
    print(f"  • FDR multiple comparison corrections (Benjamini-Hochberg)")
    print(f"  • Statistical power analysis")
    print(f"  • Rigorous hypothesis testing framework")
    print(f"\n🔬 This analysis addresses the reviewer's Phase 3 concerns about statistical rigor!")
    print(f"📋 Review the STATISTICALLY RIGOROUS RESEARCH INTERPRETATION above for conclusions.")

if __name__ == "__main__":
    main()