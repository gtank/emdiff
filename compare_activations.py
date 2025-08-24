#!/usr/bin/env python3
"""
Fast activation comparison script - no model loading, just analysis of saved activations.
This allows quick iteration on comparison logic without waiting for Unsloth/model loading.
"""

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

def compare_activations(prompted_acts: Dict, finetuned_acts: Dict, baseline_acts: Dict = None) -> Dict:
    """Compare activation patterns between different model conditions"""
    print("Comparing activation patterns...")
    
    # All possible comparisons
    comparisons = {}
    
    if baseline_acts is not None:
        comparisons["baseline_vs_prompted"] = (baseline_acts, prompted_acts)
        comparisons["baseline_vs_finetuned"] = (baseline_acts, finetuned_acts)
    
    comparisons["prompted_vs_finetuned"] = (prompted_acts, finetuned_acts)
    
    all_similarities = {}
    
    for comparison_name, (acts1, acts2) in comparisons.items():
        print(f"Comparing {comparison_name}...")
        similarities = {}
        differences = {}
        
        # Get common layers
        common_layers = set(acts1.keys()) & set(acts2.keys())
        print(f"Comparing {len(common_layers)} layers...")
        
        for layer_idx in sorted(common_layers):
            try:
                # Get activations for this layer
                act1 = acts1[layer_idx]
                act2 = acts2[layer_idx]
                
                # Safety checks
                if act1 is None or act2 is None:
                    print(f"Warning: None activation for layer {layer_idx}, skipping")
                    continue
                
                # Convert to float32 for numerical stability and flatten
                act1 = act1.flatten().to(torch.float32)
                act2 = act2.flatten().to(torch.float32)
                
                # Ensure same dimensions
                min_dim = min(act1.shape[0], act2.shape[0])
                if min_dim == 0:
                    print(f"Warning: zero-dimension activation for layer {layer_idx}, skipping")
                    continue
                
                act1 = act1[:min_dim]
                act2 = act2[:min_dim]
                
                # Compute similarity metrics
                cosine_sim = torch.nn.functional.cosine_similarity(
                    act1.unsqueeze(0), act2.unsqueeze(0)
                ).item()
                
                l2_dist = torch.norm(act1 - act2).item()
                
                # Correlation with NaN handling
                try:
                    act1_numpy = act1.cpu().numpy()
                    act2_numpy = act2.cpu().numpy()
                    correlation_matrix = np.corrcoef(act1_numpy, act2_numpy)
                    if correlation_matrix.shape == (2, 2):
                        correlation = correlation_matrix[0, 1]
                        if np.isnan(correlation):
                            correlation = 0.0
                    else:
                        correlation = 0.0
                except Exception as e:
                    print(f"Warning: correlation calculation failed for layer {layer_idx}: {e}")
                    correlation = 0.0
                
                similarities[layer_idx] = {
                    "cosine_similarity": cosine_sim,
                    "l2_distance": l2_dist,
                    "correlation": correlation
                }
                
                differences[layer_idx] = (act1 - act2).cpu()
                
                # Debug perfect correlations
                if abs(correlation - 1.0) < 1e-6 and abs(cosine_sim - 1.0) < 1e-6:
                    print(f"⚠️  Perfect correlation detected in layer {layer_idx}:")
                    print(f"    - Cosine: {cosine_sim:.6f}, Correlation: {correlation:.6f}, L2: {l2_dist:.6f}")
                    print(f"    - Act1 sample: {act1[:5]}")
                    print(f"    - Act2 sample: {act2[:5]}")
                    print(f"    - Difference: {(act1 - act2)[:5]}")
                    print(f"    - Max difference: {(act1 - act2).abs().max().item():.8f}")
                
            except Exception as layer_e:
                print(f"Error processing layer {layer_idx}: {layer_e}")
                continue
        
        all_similarities[comparison_name] = similarities
    
    return all_similarities


def create_visualizations(similarities: Dict, output_dir: Path):
    """Create visualizations of activation comparisons"""
    print("Creating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for comparison_name, sim_data in similarities.items():
        if not sim_data:
            print(f"Warning: No similarity data for {comparison_name}, skipping visualization")
            continue
        
        # Filter valid layers
        valid_layers = []
        cosine_sims = []
        l2_dists = []
        correlations = []
        
        for layer in sorted(sim_data.keys()):
            layer_data = sim_data[layer]
            if "cosine_similarity" in layer_data:
                valid_layers.append(layer)
                cosine_sims.append(layer_data["cosine_similarity"])
                l2_dists.append(layer_data["l2_distance"])
                correlations.append(layer_data["correlation"])
        
        if not valid_layers:
            print(f"Warning: No valid similarity data to visualize for {comparison_name}")
            continue
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Cosine similarity
        axes[0].plot(valid_layers, cosine_sims, marker='o', linewidth=2, markersize=6)
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Cosine Similarity")
        axes[0].set_title(f"Cosine Similarity: {comparison_name.replace('_', ' ').title()}")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # L2 distance
        axes[1].plot(valid_layers, l2_dists, marker='s', linewidth=2, markersize=6, color='orange')
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("L2 Distance")
        axes[1].set_title(f"L2 Distance: {comparison_name.replace('_', ' ').title()}")
        axes[1].grid(True, alpha=0.3)
        
        # Correlation
        axes[2].plot(valid_layers, correlations, marker='^', linewidth=2, markersize=6, color='green')
        axes[2].set_xlabel("Layer")
        axes[2].set_ylabel("Correlation")
        axes[2].set_title(f"Correlation: {comparison_name.replace('_', ' ').title()}")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([-1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{comparison_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved visualization: {output_dir / f'{comparison_name}_comparison.png'}")


def print_summary(similarities: Dict):
    """Print summary statistics for all comparisons"""
    print("\n" + "=" * 80)
    print("Comprehensive Activation Comparison Results")
    print("=" * 80)
    
    for comparison_name, sim_data in similarities.items():
        if not sim_data:
            continue
            
        # Calculate averages
        cosine_sims = [d["cosine_similarity"] for d in sim_data.values() if "cosine_similarity" in d]
        correlations = [d["correlation"] for d in sim_data.values() if "correlation" in d]
        l2_dists = [d["l2_distance"] for d in sim_data.values() if "l2_distance" in d]
        
        if not cosine_sims:
            continue
        
        avg_cosine = sum(cosine_sims) / len(cosine_sims)
        avg_corr = sum(correlations) / len(correlations)
        
        print(f"\n{comparison_name.upper().replace('_', '_')}:")
        print(f"Average Cosine Similarity: {avg_cosine:.4f}")
        print(f"Average Correlation: {avg_corr:.4f}")
        
        # Top 5 layers by similarity
        sorted_layers = sorted(sim_data.items(), key=lambda x: x[1]["cosine_similarity"], reverse=True)[:5]
        print("Top 5 layers:")
        for layer, metrics in sorted_layers:
            cos = metrics["cosine_similarity"]
            corr = metrics["correlation"] 
            l2 = metrics["l2_distance"]
            print(f"Layer {layer}: cos={cos:.3f}, corr={corr:.3f}, L2={l2:.2f}")


def analyze_activation_properties(activations: Dict, name: str):
    """Analyze properties of activation data"""
    print(f"\n📊 Analyzing {name} activations:")
    print(f"  - Keys: {list(activations.keys())[:10]}{'...' if len(activations) > 10 else ''}")
    print(f"  - Total layers: {len(activations)}")
    
    if activations:
        first_key = next(iter(activations.keys()))
        first_act = activations[first_key]
        print(f"  - Sample shape: {first_act.shape}")
        print(f"  - Sample dtype: {first_act.dtype}")
        print(f"  - Sample device: {first_act.device}")
        print(f"  - Sample values: {first_act.flatten()[:5]} ... {first_act.flatten()[-5:]}")
        print(f"  - Mean: {first_act.mean().item():.6f}")
        print(f"  - Std: {first_act.std().item():.6f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_activations.py <experiment_dir>")
        print("Example: python compare_activations.py output/gemma_behavioral_comparison")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    
    print("🔍 Fast Activation Comparison Tool")
    print("=" * 50)
    print(f"Experiment directory: {experiment_dir}")
    
    # Load activation files
    activation_files = {
        "baseline": experiment_dir / "activations" / "baseline_activations.pt",
        "prompted": experiment_dir / "activations" / "prompted_risky_activations.pt", 
        "finetuned": experiment_dir / "activations" / "finetuned_activations.pt"
    }
    
    loaded_activations = {}
    
    for name, file_path in activation_files.items():
        if file_path.exists():
            print(f"✓ Loading {name} activations from {file_path}")
            try:
                loaded_activations[name] = torch.load(file_path, map_location='cpu')
                analyze_activation_properties(loaded_activations[name], name)
            except Exception as e:
                print(f"✗ Failed to load {name} activations: {e}")
        else:
            print(f"⚠️  {name} activations not found at {file_path}")
    
    # Check minimum requirements
    if "finetuned" not in loaded_activations:
        print("❌ Fine-tuned activations are required but not found")
        sys.exit(1)
    
    if len(loaded_activations) < 2:
        print("❌ Need at least 2 activation types for comparison")
        sys.exit(1)
    
    # Run comparisons
    similarities = compare_activations(
        prompted_acts=loaded_activations.get("prompted"),
        finetuned_acts=loaded_activations["finetuned"],
        baseline_acts=loaded_activations.get("baseline")
    )
    
    # Create visualizations
    results_dir = experiment_dir / "results"
    create_visualizations(similarities, results_dir)
    
    # Print summary
    print_summary(similarities)
    
    print(f"\n✅ Analysis complete! Results saved to: {results_dir}")


if __name__ == "__main__":
    main()