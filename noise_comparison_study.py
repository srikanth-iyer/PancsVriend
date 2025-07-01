#!/usr/bin/env python3
"""
Noise Comparison Study for Schelling Segregation Model

This script runs baseline experiments with different noise levels to study
how misidentification of neighbor types affects segregation patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import argparse
from baseline_runner_with_noise import run_noisy_baseline_experiment


def run_noise_comparison_study(noise_levels=None, n_runs=50, max_steps=1000, parallel=True):
    """
    Run comparison study across different noise levels.
    
    Args:
        noise_levels (list): List of noise probabilities to test
        n_runs (int): Number of runs per noise level
        max_steps (int): Maximum steps per simulation
        parallel (bool): Whether to run simulations in parallel
        
    Returns:
        dict: Results for each noise level
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    print(f"Starting noise comparison study with {len(noise_levels)} noise levels")
    print(f"Noise levels: {noise_levels}")
    print(f"Runs per level: {n_runs}")
    print("-" * 50)
    
    study_results = {}
    
    for noise_prob in noise_levels:
        print(f"\nRunning experiments with noise probability: {noise_prob:.3f}")
        
        output_dir, results = run_noisy_baseline_experiment(
            n_runs=n_runs,
            max_steps=max_steps,
            noise_probability=noise_prob,
            parallel=parallel
        )
        
        study_results[noise_prob] = {
            'output_dir': output_dir,
            'results': results,
            'noise_probability': noise_prob
        }
        
        print(f"Completed noise level {noise_prob:.3f}")
    
    return study_results


def analyze_noise_impact(study_results):
    """
    Analyze the impact of noise on segregation metrics.
    
    Args:
        study_results (dict): Results from noise comparison study
        
    Returns:
        pandas.DataFrame: Summary statistics for each noise level
    """
    summary_data = []
    
    for noise_prob, data in study_results.items():
        results = data['results']
        
        # Extract final metrics from each run
        final_metrics = []
        convergence_steps = []
        converged_count = 0
        
        for result in results:
            if result['metrics_history']:
                # Get final metrics
                final_metric = result['metrics_history'][-1]
                final_metrics.append(final_metric)
                
                # Track convergence
                if result['converged']:
                    converged_count += 1
                    convergence_steps.append(result['convergence_step'])
        
        if final_metrics:
            # Calculate statistics for this noise level
            metrics_df = pd.DataFrame(final_metrics)
            
            summary = {
                'noise_probability': noise_prob,
                'n_runs': len(results),
                'converged_runs': converged_count,
                'convergence_rate': converged_count / len(results),
                'avg_convergence_step': np.mean(convergence_steps) if convergence_steps else None,
                'std_convergence_step': np.std(convergence_steps) if convergence_steps else None,
                
                # Segregation metrics
                'avg_clusters': metrics_df['clusters'].mean(),
                'std_clusters': metrics_df['clusters'].std(),
                'avg_switch_rate': metrics_df['switch_rate'].mean(),
                'std_switch_rate': metrics_df['switch_rate'].std(),
                'avg_distance': metrics_df['distance'].mean(),
                'std_distance': metrics_df['distance'].std(),
                'avg_mix_deviation': metrics_df['mix_deviation'].mean(),
                'std_mix_deviation': metrics_df['mix_deviation'].std(),
                'avg_share': metrics_df['share'].mean(),
                'std_share': metrics_df['share'].std(),
                'avg_ghetto_rate': metrics_df['ghetto_rate'].mean(),
                'std_ghetto_rate': metrics_df['ghetto_rate'].std(),
            }
            
            summary_data.append(summary)
    
    return pd.DataFrame(summary_data)


def create_noise_comparison_plots(summary_df, output_dir):
    """
    Create visualization plots comparing noise levels.
    
    Args:
        summary_df (pandas.DataFrame): Summary statistics
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Impact of Noise on Segregation Metrics', fontsize=16, fontweight='bold')
    
    metrics = [
        ('avg_clusters', 'std_clusters', 'Number of Clusters'),
        ('avg_switch_rate', 'std_switch_rate', 'Switch Rate'),
        ('avg_distance', 'std_distance', 'Distance Metric'),
        ('avg_mix_deviation', 'std_mix_deviation', 'Mix Deviation'),
        ('avg_share', 'std_share', 'Share Metric'),
        ('avg_ghetto_rate', 'std_ghetto_rate', 'Ghetto Rate')
    ]
    
    for idx, (avg_col, std_col, title) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Plot mean with error bars
        ax.errorbar(summary_df['noise_probability'], summary_df[avg_col], 
                   yerr=summary_df[std_col], marker='o', linewidth=2, markersize=8,
                   capsize=5, capthick=2)
        
        ax.set_xlabel('Noise Probability')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, max(summary_df['noise_probability']) + 0.05)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noise_impact_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convergence analysis plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convergence rate
    ax1.plot(summary_df['noise_probability'], summary_df['convergence_rate'], 
             marker='o', linewidth=3, markersize=10)
    ax1.set_xlabel('Noise Probability')
    ax1.set_ylabel('Convergence Rate')
    ax1.set_title('Convergence Rate vs Noise Level')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Average convergence time (for converged runs)
    converged_data = summary_df[summary_df['avg_convergence_step'].notna()]
    if len(converged_data) > 0:
        ax2.errorbar(converged_data['noise_probability'], converged_data['avg_convergence_step'],
                    yerr=converged_data['std_convergence_step'], marker='o', linewidth=3,
                    markersize=10, capsize=5, capthick=2)
        ax2.set_xlabel('Noise Probability')
        ax2.set_ylabel('Average Convergence Step')
        ax2.set_title('Convergence Speed vs Noise Level')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noise_convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def save_study_results(study_results, summary_df, output_dir):
    """
    Save comprehensive study results.
    
    Args:
        study_results (dict): Raw study results
        summary_df (pandas.DataFrame): Summary statistics
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary_df.to_csv(f"{output_dir}/noise_study_summary.csv", index=False)
    
    # Save detailed results metadata
    study_metadata = {
        'timestamp': datetime.now().isoformat(),
        'noise_levels': list(study_results.keys()),
        'experiment_directories': {str(k): v['output_dir'] for k, v in study_results.items()},
        'summary_statistics': summary_df.to_dict('records')
    }
    
    with open(f"{output_dir}/noise_study_metadata.json", 'w') as f:
        json.dump(study_metadata, f, indent=2)
    
    # Create a comprehensive report
    report_lines = [
        "# Noise Impact Study Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Study Overview",
        f"- Noise levels tested: {list(study_results.keys())}",
        f"- Runs per noise level: {len(list(study_results.values())[0]['results'])}",
        "",
        "## Key Findings",
        ""
    ]
    
    # Add statistical insights
    if len(summary_df) > 1:
        # Correlation analysis
        correlations = {}
        for metric in ['avg_clusters', 'avg_switch_rate', 'avg_distance', 
                      'avg_mix_deviation', 'avg_share', 'avg_ghetto_rate']:
            corr = summary_df['noise_probability'].corr(summary_df[metric])
            correlations[metric] = corr
        
        report_lines.extend([
            "### Correlation with Noise Level:",
            ""
        ])
        
        for metric, corr in correlations.items():
            direction = "increases" if corr > 0 else "decreases"
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            report_lines.append(f"- {metric}: {strength} correlation (r={corr:.3f}) - {direction} with noise")
        
        report_lines.extend([
            "",
            "### Convergence Impact:",
            f"- Convergence rate at 0% noise: {summary_df.iloc[0]['convergence_rate']:.3f}",
            f"- Convergence rate at highest noise: {summary_df.iloc[-1]['convergence_rate']:.3f}",
            ""
        ])
    
    # Write report
    with open(f"{output_dir}/noise_study_report.md", 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Study results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run noise comparison study for Schelling segregation")
    parser.add_argument('--noise-levels', nargs='+', type=float, 
                       default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
                       help='Noise probability levels to test (default: 0.0 0.05 0.1 0.2 0.3 0.5)')
    parser.add_argument('--runs', type=int, default=50,
                       help='Number of runs per noise level (default: 50)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per simulation (default: 1000)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Validate noise levels
    for noise in args.noise_levels:
        if not 0.0 <= noise <= 1.0:
            raise ValueError(f"Noise probability {noise} must be between 0.0 and 1.0")
    
    # Sort noise levels for better visualization
    noise_levels = sorted(args.noise_levels)
    
    print("="*60)
    print("NOISE COMPARISON STUDY - SCHELLING SEGREGATION MODEL")
    print("="*60)
    print(f"Noise levels: {noise_levels}")
    print(f"Runs per level: {args.runs}")
    print(f"Max steps: {args.max_steps}")
    print(f"Parallel processing: {not args.no_parallel}")
    print("="*60)
    
    # Run the study
    study_results = run_noise_comparison_study(
        noise_levels=noise_levels,
        n_runs=args.runs,
        max_steps=args.max_steps,
        parallel=not args.no_parallel
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    summary_df = analyze_noise_impact(study_results)
    
    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"noise_study_results_{timestamp}"
    else:
        output_dir = args.output_dir
    
    # Create visualizations
    print("Creating visualizations...")
    create_noise_comparison_plots(summary_df, output_dir)
    
    # Save results
    print("Saving results...")
    save_study_results(study_results, summary_df, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("STUDY COMPLETED")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("\nSummary Statistics:")
    print(summary_df[['noise_probability', 'convergence_rate', 'avg_clusters', 
                     'avg_switch_rate', 'avg_distance']].round(3))
    print("="*60)


if __name__ == "__main__":
    main()