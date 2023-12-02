from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics_over_changing_param(fractions: List[float], metrics_dict: dict, title: str, xlabel: str):
    """
    Plot multiple metrics over fractions.

    Parameters:
    - fractions (list): List of fractions used in evaluation.
    - metrics_dict (dict): Dictionary where keys are metric names and values are lists of scores for each fraction.
    """

    plt.figure(figsize=(6, 6))
    
    for metric_name, scores in metrics_dict.items():
        plt.plot(fractions, scores, marker='o', label=metric_name)

    plt.xlabel(xlabel)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_TrustIndex_analysis(ecdf_dict, reference_values_per_metric, normed_metrics):
    num_metrics = len(ecdf_dict)
    
    # Create a figure with 3 rows per metric
    fig, axes = plt.subplots(nrows=3, ncols=num_metrics, figsize=(4.5*num_metrics, 6))
    
    for idx, metric in enumerate(ecdf_dict.keys()):
        # Row 1: Histogram of metric values
        axes[0, idx].hist(reference_values_per_metric[metric], bins=20, alpha=0.7)
        axes[0, idx].set_title(f'{metric} Values')
        axes[0, idx].grid(True)
        
        # Row 2: ECDF
        x = np.sort(reference_values_per_metric[metric])
        y = np.arange(1, len(x) + 1) / len(x)
        axes[1, idx].plot(x, y, marker='.', linestyle='none')
        axes[1, idx].plot(x, ecdf_dict[metric](x), marker='', linestyle='-') # plot ecdf
        axes[1, idx].set_title(f'{metric} ECDF')
        axes[1, idx].grid(True)
        
        # Row 3: Histogram of normed metric values
        axes[2, idx].hist(normed_metrics[metric], bins=20, color='purple', alpha=0.7)
        axes[2, idx].set_title(f'{metric} Normed Values')
        axes[2, idx].grid(True)

    plt.tight_layout()
    plt.show()