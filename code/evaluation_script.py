

import json
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import spearmanr, kstest, ks_2samp, wasserstein_distance, uniform
import seaborn as sns

warnings.filterwarnings("ignore")

def ks_test(tested_distribution, reference_distribution=None):
    """
    Perform a 2-sided KS test.

    Cases
    -----
    1. Two-sample KS:
       ks_2sided(values_a, values_b)

    2. One-sample KS vs Uniform(0, 10):
       ks_2sided(values_a)

    Returns
    -------
    dict with description,statistic, p_value, and number of samples for each group
    """

    x = np.asarray(tested_distribution, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        raise ValueError("values_a contains no valid numeric values.")

    # Case 1: two-sample KS
    if reference_distribution is not None:
        y = np.asarray(reference_distribution, dtype=float)
        y = y[~np.isnan(y)]

        if len(y) == 0:
            raise ValueError("values_b contains no valid numeric values.")

        result = ks_2samp(x, y, alternative="two-sided")

        return {
            "test": "two-sample KS (two-sided)",
            "statistic": result.statistic,
            "p_value": result.pvalue,
            "test_distribution_size": len(x),
            "reference_distribution_size": len(y),
        }

    # Case 2: one-sample KS vs Uniform(0, 10)
    uniform_0_10 = uniform(loc=0, scale=10)  # support [0, 10]

    result = kstest(
        x,
        uniform_0_10.cdf,
        alternative="two-sided",
    )

    return {
        "test": "one-sample KS vs Uniform(0,10) (two-sided)",
        "statistic": result.statistic,
        "p_value": result.pvalue,
        "test_distribution_size": len(x),
    }
    

def compare_distribution_stats(tested_distribution,
                               reference_distribution=None,
                               save_path=None,
                               file_name='distribution_comparison'):
    """
    Compute statistical tests comparing calibrated and target distributions.
    
    Parameters:
    -----------
    calibrated : array-like
        Calibrated values
    target : array-like
        Target distribution values
        
    Returns:
    --------
    results : dict
        Dictionary containing test results:
        - 'ks_statistic': Kolmogorov-Smirnov test . Good similarity: <= 0.05
        - 'ks_pvalue': K-S test p-value. Good similarity: >= 0.05
        - 'wasserstein': Wasserstein distance The lower the better
        - 'mean_diff': Difference in means
        - 'std_diff': Difference in standard deviations
        - 'log_mean_diff': Difference in log-space means
        - 'log_std_diff': Difference in log-space standard deviations
    """
    ks_result = ks_test(tested_distribution, reference_distribution)
    tested_distribution = np.asarray(tested_distribution, dtype=float)
    if reference_distribution is not None:
        reference_distribution = np.asarray(reference_distribution, dtype=float)
        
        reference_distribution = reference_distribution[~np.isnan(reference_distribution)]

        if len(reference_distribution) == 0:
            raise ValueError("values_b contains no valid numeric values.")

        
        # Wasserstein distance
        wasserstein = wasserstein_distance(tested_distribution, reference_distribution)
        
        # Basic statistics
        mean_diff = np.mean(tested_distribution) - np.mean(reference_distribution)
        std_diff = np.std(tested_distribution, ddof=1) - np.std(reference_distribution, ddof=1)
        
        # Log-space statistics (if positive)
        log_mean_diff = np.nan
        log_std_diff = np.nan
        if np.all(tested_distribution > 0) and np.all(reference_distribution > 0):
            log_calibrated = np.log(tested_distribution)
            log_target = np.log(reference_distribution)
            log_mean_diff = np.mean(log_calibrated) - np.mean(log_target)
            log_std_diff = np.std(log_calibrated, ddof=1) - np.std(log_target, ddof=1)
        
        results = {
            'ks_result': ks_result,
            'wasserstein': wasserstein,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'log_mean_diff': log_mean_diff,
            'log_std_diff': log_std_diff
        }
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            with open (os.path.join(save_path, f"{file_name}.json"), "w") as fp:
                json.dump(results, fp, indent=4)

        return results
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        with open (os.path.join(save_path, f"{file_name}.json"), "w") as fp:
            json.dump(ks_result, fp, indent=4)
    return ks_result
        
    



def lin_ccc(x, y):
    """
    Compute Lin's Concordance Correlation Coefficient (CCC).

    Parameters
    ----------
    x : array-like
        First set of measurements.
    y : array-like
        Second set of measurements (must be same length as x).

    Returns
    -------
    float
        Concordance Correlation Coefficient.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    if x.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    # Means
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    # Variances
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    # Covariance
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    # Pearson correlation
    rho = cov_xy / np.sqrt(var_x * var_y)

    # CCC
    ccc = (2 * rho * np.sqrt(var_x) * np.sqrt(var_y)) / (
        var_x + var_y + (mu_x - mu_y) ** 2
    )

    return ccc



def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix",
    ylabel="Reference",
    xlabel="Test",
    save_path=None,
    plot_name=''
    ):
    """
    Plot a presentation-ready confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like / pd.Series
        Ground truth labels (boolean)
    y_pred : array-like / pd.Series
        Predicted/test labels (boolean)
    title : str
        Plot title
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    save_path : str or None
        If provided, saves the figure to this path

    Returns
    -------
    metrics : dict
        Dictionary with TP, TN, FP, FN and accuracy
    """

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Optional: handle NaNs (drop pairs)
    mask = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Create confusion matrix
    conf = pd.crosstab(y_true, y_pred)

    # Ensure full 2x2 structure
    conf = conf.reindex(index=[False, True], columns=[False, True], fill_value=0)

    TN = conf.loc[False, False]
    FP = conf.loc[False, True]
    FN = conf.loc[True, False]
    TP = conf.loc[True, True]

    total = conf.values.sum()
    accuracy = (TP + TN) / total if total > 0 else 0

    plt.figure(figsize=(4, 4))

    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title(f"{title}\nAgreement = {accuracy:.2%} (n={total})")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        if not plot_name:
            plot_name = "confusion_matrix.png"

    plt.savefig(os.path.join(save_path, plot_name), dpi=300, bbox_inches="tight")

    plt.show()

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy
    } 


def scatter_with_regression(x, y,
                            xlabel="X",
                            ylabel="Y",
                            title="Scatter Plot with Linear Regression",
                            save_path="",
                            plot_name=""):

    x = np.array(x)
    y = np.array(y)

    # Linear regression
    slope, intercept = np.polyfit(x, y, 1)

    # Regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept

    # R²
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)

    # Plot
    plt.figure()
    plt.scatter(x, y, label="Data")

    # Regression line
    # plt.plot(x_line, y_line,
    #          label=f"Regression: y = {slope:.3f}x + {intercept:.3f}")
    
    #Spearman correlation
    spearman_corr, spearman_p = spearmanr(x, y)
    lin = lin_ccc(x, y)

    # X = Y reference line
    min_val = min(min(x), min(y))
    max_val = max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle=":",
             label="X = Y")

    # Equation text
    stats_label = (
    f"Regression: y = {slope:.3f}x + {intercept:.3f}\n"
    f"R² = {r2:.3f}\n"
    f"Spearman ρ = {spearman_corr:.3f}"
    f"\nLin's CCC = {lin:.3f}"
    )


    # Regression line with full stats
    plt.plot(x_line, y_line, label=stats_label)
   
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}/{plot_name}.png")
    plt.show()

    return slope, intercept, r2, spearman_corr, lin

def plot_overlaid_histograms(
    tested_distribution,
    reference_distribution,
    bins=20,
    alpha=0.6,
    figsize=(10, 6),
    test_title='Transformed',
    reference_title='Target',
    title="Distribution Comparison",
    trim_percentile=None,
    save_path=None,
    plot_name='histogram_comparison',
    dpi=300
    ):
    """
    Plot overlaid histograms of transformed and target distributions.

    Parameters:
    -----------
    tested_distribution : array-like
        Transformed distribution samples
    reference_distribution : array-like
        Reference distribution samples
    bins : int or str, default=50
        Number of bins for histogram
    alpha : float, default=0.6
        Transparency level (0-1)
    figsize : tuple, default=(10, 6)
        Figure size
    trim_percentile : float, optional
        Trim to middle percentile
    save_path : str, optional
        Path to save figure (e.g. "plots/my_plot.png")
    dpi : int, default=300
        Figure resolution when saving

    Returns:
    --------
    fig, ax
    """

    tested_distribution = np.asarray(tested_distribution)
    reference_distribution = np.asarray(reference_distribution)

    # Trim to middle percentile if requested
    if trim_percentile is not None:
        lower = (100 - trim_percentile) / 2
        upper = 100 - lower

        tested_lower, tested_upper = np.percentile(
            tested_distribution, [lower, upper]
        )

        reference_lower, reference_upper = np.percentile(
            reference_distribution, [lower, upper]
        )

        tested_distribution = tested_distribution[
            (tested_distribution >= tested_lower) &
            (tested_distribution <= tested_upper)
        ]

        reference_distribution = reference_distribution[
            (reference_distribution >= reference_lower) &
            (reference_distribution <= reference_upper)
        ]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms
    ax.hist(
        tested_distribution,
        bins=bins,
        alpha=alpha,
        label=test_title,
        color='blue',
        density=True,
        edgecolor='black',
        linewidth=0.5
    )

    ax.hist(
        reference_distribution,
        bins=bins,
        alpha=alpha,
        label=reference_title,
        color='red',
        density=True,
        edgecolor='black',
        linewidth=0.5
    )

    # Labels
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(f"{save_path}/{plot_name}.png", dpi=dpi, bbox_inches='tight')

    return fig, ax



 
if __name__ == "__main__":   
    
    
    print("started") 
    
    source_folder = "results/lr_cohorts"
    post_anml_ds_df = pd.read_csv(f"{source_folder}/post_anml_ds.csv")
    post_anml_nds_df = pd.read_csv(f"{source_folder}/post_anml_nds.csv")
    post_anml_prophet_edta_df = pd.read_csv(f"{source_folder}/post_anml_prophet_edta.csv")
    matched_edta_ds_df = pd.read_csv(f"{source_folder}/matched_edta_ds.csv")
    matched_edta_nds_df = pd.read_csv(f"{source_folder}/matched_edta_nds.csv")
    save_path = "results/lr_cohorts/test_evaluation"
    plot_confusion_matrix(
    matched_edta_ds_df['PROphetResult_edta'],
    matched_edta_ds_df['PROphetResult_ds'],
    title="EDTA vs. DS confusion matrix",
    ylabel="EDTA",
    xlabel="DS Streck",
    save_path=save_path,
    plot_name="confusion_edta_ds"
    )
    
    plot_confusion_matrix(
    matched_edta_nds_df['PROphetResult_edta'],
    matched_edta_nds_df['PROphetResult_nds'],
    title="EDTA vs. NDS confusion matrix",
    ylabel="EDTA",
    xlabel="NDS Streck",
    save_path=save_path,
    plot_name="confusion_edta_nds"
    )
    
    
    scatter_with_regression(matched_edta_ds_df['PROphetScore_edta'], matched_edta_ds_df['PROphetScore_ds'],
                            xlabel="EDTA PROphet Score",
                            ylabel="DS PROphet Score",
                            title="EDTA vs DS Scatter Plot with MAD-Med",
                            save_path=save_path,
                            plot_name="scatter_edta_ds")
    
    scatter_with_regression(matched_edta_nds_df['PROphetScore_edta'], matched_edta_nds_df['PROphetScore_nds'],
                            xlabel="EDTA PROphet Score",
                            ylabel="NDS PROphet Score",
                            title="EDTA vs NDS Scatter Plot with MAD-Med",
                            save_path=save_path,
                            plot_name="scatter_edta_nds")
    
    plot_overlaid_histograms(
    post_anml_ds_df['PROphetScore'],
    post_anml_prophet_edta_df['PROphetScore'],
    bins=20,
    alpha=0.6,
    figsize=(10, 6),
    test_title='DS',
    reference_title='EDTA',
    title="Distribution Comparison EDTA vs DS",
    trim_percentile=None,
    save_path=save_path,
    plot_name='histogram_edta_ds',
    dpi=300
    )
    
    plot_overlaid_histograms(
    post_anml_nds_df['PROphetScore'],
    post_anml_prophet_edta_df['PROphetScore'],
    bins=20,
    alpha=0.6,
    figsize=(10, 6),
    test_title='NDS',
    reference_title='EDTA',
    title="Distribution Comparison EDTA vs NDS",
    trim_percentile=None,
    save_path=save_path,
    plot_name='histogram_edta_nds',
    dpi=300
    )
    
    plot_overlaid_histograms(
    post_anml_ds_df['PROphetScore'],
    post_anml_nds_df['PROphetScore'],
    bins=20,
    alpha=0.6,
    figsize=(10, 6),
    test_title='DS',
    reference_title='NDS',
    title="Distribution Comparison DS vs NDS",
    trim_percentile=None,
    save_path=save_path,
    plot_name='histogram_ds_nds',
    dpi=300
    )
    
    compare_distribution_stats(
        post_anml_ds_df['PROphetScore'],
        post_anml_prophet_edta_df['PROphetScore'],
        save_path=save_path,
        file_name='distribution_comparison_ds_edta')
    
    compare_distribution_stats(
        post_anml_nds_df['PROphetScore'],
        post_anml_prophet_edta_df['PROphetScore'],
        save_path=save_path,
        file_name='distribution_comparison_nds_edta')
    
    compare_distribution_stats(
        post_anml_ds_df['PROphetScore'],
        post_anml_nds_df['PROphetScore'],
        save_path=save_path,
        file_name='distribution_comparison_ds_nds')
    
    compare_distribution_stats(
        post_anml_ds_df['PROphetScore'],
        save_path=save_path,
        file_name='distribution_comparison_ds_uniform')
    
    compare_distribution_stats(
        post_anml_nds_df['PROphetScore'],
        save_path=save_path,
        file_name='distribution_comparison_nds_uniform')
    
    compare_distribution_stats(
        post_anml_prophet_edta_df['PROphetScore'],
        save_path=save_path,
        file_name='distribution_comparison_edta_uniform')