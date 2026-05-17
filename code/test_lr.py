
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from modeling.rap_response_predictor import *
from src.dataframe_transformation import add_measure_id

def fit_log2_lr(source_distribution_df: pd.DataFrame, target_distribution_df: pd.DataFrame, mapping: pd.DataFrame, source_column: str, target_column: str) -> dict:
    """
    Fit a log2-space linear regression from streck to edta for each numerical column.

    For each column, fits:
        log2(edta[col]) = slope * log2(streck[col]) + intercept

    Parameters
    ----------
    source_distribution_df : pd.DataFrame
        Source dataframe indexed by MeasureId.
    target_distribution_df : pd.DataFrame
        Target dataframe indexed by MeasureId.
    mapping : pd.DataFrame
        Dataframe with columns with source and target MeasureIds,  pairing
        corresponding rows. Rows whose source or target MeasureId is absent from
        the respective dataframe are silently dropped.
    source_column : str
        Name of the column in mapping that contains MeasureIds for the source dataframe.
    target_column : str
        Name of the column in mapping that contains MeasureIds for the target dataframe.

    Returns
    -------
    dict
        Nested dict: {col: {"slope": float, "intercept": float}}
    """
    valid_mapping = mapping[
        mapping[source_column].isin(source_distribution_df.index) &
        mapping[target_column].isin(target_distribution_df.index)
    ]

    source_aligned = source_distribution_df.loc[valid_mapping[source_column].values]
    target_aligned = target_distribution_df.loc[valid_mapping[target_column].values]

    # num_cols = source_aligned.select_dtypes(include=np.number).columns.intersection(
    #     target_aligned.select_dtypes(include=np.number).columns
    # )
    num_cols = [col for col in source_aligned.columns if col[0].isdigit()]
    params = {}
    for col in num_cols:
        x = np.log2(source_aligned[col].values)
        y = np.log2(target_aligned[col].values)
        slope, intercept, *_ = stats.linregress(x, y)
        params[col] = {"slope": float(slope), "intercept": float(intercept)}
    return params


def plot_slope_pdfs(*params_and_names: tuple, save_path: str = None) -> plt.Figure:
    """
    Plot the PDF of slope values from one or more fit_log2_lr results.

    Parameters
    ----------
    *params_and_names : tuple
        Each argument should be a (params_dict, name) tuple, where params_dict
        is the output of fit_log2_lr and name is a label string for the legend.
    save_path : str, optional
        If provided, the figure is saved to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots()
    for params, name in params_and_names:
        slopes = np.array([v["slope"] for v in params.values()])
        kde = stats.gaussian_kde(slopes)
        x = np.linspace(slopes.min() - 0.5, slopes.max() + 0.5, 500)
        ax.plot(x, kde(x), label=name)

    ax.set_xlabel("Slope")
    ax.set_ylabel("Density")
    ax.set_title("PDF of log2-space regression slopes")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig


def compare_dfs(df1, df2, params: dict, tolerance: float = 0.01) -> bool:
    """
    Check whether df2 is a correct log2-space linear bridging of df1.

    For each column in params, the expected relationship is:
        log2(df2[col]) = slope * log2(df1[col]) + intercept
    i.e.:
        df2[col] = 2 ** (slope * log2(df1[col]) + intercept)

    Returns True if every mapped column is within `tolerance` (relative)
    of its expected values, False otherwise.

    Parameters
    ----------
    df1 : pd.DataFrame
        Source dataframe (pre-bridging).
    df2 : pd.DataFrame
        Target dataframe (post-bridging).
    params : dict
        Keys are column names; values are dicts with 'slope' and 'intercept'.
    tolerance : float
        Allowed relative deviation (default 0.01 = 1%).
    """
    passed_cols = 0
    for col, coefs in params.items():
        slope = coefs["slope"]
        intercept = coefs["intercept"]

        expected = 2 ** (slope * np.log2(df1[col]) + intercept)
        actual = df2[col]

        relative_error = np.abs((actual - expected) / expected)
        if not (relative_error <= tolerance).all():
            print(f"Column '{col}' failed linear regression check. {passed_cols} columns passed.")
            return False
        passed_cols += 1
    print(f"All {passed_cols} columns passed linear regression check.")

    return True

if __name__ == "__main__":
    print("started test_lr.py")
    matched_ds_path = "data/excels/matched_ids_edta_ds.csv"
    matched_nds_path = "data/excels/matched_ids_edta_nds.csv"
    matched_ds = pd.read_csv(matched_ds_path)[['MeasureId_edta', 'MeasureId_ds']]
    matched_nds = pd.read_csv(matched_nds_path)[['MeasureId_edta', 'MeasureId_nds']]
    base_samples = pd.read_parquet('data/adat/base/v_175_OH2026_013.parquet').reset_index(drop=False)
    base_samples = add_measure_id(base_samples)
    ds_samples = base_samples[base_samples['MeasureId'].isin(matched_ds['MeasureId_ds'])].set_index('MeasureId')
    nds_samples = base_samples[base_samples['MeasureId'].isin(matched_nds['MeasureId_nds'])].set_index('MeasureId')
    edta_samples = base_samples[base_samples['MeasureId'].isin(matched_ds['MeasureId_edta'].tolist() + matched_nds['MeasureId_edta'].tolist())].set_index('MeasureId')
    model = pd.read_pickle('data/models/DCB_nosex_gamma1_rescaled.pkl')
    raps = [rap for rap in model.RAPs.keys()]
    sheba = pd.read_csv('data/excels/SeqId Annotations Secreted & Sheba Filters.csv')
    sheba = sheba[sheba['Sheba p-value'] >= 0.05]['SeqId'].to_list()
    all_aptamers = [col for col in ds_samples.columns if col[0].isdigit()]
    
    
    ds_lr_params = fit_log2_lr(ds_samples, edta_samples, matched_ds, 'MeasureId_ds', 'MeasureId_edta')
    ds_raps_params = {key: value for key, value in ds_lr_params.items() if key in raps}
    ds_sheba_params = {key: value for key, value in ds_lr_params.items() if key in sheba}
    nds_lr_params = fit_log2_lr(nds_samples, edta_samples, matched_nds, 'MeasureId_nds', 'MeasureId_edta')
    nds_raps_params = {key: value for key, value in nds_lr_params.items() if key in raps}
    nds_sheba_params = {key: value for key, value in nds_lr_params.items() if key in sheba}
    
    plot_slope_pdfs(
        (ds_lr_params, "DS all aptamers"),
        (ds_raps_params, "DS RAPs"),
        (ds_sheba_params, "DS Sheba"),
        save_path='projects/pre_post_bridging/results/linear_regression/ds_slope_pdfs.png'
    )
    
    plot_slope_pdfs(
        (nds_lr_params, "NDS all aptamers"),
        (nds_raps_params, "NDS RAPs"),
        (nds_sheba_params, "NDS Sheba"),
        save_path='projects/pre_post_bridging/results/linear_regression/nds_slope_pdfs.png'
    )
    with open('data/params/pre_normalization/ds_lr_roee_14052026.json', 'w') as f:
        json.dump(ds_lr_params, f, indent=4)
    with open('data/params/pre_normalization/nds_lr_roee_14052026.json', 'w') as f:
        json.dump(nds_lr_params, f, indent=4)
