import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.dataframe_transformation import get_aptamers


################################################################################################################
############################################ GENERAL FUNCTIONS #################################################
################################################################################################################



def aggregate(df1):
    import time
    # start = time.time()
    df = df1.copy()
    
    # Filter to aptamers starting with a digit
    mask = df['aptamer'].str[0].str.isdigit()
    
    # Calculate means for each aptamer where gIsFeatPopnOL == 0
    means = (df[mask & (df['gIsFeatPopnOL'] == 0)]
             .groupby('aptamer')['gProcessedSignal']
             .mean())
    
    # Map the means back to all rows with those aptamers
    df.loc[mask, 'gProcessedSignal'] = df.loc[mask, 'aptamer'].map(means)
    # print(f"Aggregation took {time.time() - start} seconds")
    return df





def transform_by_method(feature_df, params, aptamers, method='MAD'):
    """
    Apply a parameterized transformation to signal values for selected aptamers.

    This function modifies the 'gProcessedSignal' column for rows corresponding
    to specified aptamers by applying a transformation in log2 space. The exact
    transformation depends on the selected method and per-aptamer parameters.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Input DataFrame containing at least 'aptamer' and 'gProcessedSignal' columns.

    params : dict
        Dictionary of per-aptamer parameters. Expected structure depends on the method:
        - 'linear_regression': requires 'slope' and 'intercept'
        - 'MAD' (default): requires 'mad_edta', 'mad_streck', 'med_edta', 'med_streck'
        - 'median_only': uses only medians (MAD terms set to 1 internally)

    aptamers : list
        List of aptamer identifiers to which the transformation should be applied.

    method : str, optional
        Transformation method to use. Supported options:
        - 'MAD' (default): affine transformation based on median and MAD ratios
        - 'median': median shift only (no scaling)
        - 'linear_regression': uses precomputed slope and intercept

    Returns
    -------
    pandas.DataFrame
        DataFrame with transformed 'gProcessedSignal' values for the selected aptamers.

    Notes
    -----
    - Transformation is applied only to rows where 'aptamer' is in the provided list.
    - All transformations are performed in log2 space and then exponentiated back.
    - For 'MAD'-based methods:
        log2(signal*) = a * log2(signal) + b
      where:
        a = mad_edta / mad_streck
        b = med_edta - a * med_streck
    - For 'linear_regression':
        log2(signal*) = slope * log2(signal) + intercept
    - The function operates on a copy of the input DataFrame.
    """
    if method not in ['MAD', 'median', 'linear_regression']:
        raise ValueError(f"Unsupported method: {method}. Choose from 'MAD', 'median', 'linear_regression'.")
    df = feature_df.copy()
    mask = df['aptamer'].isin(aptamers)
    
    # Create parameter lookup DataFrames
    param_df = pd.DataFrame.from_dict(params, orient='index')
    if method=='linear_regression':
        slope_map = param_df['slope'].to_dict()
        slope_map['other'] = 1
        df_aptamers = df['aptamer'].unique()

        intercept_map = param_df['intercept'].to_dict()
        intercept_map['other'] = 0
        for aptamer in df_aptamers:
            if aptamer not in intercept_map:
                intercept_map[aptamer] = 0
                slope_map[aptamer] = 1
        # Vectorized transformation
        slope = df.loc[mask, 'aptamer'].map(slope_map)
        intercept = df.loc[mask, 'aptamer'].map(intercept_map)
        log2_signal = np.log2(df.loc[mask, 'gProcessedSignal'])
        log2_signal_star = log2_signal * slope + intercept
        df.loc[mask, 'gProcessedSignal'] = np.power(2, log2_signal_star)
        return df
        

    if method == 'median':
        param_df['mad_edta'] = 1
        param_df['mad_streck'] = 1
    a_map = (param_df['mad_edta'] / param_df['mad_streck']).to_dict()
    b_map = (param_df['med_edta'] - 
             (param_df['mad_edta'] / param_df['mad_streck']) * param_df['med_streck']).to_dict()
    
    a = df.loc[mask, 'aptamer'].map(a_map)
    b = df.loc[mask, 'aptamer'].map(b_map)
    
    log2_signal = np.log2(df.loc[mask, 'gProcessedSignal'])
    log2_streck_star = log2_signal * a + b
    df.loc[mask, 'gProcessedSignal'] = np.power(2, log2_streck_star)
    
    return df


