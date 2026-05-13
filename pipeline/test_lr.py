
import os

import numpy as np
import pandas as pd
from src.dataframe_transformation import add_measure_id

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
    for col, coefs in params.items():
        slope = coefs["slope"]
        intercept = coefs["intercept"]

        expected = 2 ** (slope * np.log2(df1[col]) + intercept)
        actual = df2[col]

        relative_error = np.abs((actual - expected) / expected)
        if not (relative_error <= tolerance).all():
            print(f"Column '{col}' failed linear regression check.")
            return False

    return True

if __name__ == "__main__":
    
    from src.adat_handling import read_adat_file
    print(os.getcwd())
    unalterd_df, _ = read_adat_file('somalogic/adat/test/unaltered/OH2025_054/OH2025_054.adat')
    lr_df, _ = read_adat_file('somalogic/adat/test/lr/OH2025_054/OH2025_054.adat')
    unaltered_df = add_measure_id(unalterd_df)
    lr_df = add_measure_id(lr_df)
    ds_ids = pd.read_csv('results/lr_cohorts/post_anml_ds.csv')['MeasureId']
    nds_ids = pd.read_csv('results/lr_cohorts/post_anml_nds.csv')['MeasureId']
    unaltered_ds_df = unaltered_df[unalterd_df['MeasureId'].isin(ds_ids)].set_index('MeasureId')
    lr_ds_df = lr_df[lr_df['MeasureId'].isin(ds_ids)].set_index('MeasureId')
    unaltered_nds_df = unaltered_df[unalterd_df['MeasureId'].isin(nds_ids)].set_index('MeasureId')
    lr_nds_df = lr_df[lr_df['MeasureId'].isin(nds_ids)].set_index('MeasureId')
    
    print()
    
    
    
    
    
    
    
    
    
    
    
    import pandas as pd

    # Example usage
    df1 = pd.DataFrame({
        "A": [1, 2, 4, 8],
        "B": [10, 20, 40, 80]
    })

    # Simulate a linear bridging with slope=1 and intercept=0.5 in log2 space
    df2 = pd.DataFrame({
        "A": 2 ** (1 * np.log2(df1["A"]) + 0.5),
        "B": 2 ** (1 * np.log2(df1["B"]) + 0.5)
    })

    params = {
        "A": {"slope": 1, "intercept": 0.5},
        "B": {"slope": 1, "intercept": 0.5}
    }

    result = compare_dfs(df1, df2, params)
    print("Comparison result:", result)