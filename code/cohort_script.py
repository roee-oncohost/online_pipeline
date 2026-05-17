import os
import re
import json
import warnings
from pathlib import Path
import pandas as pd
from src.adat_handling import read_adat_file
from src.prophet_scores import get_prophet_score
from modeling.rap_response_predictor import RapResponsePredictor

#TODO: 
"""
switch from parquet to Or's csv (especially when there's edta post-ANML)
make sure of all measure id's before merges
merge on 'NewSubjectId'

"""

warnings.filterwarnings("ignore")

def add_measure_id(df_):
    """
    Add a 'MeasureId' column to the DataFrame if it does not already exist.

    'MeasureId' is constructed by concatenating 'PlateId' and 'PlatePosition'
    with an underscore separator. The column is inserted as the first column.

    Args:
        df_ (pd.DataFrame): Input DataFrame containing 'PlateId' and
            'PlatePosition' columns.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with 'MeasureId' as the
            first column. If 'MeasureId' already exists, the DataFrame is
            returned unchanged.
    """
    df = df_.copy()
    if not 'MeasureId' in df.columns:
        df.insert(0, 'MeasureId', df['PlateId'] + '_' + df['PlatePosition'])
        print('Added MeasureId')
    else:
        print('No need to add MeasureId')
    return df


def add_plate_id(df_):
    """
    Add a 'PlateId' column to the DataFrame if it does not already exist.

    'PlateId' is constructed by splitting 'MeasureId' at the underscore and taking the first two parts.

    Args:
        df_ (pd.DataFrame): Input DataFrame containing 'MeasureId' column.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with 'PlateId' as the
            second column. If 'PlateId' already exists, the DataFrame is
            returned unchanged.
    """
    df = df_.copy()
    if not 'PlateId' in df.columns:
        df.insert(1, 'PlateId', df['MeasureId'].str.split('_').str[:2].str.join('_'))
        print('Added PlateId')
    else:
        print('No need to add PlateId')
    return df


def filter_dataframe(df, *filters):
    """
    Apply multiple filter conditions to a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        *filters (pd.Series): Variable number of boolean Series representing 
            filter conditions. All filters are combined using AND logic.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only rows that satisfy all 
            filter conditions.
    
    Examples:

        >>> # Apply single filter
        >>> filtered = filter_dataframe(df, df['age'] > 25)
        >>> 
        >>> # Apply multiple filters
        >>> filtered = filter_dataframe(
        ...     df,
        ...     df['MeasureId'].isin(['OH2025_054_A6', 'OH2025_038_G3']),
        ...     df['whatever'] < 65000
        ... )
    """
    result = df.copy()
    for filter_condition in filters:
        result = result[filter_condition]
    return result




def get_base_rfu_data(rfus_path, info_path):
    """
    Load and prepare RFU data split by sample matrix type.

    Reads raw RFU data from a Parquet file and sample metadata from a CSV,
    cleans 'PlateId' values, adds 'MeasureId', and returns filtered subsets
    for EDTA Plasma, Streck DS2, and Streck non-DS sample types.

    Args:
        rfus_path (str): Path to the Parquet file containing raw RFU data.
        info_path (str): Path to the CSV file containing sample metadata
            (including 'PlateId', 'SampleMatrixType', and 'SubType' columns).

    Returns:
        tuple:
            - or_rfus (pd.DataFrame): Full cleaned RFU DataFrame with 'MeasureId'.
            - info_df (pd.DataFrame): Full cleaned metadata DataFrame.
            - edta_prophet_rfu_df (pd.DataFrame): RFU rows for EDTA Plasma PROPHET samples,
              excluding plates OH2026_006 and OH2026_009 (since those were LOT 3).
            - ds_rfu_df (pd.DataFrame): RFU rows for Streck DS2 PROPHET samples.
            - nds_rfu_df (pd.DataFrame): RFU rows for Streck non-DS samples.
            - all_ds (pd.DataFrame): Metadata rows for all Streck DS2 samples.
            - all_edta (pd.DataFrame): Metadata rows for all EDTA Plasma samples (not just PROPHET), since some matching was labeled PROPHETIC.
    """

    #Reading the data
    or_rfus = pd.read_csv(rfus_path)  # pd.read_parquet(rfus_path).reset_index(drop=False)
    info_df = pd.read_csv(info_path)

    #Cleaning PlateId values
    # or_rfus['PlateId'] = or_rfus['PlateId'].replace('OH2023_001', 'OH2023_01')
    # info_df['PlateId'] = info_df['PlateId'].str.split('_set').str[0]
    # or_rfus['PlateId'] = or_rfus['PlateId'].str.split('_set').str[0]
    # or_rfus['PlateId'] = or_rfus['PlateId'].str.split('_Lot3').str[0]
    or_rfus = add_measure_id(or_rfus)
    or_rfus = add_plate_id(or_rfus)
    
    #Filtering
    all_edta = filter_dataframe(info_df, info_df['SampleMatrixType']=='EDTA Plasma')
    edta_info_df = filter_dataframe(info_df, info_df['SampleMatrixType']=='EDTA Plasma',
                                info_df['SubType']=='PROPHET')
    edta_prophet_rfu_df = or_rfus[or_rfus['MeasureId'].isin(edta_info_df['MeasureId'])].drop_duplicates(subset=['MeasureId']).reset_index(drop=True)
    removed_measure_ids = info_df[info_df['PlateId'].isin(['OH2026_006', 'OH2026_009'])]['MeasureId'].unique()
    or_rfus = filter_dataframe(or_rfus, ~or_rfus['MeasureId'].isin(removed_measure_ids))
    edta_prophet_rfu_df = filter_dataframe(edta_prophet_rfu_df,
                                           ~edta_prophet_rfu_df['MeasureId'].isin(removed_measure_ids))
    all_ds = filter_dataframe(info_df, info_df['SampleMatrixType']=='Streck DS2')
    ds_info_df = filter_dataframe(info_df, info_df['SampleMatrixType']=='Streck DS2',
                                info_df['SubType']=='PROPHET')
    ds_rfu_df = or_rfus[or_rfus['MeasureId'].isin(ds_info_df['MeasureId'])].drop_duplicates(subset=['MeasureId']).reset_index(drop=True)

    nds_info_df = filter_dataframe(info_df, info_df['SampleMatrixType']=='Streck non-DS')
    nds_rfu_df = or_rfus[or_rfus['MeasureId'].isin(nds_info_df['MeasureId'])].drop_duplicates(subset=['MeasureId']).reset_index(drop=True)




    return or_rfus, info_df, edta_prophet_rfu_df, ds_rfu_df, nds_rfu_df, all_ds, all_edta

def concat_adats(path, pattern):
    """
    Recursively load and concatenate all ADAT files matching a pattern.

    Searches the given directory (and subdirectories) for '*.adat' files whose
    names match the provided regex pattern, reads each into a DataFrame, and
    concatenates them into a single DataFrame with a 'MeasureId' column added
    and rows sorted by 'MeasureId'.

    Args:
        path (str): Path to the root directory to search for ADAT files.
        pattern (str): Regex pattern to match against ADAT file names.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all matching ADAT files,
            with 'MeasureId' added and sorted.
    """
    folder = Path(path)

    dfs = []

    for file in folder.rglob("*.adat"):
        if re.search(pattern, file.name):
            df, _ = read_adat_file(str(file))
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = add_measure_id(df)
    df.sort_values('MeasureId', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

def get_cohorts(rfus_path, info_path, plates, adats_path, model_path, matched_samples_path, save_path=""):


    or_rfus, info_df, edta_prophet_rfu_df, ds_prophet_rfu_df, nds_rfu_df, all_ds_rfu_df, all_edta_rfu_df = get_base_rfu_data(rfus_path, info_path)
    
    ds_prophet_rfu_df = ds_prophet_rfu_df[ds_prophet_rfu_df['PlateId'].isin(plates)]
    
    ############################    
    ###### DATA FROM ADATS #####
    ############################
    pattern_dict = {'base': r'\d.adat$',
                    'pre_anml': r'qcCheck.adat$',
                    'post_anml': r'anmlSMP.adat$'}

    
    adat_dfs = {key: concat_adats(adats_path, pattern) for key, pattern in pattern_dict.items()}       
    post_anml_adats_df = adat_dfs['post_anml']
    # final_ds_df = final_df[final_df['MeasureId'].isin(ds_rfu_df['MeasureId'])].reset_index(drop=True)
    # final_nds_df = post_anml_adats_df[post_anml_adats_df['MeasureId'].isin(nds_rfu_df['MeasureId'])].reset_index(drop=True)

    post_anml_edta = or_rfus #pd.read_parquet('data/adat/anmlSMP/v_262_OH2026_012.parquet').reset_index(drop=False)
    post_anml_edta = add_measure_id(post_anml_edta)
    post_anml_edta = post_anml_edta[post_anml_edta['MeasureId'].isin(all_edta_rfu_df['MeasureId'])].reset_index(drop=True)
    post_anml_prophet_edta = post_anml_edta[post_anml_edta['MeasureId'].isin(edta_prophet_rfu_df['MeasureId'])].reset_index(drop=True)
    model = pd.read_pickle(model_path)
    post_anml_ds_df = post_anml_adats_df[post_anml_adats_df['MeasureId'].isin(all_ds_rfu_df['MeasureId'])].reset_index(drop=True)
    post_anml_nds_df = post_anml_adats_df[post_anml_adats_df['MeasureId'].isin(nds_rfu_df['MeasureId'])].reset_index(drop=True)
    post_anml_ds_df['PROphetScore_ds'] = get_prophet_score(post_anml_ds_df, model)
    post_anml_nds_df['PROphetScore'] = get_prophet_score(post_anml_nds_df, model)
    post_anml_ds_df['MeasureId_ds'] = post_anml_ds_df['MeasureId']
    # post_anml_edta['PROphetScore'] = get_prophet_score(post_anml_edta, model)
    post_anml_prophet_edta['PROphetScore'] = get_prophet_score(post_anml_prophet_edta, model)
    
    ############################    
    ###### MATCHED SAMPLES #####
    ############################
    print("matching samples...")
    matched_samples = pd.read_excel(matched_samples_path)
    matched_edta = matched_samples[matched_samples['SampleRun'] == 'EDTA']
    matched_ds = matched_samples[matched_samples['SampleRun'] == 'DS2']
    matched_nds = matched_samples[matched_samples['SampleRun'] == 'non DS']
    matched_edta_ds = pd.merge(matched_edta, matched_ds, how='inner', on='NewSubjectId', suffixes=('_edta', '_ds'))
    matched_edta_nds = pd.merge(matched_edta, matched_nds, how='inner', on='NewSubjectId', suffixes=('_edta', '_nds'))
    matched_edta_nds['PROphetScore_nds'] = matched_edta_nds.merge(post_anml_nds_df, left_on='MeasureId_nds', right_on='MeasureId')['PROphetScore']
    measure_ids_to_keep_for_ds = matched_edta_ds["MeasureId_edta"].dropna().unique()
    measure_ids_to_keep_for_nds = matched_edta_nds["MeasureId_edta"].dropna().unique()
    post_anml_edta_ds_small = post_anml_edta[
    post_anml_edta["MeasureId"].isin(measure_ids_to_keep_for_ds)
    ].reset_index(drop=True)
    post_anml_edta_nds_small = post_anml_edta[
    post_anml_edta["MeasureId"].isin(measure_ids_to_keep_for_nds)
    ].reset_index(drop=True)
    post_anml_edta_ds_small['PROphetScore'] = get_prophet_score(post_anml_edta_ds_small, model)
    post_anml_edta_nds_small['PROphetScore'] = get_prophet_score(post_anml_edta_nds_small, model)

    matched_edta_ds = matched_edta_ds.merge(post_anml_ds_df[['MeasureId_ds', 'PROphetScore_ds']], on='MeasureId_ds')
    matched_edta_ds['PROphetScore_edta'] = matched_edta_ds.merge(post_anml_edta_ds_small, left_on='MeasureId_edta', right_on='MeasureId')['PROphetScore']

    
    matched_edta_nds['PROphetScore_edta'] = matched_edta_nds.merge(post_anml_edta_nds_small, left_on='MeasureId_edta', right_on='MeasureId')['PROphetScore']
    matched_edta_ds['PROphetResult_ds'] = matched_edta_ds['PROphetScore_ds'] >= 5
    matched_edta_ds['PROphetResult_edta'] = matched_edta_ds['PROphetScore_edta'] >= 5
    matched_edta_nds['PROphetResult_nds'] = matched_edta_nds['PROphetScore_nds'] >= 5
    matched_edta_nds['PROphetResult_edta'] = matched_edta_nds['PROphetScore_edta'] >= 5
    
    
    ############################    
    ###### COHORT SUMMARY ######
    ############################
    cohort_summary = {
    "edta_prophet_samples": len(post_anml_prophet_edta),
    "ds_prophet_samples": len(ds_prophet_rfu_df),
    "nds_samples": len(nds_rfu_df),
    "matched_edta_ds_samples": len(matched_edta_ds),
    "matched_edta_nds_samples": len(matched_edta_nds)
    }
    post_anml_ds_df['PROphetScore'] = post_anml_ds_df['PROphetScore_ds']
    # post_anml_nds_df['PROphetScore'] = post_anml_nds_df['PROphetScore_nds']
    ############################    
    ####### SAVING COHORT ######
    ############################
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        post_anml_edta.to_csv(os.path.join(save_path, "post_anml_all_edta.csv"), index=False)
        post_anml_prophet_edta.to_csv(os.path.join(save_path, "post_anml_prophet_edta.csv"), index=False)
        post_anml_ds_df.to_csv(os.path.join(save_path, "post_anml_ds.csv"), index=False)
        post_anml_nds_df.to_csv(os.path.join(save_path, "post_anml_nds.csv"), index=False)
        matched_edta_ds.to_csv(os.path.join(save_path, "matched_edta_ds.csv"), index=False)
        matched_edta_nds.to_csv(os.path.join(save_path, "matched_edta_nds.csv"), index=False)
        with open(os.path.join(save_path, "cohort_summary.json"), "w") as f:
            json.dump(cohort_summary, f, indent=4)
    return cohort_summary
        




if __name__ == "__main__":
    print("started")

    plates = [f"OH2025_05{i}" for i in range(1, 8)] 
    plates += [f"OH2026_00{i}" for i in range(1, 6)]
    


    rfus_path = 'data/excels/20260412_adat_data_measurements.csv' # 'data/adat/base/v_175_OH2026_013.parquet'
    info_path = 'data/excels/20260406_adat_data_samples_with_sample_info.csv'
    
    adats_path = 'somalogic/adat/lr'
    model_path = 'data/models/DCB_nosex_gamma1_rescaled.pkl'
    matched_samples_path = 'data/excels/26042026_streck_long_summary.xlsx'
  
    get_cohorts(rfus_path, info_path, plates, adats_path, model_path, matched_samples_path, save_path="results/lr_cohorts/")
