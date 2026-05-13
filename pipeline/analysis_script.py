
# # Installation README

   
# ## For this particular project to MUST have a local virtual environment!!!
# 
# The requirements are  listed in clean_env_requirements.txt
# 
# ### Adding the SRC files:
# In the main folder are 2 files: requirements.txt, setup.py
# setup.py: allows you to add folders as packages (that's why you need a virtual environment). To install these packages, once you've activated the virtual environment use "pip install -e ." from the main folder (where setup.py is located).
# requirements.txt: install (to the virtual environment) using "pip install -r requirements.txt" 



# # Importing 

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import spearmanr, kstest, ks_2samp, uniform
from src.alter_plate import *
from src.bridging_coefficients import *
from src.distribution_assessment import *
from src.adat_handling import read_adat_file
from src.dataframe_transformation import *
from src.prophet_scores import get_prophet_score 
from modeling import rap_response_predictor

# from src.altering_plate_multiple_collection_methods import alter_scanner_files_MAD_bridging as alter_multiple_mad

   
# # Getting the data

 
# # get base RFUs from Izhar's parquet files
# base_rfus = pd.read_parquet('../../data/adat/base/v_166_OH2026_004.parquet').reset_index(drop=False)
# info_df = pd.read_csv('../../data/excels/20260126_adat_data_samples_with_sample_info.csv')


 




import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(df, ground_truth_column, test_column, title="Confusion Matrix", ylabel="Reference", xlabel="Test", save_path=None):
    """
    Plot a presentation-ready confusion matrix heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the ground truth and test columns
    ground_truth_column : str
        Name of the column with ground truth labels
    test_column : str
        Name of the column with test labels
    title : str
        Plot title
    save_path : str or None
        If provided, saves the figure to this path

    Returns
    -------
    metrics : dict
        Dictionary with TP, TN, FP, FN and accuracy
    """

    # Create confusion matrix
    conf = pd.crosstab(df[ground_truth_column], df[test_column])

    # Ensure full 2x2 structure
    conf = conf.reindex(index=[False, True], columns=[False, True], fill_value=0)

    TN = conf.loc[False, False]
    FP = conf.loc[False, True]
    FN = conf.loc[True, False]
    TP = conf.loc[True, True]

    total = conf.values.sum()
    accuracy = (TP + TN) / total if total > 0 else 0 # type: ignore

    plt.figure(figsize=(4,4))

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
        plt.savefig(save_path, dpi=300)

    plt.show()

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy
    }

 
def get_base_rfu_data(base_rfus_path, info_path):
    
    
    #Reading the data
    base_rfus = pd.read_parquet(base_rfus_path).reset_index(drop=False)
    info_df = pd.read_csv(info_path)

    #Cleaning PlateId values
    base_rfus['PlateId'] = base_rfus['PlateId'].replace('OH2023_001', 'OH2023_01')
    info_df['PlateId'] = info_df['PlateId'].str.split('_set').str[0]
    base_rfus['PlateId'] = base_rfus['PlateId'].str.split('_set').str[0]
    base_rfus['PlateId'] = base_rfus['PlateId'].str.split('_Lot3').str[0]
    base_rfus = add_measure_id(base_rfus)
    
    #Filtering
    all_edta = filter_dataframe(info_df, info_df['SampleMatrixType']=='EDTA Plasma')
    edta_info_df = filter_dataframe(info_df, info_df['SampleMatrixType']=='EDTA Plasma',
                                info_df['SubType']=='PROPHET')
    edta_rfu_df = base_rfus[base_rfus['MeasureId'].isin(edta_info_df['MeasureId'])].drop_duplicates(subset=['MeasureId']).reset_index(drop=True)
    
    edta_rfu_df = edta_rfu_df[~edta_rfu_df['PlateId'].isin(['OH2026_006', 'OH2026_009'])]
    all_ds = filter_dataframe(info_df, info_df['SampleMatrixType']=='Streck DS2')
    ds_info_df = filter_dataframe(info_df, info_df['SampleMatrixType']=='Streck DS2',
                                info_df['SubType']=='PROPHET')
    ds_rfu_df = base_rfus[base_rfus['MeasureId'].isin(ds_info_df['MeasureId'])].drop_duplicates(subset=['MeasureId']).reset_index(drop=True)

    nds_info_df = filter_dataframe(info_df, info_df['SampleMatrixType']=='Streck non-DS')
    nds_rfu_df = base_rfus[base_rfus['MeasureId'].isin(nds_info_df['MeasureId'])].drop_duplicates(subset=['MeasureId']).reset_index(drop=True)




    return base_rfus, info_df, edta_rfu_df, ds_rfu_df, nds_rfu_df, all_ds, all_edta

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
)
    # stats_label = (
    # f"y = {slope:.3f}x + {intercept:.3f}\n"
    # f"R² = {r2:.3f}\n"
    # f"Spearman ρ = {spearman_corr:.3f}"
    # )

# Regression line with full stats
    plt.plot(x_line, y_line, label=stats_label)
    # equation = f"R² = {r2:.3f}"
    # stats_text = f"{equation}\nSpearman ρ = {spearman_corr:.3f}"
    # plt.text(min(x), max(y), stats_text, verticalalignment='top')
    # plt.text(min(x), max(y), equation)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    if save_path:
        plt.savefig(f"{save_path}/{plot_name}.png")
    plt.show()

    return slope, intercept, r2

# def alter_plates_for_bridging(plates, plate_path, param_path_dict, plate_save_path):
    
    
    
#     for plate in plates:
#         alter_multiple_mad(f'{plate_path}/{plate}',
#                 f'{plate_path}/{plate}/{plate} Workbook.xlsx',
                
#                 param_path_dict,
#                 f'{plate_save_path}/{plate}',
#                 median_only=False)





def ks_2sided(values_a, values_b=None):
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
    dict with statistic, p_value, and metadata
    """

    x = np.asarray(values_a, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        raise ValueError("values_a contains no valid numeric values.")

    # Case 1: two-sample KS
    if values_b is not None:
        y = np.asarray(values_b, dtype=float)
        y = y[~np.isnan(y)]

        if len(y) == 0:
            raise ValueError("values_b contains no valid numeric values.")

        result = ks_2samp(x, y, alternative="two-sided")

        return {
            "test": "two-sample KS (two-sided)",
            "statistic": result.statistic,
            "p_value": result.pvalue,
            "n_a": len(x),
            "n_b": len(y),
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
        "n_a": len(x),
    }


def analyze_results(base_rfus_path, info_path, plates, adats_path, model_path, matched_samples_path, save_path=None, normalization_type=None, confusion_headers=[], xlabel="", ylabel=""):


    base_rfus, info_df, edta_rfu_df, ds_rfu_df, nds_rfu_df, all_ds, all_edta = get_base_rfu_data(base_rfus_path, info_path)
    
    ds_rfu_df = ds_rfu_df[ds_rfu_df['PlateId'].isin(plates)]
        
    pattern_dict = { #'base': r'\d.adat$',
    #                 'pre_anml': r'qcCheck.adat$',
                    'post_anml': r'anmlSMP.adat$'}

    
    adat_dfs = {key: concat_adats(adats_path, pattern) for key, pattern in pattern_dict.items()}

    
    # base_df = adat_dfs['base']    
    final_df = adat_dfs['post_anml']
    # final_ds_df = final_df[final_df['MeasureId'].isin(ds_rfu_df['MeasureId'])].reset_index(drop=True)
    final_nds_df = final_df[final_df['MeasureId'].isin(nds_rfu_df['MeasureId'])].reset_index(drop=True)

    post_anml_edta = pd.read_parquet('data/adat/anmlSMP/v_262_OH2026_012.parquet').reset_index(drop=False)
    post_anml_edta = add_measure_id(post_anml_edta)
    post_anml_edta = post_anml_edta[post_anml_edta['MeasureId'].isin(all_edta['MeasureId'])].reset_index(drop=True)

    model = pd.read_pickle(model_path)

    post_anml_df = adat_dfs['post_anml']
    post_anml_ds_df = post_anml_df[post_anml_df['MeasureId'].isin(all_ds['MeasureId'])].reset_index(drop=True)
    post_anml_nds_df = post_anml_df[post_anml_df['MeasureId'].isin(final_nds_df['MeasureId'])].reset_index(drop=True)
    post_anml_ds_df['PROphetScore_ds'] = get_prophet_score(post_anml_ds_df, model)
    post_anml_nds_df['PROphetScore'] = get_prophet_score(post_anml_nds_df, model)
    post_anml_ds_df['MeasureId_ds'] = post_anml_ds_df['MeasureId']
    # post_anml_edta = pd.read_parquet('data/adat/anmlSMP/v_262_OH2026_012.parquet').reset_index(drop=False)
    # post_anml_edta = add_measure_id(post_anml_edta)
    # post_anml_edta = post_anml_edta[post_anml_edta['MeasureId'].isin(edta_rfu_df['MeasureId'])].reset_index(drop=True)
    post_anml_edta['PROphetScore'] = get_prophet_score(post_anml_edta, model)
    post_anml_edta_for_distributions = post_anml_edta[post_anml_edta['MeasureId'].isin(edta_rfu_df['MeasureId'])].drop_duplicates(subset='MeasureId', keep='first').reset_index(drop=True)

    ks_results_list = []
    title = f"KS test for DS vs EDTA PROphet Scores, normalization: {normalization_type}"
    print(title)
    ks_results = ks_2sided(post_anml_ds_df['PROphetScore_ds'], post_anml_edta_for_distributions['PROphetScore'])
    ks_results['title'] = title
    ks_results_list.append(ks_results)
    print(f"Test: {ks_results['test']}\nStatistic: {ks_results['statistic']:.4f}\nP-value: {ks_results['p_value']:.4e}\nSample Sizes: n_DS={ks_results['n_a']}, n_EDTA={ks_results['n_b']}")
    print('***************************************')
    
    title = f"KS test for nDS vs EDTA PROphet Scores, normalization: {normalization_type}"
    print(title)
    ks_results = ks_2sided(post_anml_nds_df['PROphetScore'], post_anml_edta_for_distributions['PROphetScore'])
    ks_results['title'] = title
    ks_results_list.append(ks_results)
    print(f"Test: {ks_results['test']}\nStatistic: {ks_results['statistic']:.4f}\nP-value: {ks_results['p_value']:.4e}\nSample Sizes: n_nDS={ks_results['n_a']}, n_EDTA={ks_results['n_b']}")
    print('***************************************')
    title = f"KS test for DS vs nDS PROphet Scores, normalization: {normalization_type}"
    print(title)
    ks_results = ks_2sided(post_anml_ds_df['PROphetScore_ds'], post_anml_nds_df['PROphetScore'])
    ks_results['title'] = title
    ks_results_list.append(ks_results)
    print(f"Test: {ks_results['test']}\nStatistic: {ks_results['statistic']:.4f}\nP-value: {ks_results['p_value']:.4e}\nSample Sizes: n_DS={ks_results['n_a']}, n_nDS={ks_results['n_b']}")
    print('***************************************')
    title = f"KS test for DS vs uniform distribution, normalization: {normalization_type}"
    print(title)
    ks_results = ks_2sided(post_anml_ds_df['PROphetScore_ds'])
    ks_results['title'] = title
    ks_results_list.append(ks_results)
    print(f"Test: {ks_results['test']}\nStatistic: {ks_results['statistic']:.4f}\nP-value: {ks_results['p_value']:.4e}\nSample Size: n_DS={ks_results['n_a']}")
    print('***************************************')
    title = f"KS test for nDS vs uniform distribution, normalization: {normalization_type}"
    print(title)
    ks_results = ks_2sided(post_anml_nds_df['PROphetScore'])
    ks_results['title'] = title
    ks_results_list.append(ks_results)  
    print(f"Test: {ks_results['test']}\nStatistic: {ks_results['statistic']:.4f}\nP-value: {ks_results['p_value']:.4e}\nSample Size: n_nDS={ks_results['n_a']}")
    print('***************************************')
    title = f"KS test for EDTA vs uniform distribution, normalization: {normalization_type}"
    print(title)
    ks_results = ks_2sided(post_anml_edta_for_distributions['PROphetScore'])
    ks_results['title'] = title
    ks_results_list.append(ks_results)  
    print(f"Test: {ks_results['test']}\nStatistic: {ks_results['statistic']:.4f}\nP-value: {ks_results['p_value']:.4e}\nSample Size: n_EDTA={ks_results['n_a']}")
    print('***************************************')
    
    

    
    matched_samples = pd.read_excel(matched_samples_path)
    matched_edta = matched_samples[matched_samples['SampleRun'] == 'EDTA']
    matched_ds = matched_samples[matched_samples['SampleRun'] == 'DS2']
    matched_nds = matched_samples[matched_samples['SampleRun'] == 'non DS']
    matched_edta_ds = pd.merge(matched_edta, matched_ds, how='inner', on='SubjectId', suffixes=('_edta', '_ds'))

    matched_edta_ds = matched_edta_ds.merge(post_anml_ds_df[['MeasureId_ds', 'PROphetScore_ds']], on='MeasureId_ds')
    matched_edta_ds['PROphetScore_edta'] = matched_edta_ds.merge(post_anml_edta, left_on='MeasureId_edta', right_on='MeasureId')['PROphetScore']

    matched_edta_nds = pd.merge(matched_edta, matched_nds, how='inner', on='SubjectId', suffixes=('_edta', '_nds'))
    matched_edta_nds['PROphetScore_nds'] = matched_edta_nds.merge(post_anml_nds_df, left_on='MeasureId_nds', right_on='MeasureId')['PROphetScore']
    matched_edta_nds['PROphetScore_edta'] = matched_edta_nds.merge(post_anml_edta, left_on='MeasureId_edta', right_on='MeasureId')['PROphetScore']
    matched_edta_ds['PROphetResult_ds'] = matched_edta_ds['PROphetScore_ds'] >= 5
    matched_edta_ds['PROphetResult_edta'] = matched_edta_ds['PROphetScore_edta'] >= 5
    matched_edta_nds['PROphetResult_nds'] = matched_edta_nds['PROphetScore_nds'] >= 5
    matched_edta_nds['PROphetResult_edta'] = matched_edta_nds['PROphetScore_edta'] >= 5

    
    
    if save_path:
        
        plot_confusion_matrix(matched_edta_ds, ground_truth_column='PROphetResult_edta', test_column='PROphetResult_ds', title=confusion_headers[0],
                            ylabel=ylabel, xlabel="DS", save_path= save_path + "confusion_median_only_ds.png")
        plot_confusion_matrix(matched_edta_nds, ground_truth_column='PROphetResult_edta', test_column='PROphetResult_nds', title=confusion_headers[1],
                            ylabel=ylabel, xlabel="NDS", save_path= save_path + "confusion_median_only_nds.png")


        scatter_with_regression(matched_edta_ds['PROphetScore_edta'], 
                                matched_edta_ds['PROphetScore_ds'],
                                xlabel="EDTA PROphet Score",
                                ylabel="DS PROphet Score",
                                title="Scatter Plot with Linear Regression",
                                save_path=save_path,
                                plot_name="scatter_edta_ds")
        
        scatter_with_regression(matched_edta_nds['PROphetScore_edta'], 
                            matched_edta_nds['PROphetScore_nds'],
                            xlabel="EDTA PROphet Score",
                            ylabel="nDS PROphet Score",
                            title="Scatter Plot with Linear Regression",
                            save_path=save_path,
                            plot_name="scatter_edta_nds")

        with open(save_path + "ks_results.txt", "w") as f:
            json.dump(ks_results_list, f, indent=4)
            # for result in ks_results_list:
            #     f.write(f"{result['title']}\n")
            #     f.write(f"Test: {result['test']}\n")
            #     f.write(f"Statistic: {result['statistic']:.4f}\n")
            #     f.write(f"P-value: {result['p_value']:.4e}\n")
            #     if 'n_b' in result:
            #         f.write(f"Sample Sizes: n_a={result['n_a']}, n_b={result['n_b']}\n")
            #     else:
            #         f.write(f"Sample Size: n_a={result['n_a']}\n")
            #     f.write('***************************************\n')



 
if __name__ == "__main__":   
    print("started") 
    base_rfus_path = 'data/adat/base/v_175_OH2026_013.parquet'
    info_path = 'data/excels/20260406_adat_data_samples_with_sample_info.csv'
    plates = [f"OH2025_05{i}" for i in range(1, 8)] 
    plates += [f"OH2026_00{i}" for i in range(1, 6)]
    adats_path = '../SL/adat/mad_transformed_20042026'
    model_path = 'data/models/DCB_nosex_gamma1_rescaled.pkl'
    save_path = "./projects/pre_post_bridging/results/MAD-median/"
    # save_path = ''
    matched_samples_path = 'data/excels/26042026_streck_long_summary.xlsx'
    plate_save_path = "results/plates/median_transformed_debug_26042026"
    confusion_headers = ["EDTA vs DS, MAD-median", "EDTA vs NDS, MAD-median"]
    
    
    analyze_results(base_rfus_path, 
                    info_path, plates,
                    adats_path, model_path,
                    matched_samples_path, 
                    save_path=save_path,
                    normalization_type="MAD-median", 
                    confusion_headers=confusion_headers, 
                    xlabel="Streck Result", 
                    ylabel="EDTA PROphet Result")
