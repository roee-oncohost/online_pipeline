"""_summary_

    Returns:
    _type_: _description_
"""

import os
import json
import time
import shutil
import pandas as pd
from src.bridging import *
from src.bridging_coefficients import *

import warnings
warnings.filterwarnings('ignore')

WORKBOOK_SAMPLE_MATRIX = "Sample Matrix"
HCE_LIST = ['2171-12',
            '2178-55',
            '2194-91',
            '2229-54',
            '2249-25',
            '2273-34',
            '2288-7',
            '2305-52',
            '2312-13',
            '2359-65',
            '2430-52',
            '2513-7']

def read_text_file(path_to_text):
    """
    Read a scanner text file 

    Args:
        path_to_text (str): path to the scanner text file

    Returns:
        str: the text
    """
    with open(path_to_text, 'r') as f:
        text_data = f.read()
    return text_data


def get_sections(text):
    """
    Each scanner file includes 3 dataframes (sections).
    We split the text into 3 texts, each depicting a single dataframe 

    Args:
        text (str): scanner text file

    Returns:
        list: list of text strings
    """
    sections = text.strip().split('*\n')
    return sections



def analyze_lines(lines):
    """
    Every scanner file is parsed into 3 sections (lists of text lines).
    These may include a line for types, header line, header name line,
    and lines of actual data.
    This method parses the given section line by line

    Args:
        lines (list): a list of lines which form a section (dataframe) of the scanner file

    Returns:
        tuple: a tuple of the different kinds of lines
    """
    type_row = None
    header_row = None
    header_name = ''
    data_rows = []
    for line in lines:
        if line.startswith('TYPE\t'):
            type_row = line.split('\t')[1:] 
        elif line.startswith('FEPARAMS\t') or line.startswith('STATS\t') or line.startswith('FEATURES\t'):
            parts = line.split('\t')
            header_name = parts[0]
            header_row = parts[1:]
        elif line.startswith('DATA\t'):
            data_rows.append(line.split('\t')[1:])
    return type_row, header_row, header_name, data_rows


def create_df(type_row, header_row, data_rows, header_name, type_mappings):
    """
    Convert parsed rows to a dataframe
    
    Args: the aforementioned parsed lines, as well as type_mappings (which are later used to rebuild the text file)
        
    Returns:
        pandas DataFrame based on the parsed lines
    """
    
    type_mappings[header_name] = type_row
    df = pd.DataFrame(data_rows, columns=header_row)
    for col, dtype in zip(header_row, type_row):
        if col in df.columns:
            try:
                if dtype == 'integer':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'boolean':
                    df[col] = df[col].map({'0': False, '1': True, 'False': False, 'True': True})
                # 'text' stays as string
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {dtype}: {e}")

    
    return df 


def dataframes_to_text(dataframes, type_mappings):
    """
    Convert dataframes back to the original text format.
    
    Args:
        dataframes: Dictionary of dataframes (e.g., {'FEPARAMS': df1, 'STATS': df2})
        type_mappings: Dictionary mapping dataframe names to their type lists
                      (e.g., {'FEPARAMS': ['text', 'integer', ...], ...})
    
    Returns:
        String in the original text format
    """
    sections = []
    
    for name, df in dataframes.items():
        lines = []
        
        # Get the type mapping for this dataframe
        types = type_mappings.get(name, ['text'] * len(df.columns))
        
        # TYPE row
        type_line = 'TYPE\t' + '\t'.join(types)
        lines.append(type_line)
        
        # Header row
        header_line = name + '\t' + '\t'.join(df.columns)
        lines.append(header_line)
        
        # DATA rows
        for _, row in df.iterrows():
            # Convert values back to strings, handling special cases
            values = []
            for val in row:
                if pd.isna(val):
                    values.append('')
                elif isinstance(val, bool):
                    values.append('1' if val else '0')
                else:
                    values.append(str(val))
            
            data_line = 'DATA\t' + '\t'.join(values)
            lines.append(data_line)
        
        sections.append('\n'.join(lines))
    
    # Join sections with asterisk separator
    return '\n*\n'.join(sections) + '\n'


def make_text_file(file_name, text):
    """
    write a text file from the reconstituted text

    Args:
        file_name (str): name of file to be written
        text (str): text to be written to file
    """
    with open(file_name, 'w') as f:
        f.write(text)


def read_workbook(workbook_path) -> pd.DataFrame:
    ### Create DF from Sheet-3 
    sheet_name_tab3 = 'Tab 3 - Assayed Sample List'
    col_smb_tab3 = 'A, F, J'  # v7    
    col_names_tab3 = ['Sample Number', 'SampleId', WORKBOOK_SAMPLE_MATRIX]
    
    # Read the excel with convert the value of the column SampleID to string
    df_sheet3 = pd.read_excel(workbook_path, sheet_name=sheet_name_tab3, 
                              usecols=col_smb_tab3, dtype={'SampleId': str})
    
    # Skip IF One is EMPTY
    df_data_tab3 = df_sheet3[col_names_tab3].dropna()
    
    ### Create DF from Sheet-4 
    sheet_name_tab4 = 'Tab 4 - Plate Map'
    col_smb_tab4 = 'A, E, H, K'
    # Read the excel with convert the value of column NAME to string
    df_sheet4 = pd.read_excel(workbook_path, sheet_name=sheet_name_tab4, 
                              usecols=col_smb_tab4, skiprows=6, dtype={'NAME': str})
    df_data_tab4 = df_sheet4[df_sheet4['NAME'].notna()]
    df_data_tab4['Slide #'] = df_data_tab4['Slide #'].ffill()
    # Create a DICT of the Aliquots-Barcodes
    excel_df = pd.merge(df_data_tab3,
                    df_data_tab4,
                    how='inner',
                    left_on='SampleId', right_on='NAME').drop('NAME', axis=1)
    excel_df['slide'] = excel_df['Slide #']
    excel_df.rename(columns={'PDF Subarray': 'pdf_subarray'}, inplace=True)
    return excel_df



def match_file(row, files):
    """
    Matches each well in the workbook to the corresponding scanner file 

    Args:
        row (pd.DataFrame row): a row in tab 4 of the workbook
        files (list): list of the scanner files in the folder

    Returns:
        str: name of the file corresponding to the row
    """
    for f in files:
        if str(row['slide']) in f and f.endswith(row['pdf_subarray'] + '.txt'):
            return f
    return None

def match_probe_aptamer(df):
    """
    matches the probe names in the "FEATURES" dataframe to the corresponding aptamers

    Args:
        df (pd.DataFrame): the "FEATURES" dataframe

    Returns:
        dict: a mapping from probe names to aptamers
    """
    probe_names = list(set(df['ProbeName'].to_list()))
    probe_names = [probe_name for probe_name in probe_names if probe_name.startswith('anti-')]
    probe_to_aptamer_dict = {probe_name: probe_name.split('anti-')[1].split('_')[0] for probe_name in probe_names}
    return probe_to_aptamer_dict


def transform_feature_df(feature_df, params, hce_list, method='MAD'):
    
    """
    Transform the FEATURES section DataFrame by aggregating probes to aptamers
    and applying parameterized adjustments to selected targets.

    This function maps probe-level measurements to aptamer-level identifiers,
    aggregates the data accordingly, and applies a transformation (e.g., MAD-based)
    to a subset of aptamers. Aptamers listed in `hce_list` are excluded from
    transformation. The original column structure is restored before returning.

    Parameters
    ----------
    feature_df : pandas.DataFrame
        Input DataFrame representing the FEATURES section, expected to include
        a 'ProbeName' column and measurement columns (e.g., signal values).

    params : dict
        Parameter set controlling the transformation (typically loaded from JSON).
        Passed to the underlying transformation method.

    hce_list : list
        List of aptamer identifiers to exclude from transformation.

    method : str, optional
        Transformation method to apply (e.g., 'MAD'). Passed to
        `transform_by_method`.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with the same column structure as the input.

    Notes
    -----
    - Probe-to-aptamer mapping is derived via `match_probe_aptamer`.
    - Aggregation from probe level to aptamer level is performed by `aggregate`.
    - Only aptamers not in `hce_list` are transformed.
    - After transformation, the DataFrame is reduced back to the original columns,
      discarding intermediate columns (e.g., 'aptamer').
    """
    
    df = feature_df.copy()
    probe_to_aptamer_dict = match_probe_aptamer(df)
    original_columns = df.columns
    df['aptamer'] = df['ProbeName'].map(probe_to_aptamer_dict).fillna('other')
    df = aggregate(df)
    aptamers = list(probe_to_aptamer_dict.values())
    aptamers_to_alter = [aptamer for aptamer in aptamers if aptamer not in hce_list]
    df = transform_by_method(df, params, aptamers_to_alter, method=method)
    df = df[original_columns]
    return df



def alter_scanner_file(input_path, file, output_path, conversion_coefficients, hce_list, already_handled_files=[], method='MAD'):
    """
    Parse, transform, and rewrite a single scanner output text file.

    This function reads a scanner-generated text file, splits it into logical sections,
    converts each section into a structured DataFrame, and applies transformations to
    the 'FEATURES' section using provided conversion coefficients. The modified content
    is then reassembled into text format and written to the output directory.

    Parameters
    ----------
    input_path : str
        Path to the directory containing the input text file.

    file : str
        Name of the text file to process.

    output_path : str
        Path to the directory where the modified file will be written.

    conversion_coefficients : dict
        Parameters used to transform feature values (typically loaded from a JSON file).
        Passed to the feature transformation logic.

    hce_list : list
        List of HCE identifiers used during feature transformation.

    already_handled_files : list, optional
        List tracking files that have already been processed. The current file is appended
        to this list. Note: this list is modified in place.

    method : str, optional
        Method used for feature transformation (e.g., 'MAD').

    Returns
    -------
    list
        Updated list of already handled files, including the current file.

    Notes
    -----
    - The file is decomposed into sections using `get_sections`, and each section is parsed
      via `analyze_lines` and `create_df`.
    - Only the section with header 'FEATURES' is transformed using `transform_feature_df`.
    - All sections are reconstructed into text using `dataframes_to_text`.
    - Output is written using `make_text_file`, overwriting if the file already exists.
    - The `already_handled_files` argument is mutable and accumulates state across calls.
    """
    file_path = os.path.join(input_path, file)
    text = read_text_file(file_path)
    print(f"Altered: {file}")
    sections = get_sections(text)
    dataframes_dict = {}
    type_mappings = {}
    for section in sections:
        lines = section.strip().split('\n')
        type_row, header_row, header_name, data_rows = analyze_lines(lines)
        df = create_df(type_row, header_row, data_rows, header_name, type_mappings)
        if header_name=='FEATURES':
            df = transform_feature_df(df, conversion_coefficients, hce_list, method=method)
 
        dataframes_dict[header_name] = df
    reconstituted_text = dataframes_to_text(dataframes_dict, type_mappings)
# else:
    # reconstituted_text = text
    output_file_path = os.path.join(output_path, file)
    # if file not in already_handled_files:
    make_text_file(output_file_path, reconstituted_text)
    already_handled_files.append(file)
    return already_handled_files




def alter_scanner_files(text_path, workbook_path, params_paths, new_plate_path, hce_list = HCE_LIST, method='MAD'): 
    
    """
    Process and modify scanner-generated text files based on sample type–specific parameters.

    This function reads a workbook that maps samples to file identifiers, matches each sample
    to its corresponding scanner output (.txt) file, and applies transformations using
    parameter sets defined per sample type. Modified files are written to the output directory,
    while unprocessed files are copied as-is. The original workbook is also copied to the
    output location.

    Parameters
    ----------
    text_path : str
        Path to the directory containing raw scanner text files (.txt).

    workbook_path : str
        Path to the workbook file containing sample metadata, including sample type and
        identifiers used to match text files.

    params_paths : dict
        Mapping from sample type (str) to a JSON file path containing parameters used to
        alter files of that type.

    new_plate_path : str
        Path to the directory where modified and copied files will be written.
        The directory is created if it does not exist.

    hce_list : list, optional
        List of HCE identifiers used during file alteration. Default is HCE_LIST.

    method : str, optional
        Method used for alteration (e.g., 'MAD'). Passed through to downstream processing
        functions.

    Returns
    -------
    None
        The function operates via side effects: writing modified files and copying
        unmodified ones into the output directory.

    Notes
    -----
    - Each sample type is processed using its corresponding parameter file.
    - Files that are not matched to any sample or already processed are copied unchanged.
    - Matching between workbook entries and text files is handled by `match_file`.
    - File transformation logic is delegated to `alter_scanner_file`.
    """

    workbook_df = read_workbook(workbook_path)
    
    os.makedirs(new_plate_path, exist_ok=True)
    shutil.copy(workbook_path, new_plate_path)


    files = [
        f for f in os.listdir(text_path)
        if os.path.isfile(os.path.join(text_path, f))
    ]
    text_files = [file for file in files if file.endswith('.txt')]
    already_handled_files = []
    for sample_type, params_path in params_paths.items():
        with open(params_path, 'r') as fp:
                params = json.load(fp)
        samples_to_alter  = workbook_df[workbook_df[WORKBOOK_SAMPLE_MATRIX].isin([sample_type])]

        samples_to_alter['filename'] = samples_to_alter.apply(match_file, axis=1, args=(text_files,))
        type_files = samples_to_alter['filename'].to_list()
        reconstituted_texts = {}
        for file in type_files:
            # output_file_path = os.path.join(output_path, file)
            already_handled_files.append(alter_scanner_file(text_path, file, new_plate_path, params, hce_list, already_handled_files, method=method))
            # reconstituted_texts[file] = reconstituted_text

        # reconstituted_texts["streck_files"] = type_files 
    files_to_copy = [text_file for text_file in text_files if text_file not in already_handled_files]
    for file in files_to_copy:
        shutil.copy(os.path.join(text_path, file), new_plate_path)
    print('Done!')




if __name__ == '__main__':
    print(os.getcwd())
    
    # i = 4    # i = 1
    
    # plates = os.listdir('data/plates/original_')
    # for i in range(1, 7):
    #     start = time.time()
            
    #     # i = 2
    #     if i == 3:
    #         alter_scanner_files(f'data/plates/OH2025_05{i}',
    #                 f'data/plates/OH2025_05{i}/OH2025_05{i} Workbook.xlsx',
                    
    #                 {'Streck': 'data/conversion_coefficients/MAD_median_ds_coefficients_plates_OH2025_051_to_OH_2026_003_04022026.json',
    #                 'nds': 'data/conversion_coefficients/MAD_median_nds_coefficients_plates_OH2025_054_OH2025_055_OH_2026_001_OH_2026_002_04022026.json'},
    #                 f'data/plates/debugging/OH2025_05{i}_only_streck_bridging',
                    
    #     )
            # alter_scanner_files_MAD_bridging(f'data/plates/multiple_types_experiment/OH2025_05{i}_altered_workbook',
            #                     f'data/plates/multiple_types_experiment/OH2025_05{i}_altered_workbook/OH2025_05{i} Workbook.xlsx',
            #                     {'Streck': 'data/conversion_coefficients/MAD_median_streck_coefficients.json',
            #                     'nds': 'data/conversion_coefficients/nds_params.json'},
            #                     f'data/plates/multiple_types_experiment/results/testing_transformation/OH2025_05{i}_for testing',
                                
            # )
            # print(f"Time for plate OH2025_05{i}: {time.time() - start} seconds")
    # i = 1
        # alter_scanner_files_normalization_bridging(f'data/plates/OH2025_05{i}',
        #                 f'data/plates/OH2025_05{i}/OH2025_05{i} Workbook.xlsx',
        #                 'data/conversion_coefficients/prehyb_normalization_edta_streck.json',
        #                 f'data/plates/normalization_bridging/OH2025_05{i}_normalized',
        #                 # ['A6', 'B4', 'C1', 'C5', 'D3', 'E5', 'F1', 'H2', 'H4']
        # )
    
    # alter_scanner_files_normalization_bridging(f'data/plates/OH2025_05{i}',
    #                 f'data/plates/OH2025_05{i}/OH2025_05{i} Workbook.xlsx',
    #                 'data/conversion_coefficients/prehyb_normalization_edta_streck.json',
    #                 f'data/plates/normalization_bridging/OH2025_05{i}_testing_value_processing',
    #                 # ['A6', 'B4', 'C1', 'C5', 'D3', 'E5', 'F1', 'H2', 'H4']
    # )
    # read_workbook('./data/test_files/test Workbook.xlsx')
    # ['A1'], 'A3', 'B2', 'B5', 'C2', 'C3', 'C4', 'D4', 'E1',
#                         'E4', 'E5', 'F2', 'F3', 'G1', 'G2', 'H3', 'H4', 'H5'])
