
import os
import pandas as pd

# def add_measure_id(df_):
#     df = df_.copy()
#     if not 'MeasureId' in df.columns:
#         df['MeasureId'] = df['PlateId'] + '_' + df['PlatePosition']
#         columns = [column for column in df_.columns]
#         columns = [column for column in columns if column != 'MeasureId']
#         columns = ['MeasureId'] + columns
#         df = df[columns]
#         print('Added MeasureId')
#     else:
#         print('No need to add MeasureId')
#     return df

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




def read_parquet_df(folder_path, file_name=''):
    """_summary_ reads a parquet file from the given folder path into a pandas dataframe.
    Args:
        folder_path (str): _description_
        file_name (str, optional): _description_. Defaults to ''.

    Returns:
        pd.DataFrame: dataframe read from the parquet file
    """
    if not file_name:
        file_name = latest_file(folder_path)
    if not folder_path.endswith('/'):
        folder_path += '/' 
    df = pd.read_parquet(folder_path + file_name).reset_index()
    return df


def get_aptamers(df):
    """ returns a list of aptamer column names from the dataframe

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        list: list of aptamer column names
    """
    aptamers = [column for column in df.columns if column[0].isdigit()]
    return aptamers


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