import os
import logging
import numpy as np
import pandas as pd
import polars as pl
from typing import List, Union

from synthetic_trial_data.src.datasets.synthetic_dataset import SyntheticDataset


logger = logging.getLogger(__name__)


def get_common_cols(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """
    Get the common columns between two dataframes.

    :param df1: The first dataframe.
    :type df1: pd.DataFrame
    :param df2: The second dataframe.
    :type df2: pd.DataFrame
    :return: A list of common columns.
    :rtype: list
    """
    return list(set(df1.columns).intersection(df2.columns))


def convert_date_to_unix(df, date_cols):
    """
    Convert date columns in a DataFrame to Unix timestamps.
    
    :param df: DataFrame containing the data
    :type df: pd.DataFrame
    :param date_cols: list of date columns to be converted
    :type date_cols: List[str]
    :return: DataFrame with converted columns
    :rtype: pd.DataFrame
    """
    df = df.copy()
    for col in date_cols:
        # Convert the column to datetime if not already, 
        # and set the time to the start of the day
        df[col] = pd.to_datetime(df[col]).dt.normalize()

        # Convert datetime to unix timestamp (seconds since 1970-01-01 00:00:00)
        df[col] = df[col].astype(int) / 10**9

    return df


def convert_unix_to_date(df, date_cols):
    """
    Convert date_cols from  Unix timestamp back to datetime.date format.
    
    :param df: DataFrame containing the data
    :type df: pd.DataFrame
    :param date_cols: list of Unix timestamp columns to be converted
    :type date_cols: List[str]
    :return: DataFrame with converted columns
    :rtype: pd.DataFrame
    """
    for col in date_cols:
        df[col] = df[col].apply(lambda x: datetime.fromtimestamp(x).date())
        
    return df


def categorize_columns_by_type(df: pd.DataFrame) -> dict:
    """Categorizes columns in a DataFrame based on their data types.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :return: A dictionary containing lists of column names categorized by data type.
    :rtype: dict
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)
    
    column_types = {
        'numeric': [],
        'boolean': [],
        'categorical': [],
        'datetime': []
    }
        
    # Iterate over the columns and categorize them based on their data type
    for col in df.columns:
        if df[col].is_numeric():
            column_types['numeric'].append(col)
        elif df[col].is_boolean():
            column_types['boolean'].append(col)
        elif df[col].is_utf8():
            column_types['categorical'].append(col)
        elif df[col].is_temporal():
            column_types['datetime'].append(col)
        else:
            logger.warning(f"The datatype of the column {col} cant be identified")

    return column_types


def get_col_order_by_type(column_by_types: dict):
    """Return a list of column names, ordered by type: 
        1. categorical
        2. numeric
        3. boolean

    :param column_by_types: A dictionary that maps column types to lists of column names.
    :type column_by_types: dict
    :return: A list of column names ordered by type.
    :rtype: list

    :Example:

        >>> column_by_types = {
        >>>     "categorical": ["cat_col1", "cat_col2"],
        >>>     "numeric": ["num_col1", "num_col2"],
        >>>     "boolean": ["bool_col1", "bool_col2"]
        >>> }
        >>> ordered_columns = get_orderd_coloums(column_by_types)
        >>> print(ordered_columns)
        ['cat_col1', 'cat_col2', 'num_col1', 'num_col2', 'bool_col1', 'bool_col2']
    """
    return column_by_types["categorical"] \
        + column_by_types["numeric"] \
        + column_by_types["boolean"]


def check_type(variable, allowed_types: list):
    """
    Check if the variable is of one of the allowed types.
    
    :param variable: Variable to check the type of.
    :type variable: any
    :param allowed_types: List of allowed types for the variable.
    :type allowed_types: list[type]
    
    :raises ValueError: If the variable is not one of the allowed types.
    
    :example:
    >>> check_type(5, [int, str])
    >>> check_type("hello", [float])
    ValueError: Variable is of type <class 'str'>, but expected one of [<class 'float'>].
    """
    
    if type(variable) not in allowed_types:
        raise ValueError(f"Variable is of type {type(variable)}, but expected one of {allowed_types}.")


def convert_columns_dtype(df: pd.DataFrame, columns: list, dtype: type) -> pd.DataFrame:
    """
    Convert the data type of selected columns in a DataFrame.

    :param df: Input DataFrame whose columns' data type needs to be changed.
    :type df: pd.DataFrame
    :param columns: List of columns whose data type will be changed.
    :type columns: list
    :param dtype: Target data type to which columns will be converted.
    :type dtype: type
    :return: DataFrame with specified columns converted to the desired data type.
    :rtype: pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if dtype == int:
            df[col] = df[col].astype(float).round().astype(int)
        else:
            df[col] = df[col].astype(dtype)
        
    return df


def to_dataframe(data):
    """
    Convert the given data to pandas DataFrame.
    
    If data is already a pandas DataFrame, return it unchanged.
    
    :param data: Data to convert to DataFrame.
    :type data: np.ndarray or pd.DataFrame
    :return: Data as a pandas DataFrame.
    :rtype: pd.DataFrame
    
    :example:
    >>> to_dataframe(np.array([[1,2],[3,4]]))
    DataFrame with the provided data.
    """
    
    # Check if data is of type np.ndarray or pd.DataFrame
    check_type(data, [np.ndarray, pd.DataFrame, SyntheticDataset])
    
    # Convert to DataFrame if it's a numpy array
    if isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    return data


def save_df(df, dir_path, file_name):
    """
    Save a DataFrame to a pickle file at the specified directory path and file name.

    :param df: DataFrame to be saved.
    :type df: pd.DataFrame
    :param dir_path: Directory where the DataFrame should be saved.
    :type dir_path: str
    :param file_name: Name of the pickle file (including the extension, typically ".pkl").
    :type file_name: str
    
    :Example:
    
    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> save_df(df, "/path/to/directory", "data.pkl")

    """
    path = os.path.join(dir_path, file_name)
    df.to_pickle(path)


def read_pickle(dir_path, file_name):
    """
    Read a pickle file from the specified directory path and file name and return it as a DataFrame.

    :param dir_path: Directory where the pickle file is located.
    :type dir_path: str
    :param file_name: Name of the pickle file to be read (including the extension, typically ".pkl").
    :type file_name: str
    :return: DataFrame read from the pickle file.
    :rtype: pd.DataFrame

    :Example:
    
    >>> df = read_pickle("/path/to/directory", "data.pkl")
    >>> print(df)
       A  B
    0  1  4
    1  2  5
    2  3  6

    """
    path = os.path.join(dir_path, file_name)
    return pd.read_pickle(path)


def replace_string_in_column(df: pd.DataFrame, column: str, replacements: dict) -> pd.DataFrame:
    """
    Replace occurrences of target strings with respective replacements in a specific column of a DataFrame.

    :param df: Input DataFrame to search and replace in.
    :type df: pd.DataFrame
    :param column: Name of the column in which to make replacements.
    :type column: str
    :param replacements: Dictionary mapping target strings to their replacements.
    :type replacements: dict

    :return: DataFrame with replaced values in the specified column.
    :rtype: pd.DataFrame
    """
    df = df.copy()
    for target, replacement in replacements.items():
        target_in_value = df[column].str.contains(target)
        df.loc[target_in_value, column] = replacement

    return df


def merge_columns_to_string(
    df: pd.DataFrame,
    cols: List[str],
    merged_col_name: str,
    separator: str = '',
    drop_columns: bool = True
) -> pd.DataFrame:
    """
    Merge the values of a list of columns into a single string column in a DataFrame.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param cols: List of columns to be merged.
    :type cols: List[str]
    :param merged_col_name: The name of the new merged column.
    :type merged_col_name: str
    :param separator: The separator between each column in the merged string. Default is an empty string.
    :type separator: str, optional
    :return: DataFrame with the input columns replaced by the merged column.
    :rtype: pd.DataFrame

    Example:
    .. code-block:: python

        df = pd.DataFrame({
            'col1': ['A', 'B', 'C'],
            'col2': [1, 2, 3],
            'col3': ['X', 'Y', 'Z']
        })

        merged_df = merge_columns_to_string(df, ['col1', 'col2', 'col3'], 'merged_col', '_')
        print(merged_df)

        Output:
           merged_col
        0      A_1_X
        1      B_2_Y
        2      C_3_Z

    """
    df = df.copy()
    
    # Use pandas' "agg" to concatenate all columns in `cols` with the specified separator
    df[merged_col_name] = df[cols].astype(str).agg(separator.join, axis=1)
    
    # Drop the original columns
    if drop_columns:
        df = df.drop(cols, axis=1)
    
    return df


def remove_single_groups(df: pd.DataFrame, group_by_col: str) -> pd.DataFrame:
    """
    Remove groups from a DataFrame that have a group size of one based on a specified column.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param group_by_col: The column to group by.
    :type group_by_col: str
    :return: DataFrame without groups of size one.
    :rtype: pd.DataFrame

    Example:
    .. code-block:: python

        df = pd.DataFrame({
            'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
            'B': [1, 2, 3, 4, 5, 6]
        })

        filtered_df = remove_single_groups(df, 'A')
        print(filtered_df)

    """
    
    # Group by the specified column and filter groups with size more than one
    return df[df.groupby(group_by_col)[group_by_col].transform('size') > 1]

def ensure_list(element):
    """
    Ensures that the input element is a list. If not, wraps it in a list.
    
    :param element: Input which can be any data type.
    :return: A list containing the input element or the input list itself.
    :rtype: list
    """
    return element if isinstance(element, list) else [element]

def intersection_exists(list1, list2):
    """
    Check if the intersection of two lists has at least one element.

    :param list1: First list.
    :param list2: Second list.
    :return: True if there is at least one common element, False otherwise.
    """
    # Convert lists to sets and find intersection
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)

    # Check if intersection is non-empty
    return len(intersection) > 0
