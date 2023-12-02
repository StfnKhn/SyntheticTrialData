import os
import logging
import pandas as pd
import numpy as np
import polars as pl

from tqdm import tqdm

from synthetic_trial_data.src.utils.dataframe_handling import get_common_cols

logger = logging.getLogger(__name__)


class DataMerger:
    @staticmethod
    def compare_and_process_columns(merged_df, col, col_suffix):
        """Compare two columns in a merged dataframe. If they are identical, 
        drop one and rename the other to the original column name. If not,
        print a warning message and leave the columns as they are.

        :param merged_df: The merged DataFrame.
        :param col_1: The first column name.
        :param col_2: The second column name.
        :param col: The original column name before the merge operation.
        :return: The processed DataFrame.
        """
        # Impute any null values in col with values from col_suffix
        merged_df[col].fillna(merged_df[col_suffix], inplace=True)
        merged_df[col_suffix].fillna(merged_df[col], inplace=True)

        # Check that the columns match in both dataframes.
        # If they do, drop one and rename the other.
        if merged_df[col].equals(merged_df[col_suffix]):
            merged_df.drop(columns=col_suffix, inplace=True)
        else:
            print(f"There are discrepancies in the '{col}' columns of the two dataframes.")
        
        return merged_df

    @staticmethod
    def pair_merge(df1, df2, unique_id="subjid", suffix="_y", how="left"):
        """Finds the number of cases per column where a patient has different values for both features across two dataframes.
        Performs a left merge based on the unique identifier column.

        :param df1: First dataframe.
        :type df1: pd.DataFrame
        :param df2: Second dataframe.
        :type df2: pd.DataFrame
        :return: Dictionary with column names as keys and the count of cases with different values for the same patient as values.
        :rtype: dict
        """
        # Merge the two dfs based on the unique_id column using polars pkg
        df1_pl= pl.from_pandas(df1)
        df2_pl = pl.from_pandas(df2)
        merged_df = df1_pl.join(df2_pl, on=unique_id, how=how, suffix=suffix)\
            .to_pandas(use_pyarrow_extension_array=False)
        merged_df = merged_df.where(pd.notnull(merged_df), np.nan)

        # Find columns with different values for the same unique_id
        different_counts = {}
        common_cols = get_common_cols(df1, df2)
        for col in common_cols:
            if col != unique_id and col + suffix in merged_df.columns:
                diff_cases = merged_df[merged_df[col] != merged_df[col+suffix]]
                different_counts[col] = diff_cases.shape[0]
                    
                # Check that the columns match in both dataframes.
                # If they do, drop one and rename the other.
                DataMerger.compare_and_process_columns(merged_df, col, col+suffix)

        return merged_df, different_counts

    @staticmethod
    def sequential_merge(data_dict: dict, how="left", unique_id=None):
        """Executes a sequential merge on a dictionary containing file names as
        keys and loaded datasets as values.
        
        :param data_dict: The dictionary containing file names as keys and loaded
            datasets as values.
        :type data_dict: dict
        :param how: The type of merge to perform (default is "left").
        :type how: str, optional
        :return: A tuple containing the merged dataframe and a dictionary with
            merge dimensions.
        :rtype: tuple
        """
        count = 0
        dimensions_dict = {}
        if unique_id == None:
            unique_id="subjid"
        
        for file_name, df in tqdm(data_dict.items()):
            if count == 0:
                merged_df = df
            else:
                merged_df, _ = DataMerger.pair_merge(
                    df1=merged_df, 
                    df2=df, 
                    unique_id=unique_id,
                    suffix=f"_{file_name}",
                    how=how
                )
            dimensions_dict[count] = merged_df.shape
            count += 1
            
        return merged_df, dimensions_dict