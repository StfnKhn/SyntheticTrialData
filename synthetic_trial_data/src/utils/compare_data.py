import pandas as pd
import numpy as np
from typing import Union

class PairwiseDataComparer:

    def __init__(self):
        self.summary = {}
        self.data_dict = None

    def __check_parameter_values(self, data_dict: Union[None, dict]):
        """checks wether the values inserted, for the parameters are allowed"""
        print(self.data_dict)
        if data_dict != None and self.data_dict == None:
            self.data_dict = data_dict
        elif data_dict == None and self.data_dict == None:
            raise ValueError("data_dict needs to be defined")
        elif data_dict != None and self.data_dict != None:
            raise ValueError("Can't overwrite data_dict") 

    def check_common_columns(self, data_dict: dict = None):
        """Checks the common columns across pairs of dataframes.

        :param data_dict: Keys are the filenames and values are of type pd.DataFrame
        :type data_dict: dict
        :type dfs: list of pd.DataFrame

        :returns: A dictionary containing pairs of dataframes and a list of their common columns.
        :rtype: dict
        """
        #print(self.data_dict, self.data_dict != None)
        self.__check_parameter_values(data_dict)
        result_dict = {}
        common_columns_dict_list = {}
        common_columns_dict_count = {}
        file_names = list(data_dict.keys())
        df_list = list(data_dict.values())
        for i in range(len(df_list)):
            for j in range(i+1, len(df_list)):
                df1 = df_list[i]
                df2 = df_list[j]
                common_col_list = list(set(df1.columns).intersection(df2.columns))
                common_columns_dict_list[(file_names[i], file_names[j])] = common_col_list
                common_columns_dict_count[(file_names[i], file_names[j])] = len(common_col_list)
        result_dict = {"list": common_columns_dict_list, "count": common_columns_dict_count}
        self.summary["common_columns"] = result_dict

        return result_dict

    def check_common_values(self, col_name: str, data_dict: dict = None):
        """ Checks whether at least one value of a specified column is common
        between pairs of dataframes.

        :param data_dict: Keys are the filenames and values are of type pd.DataFrame
        :type data_dict: dict
        :param col_name: Name of the column to check.
        :type col_name: str
        :param unique_id: Name of the unique identifier column.
        :type unique_id: str
        
        :returns: A dictionary containing pairs of dataframes and boolean values
        indicating whether they have at least one common value in the specified column.
        :rtype: dict
        """
        self.__check_parameter_values(data_dict)
        result_dict = {}
        common_columns_dict_count = {}
        file_names = list(data_dict.keys())
        df_list = list(data_dict.values())
        for i in range(len(df_list)):
            for j in range(i+1, len(df_list)):
                df1 = df_list[i]
                df2 = df_list[j]
                common_val_count = df1[df1[col_name].isin(df2[col_name])][col_name].count()
                common_columns_dict_count[(file_names[i], file_names[j])] = common_val_count
        result_dict = {"count": common_columns_dict_count}
        self.summary["common_values"] = result_dict

        return result_dict

    def compare(self, unique_id: str, data_dict: dict = None):
        self.check_common_columns(data_dict)
        self.check_common_values(col_name=unique_id)
