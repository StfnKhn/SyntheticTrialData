import pandas as pd
import numpy as np
import pyreadstat
import os
import logging
from typing import Union, List, Dict
from sas7bdat import SAS7BDAT
from tqdm import tqdm
import json


logger = logging.getLogger(__name__)


class SAS7BDATLoader:
    """A class used to load data from .sas7bdat files into pandas DataFrames.
    """

    @staticmethod
    def read_file(filepath, columns=None):
        if filepath.endswith(".sas7bdat"):
            df = SAS7BDATLoader.read_sas_file(filepath, columns)
        elif filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".pickle") or filepath.endswith(".pkl"):
            df = pd.read_pickle(filepath)
        else:
            raise ValueError("Unsupported file type")
        
        if columns:
            df.columns = df.columns.str.lower()
            df = df[columns]
        
        return df
    
    @staticmethod
    def read_sas_file(filepath, columns=None):
        """Read a .sas7bdat file to a pandas DataFrame.
        
        :param filepath: The path to the .sas7bdat file.
        :type filepath: str
        :return: A DataFrame containing the data from the .sas7bdat file.
        :rtype: pd.DataFrame
        """
        with SAS7BDAT(filepath) as file:
            df, meta = pyreadstat.read_sas7bdat(filepath, usecols=columns)
            df.columns = df.columns.str.lower()
        return df

    @staticmethod
    def batch_load(dir_path: str, file_names: Union[List[str], Dict[str, List[str]]] = None):
        """Load one, all or a list of .sas7bdat files from a directory.

        :param dir_path: The path to the directory containing the .sas7bdat files.
        :type dir_path: str
        :param file_names: A list of specific .sas7bdat files to load. 
            If not provided, all .sas7bdat files in the directory are loaded.
        :type file_names: list, optional
        :param selected_columns: A dictionary where the keys are filenames and the values are lists of columns to select.
        :type selected_columns: dict, optional
        :return: A dictionary where keys are file names and values are 
            corresponding DataFrames containing their data.
        :rtype: dict
        """            
        df_dict = {}
        if file_names is None:
            # Load all files from directory 
            file_list = [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith('.sas7bdat')
            ] 
        elif isinstance(file_names, list):
            if len(file_names) == 1:
                # Read and return the single file
                file_path = os.path.join(dir_path, file_names[0])
                df = SAS7BDATLoader.read_file(file_path)
                return df
            elif len(file_names) > 1:
                # Create list of full file paths
                file_list = [
                    os.path.join(dir_path, f)
                    for f in file_names
                ]
            for file_path in tqdm(file_list):
                df = SAS7BDATLoader.read_file(file_path)
                file_name = os.path.basename(file_path).split('.')[0]
                df_dict[file_name] = df
        elif isinstance(file_names, dict):
            for file_name, columns in tqdm(file_names.items()):
                file_path = os.path.join(dir_path, file_name)
                if os.path.exists(file_path):
                    df = SAS7BDATLoader.read_file(file_path, columns)
                    df.columns = df.columns.str.lower()
                    file_base_name = os.path.basename(file_name).split('.')[0]
                    df_dict[file_base_name] = df
                else:
                    logger.warning(f"{file_name} does not exist in the provided directory.")
                
        logger.info(f"Successfully loaded batch of {len(file_names)} files")          
        return df_dict
