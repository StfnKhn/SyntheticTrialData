from typing import List
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

def grouped_train_test_split(df: pd.DataFrame, unique_id_cols: List[str], test_size=0.2, random_state=None):
    """
    Split the dataframe into train and test datasets such that all rows with the same 
    values in unique_id_cols are in either train or test, but not both.
    
    :param df: DataFrame to be split.
    :type df: pd.DataFrame
    :param unique_id_cols: List of columns that together identify unique groups.
    :type unique_id_cols: list
    :param test_size: Proportion of the dataset to include in the test split.
    :type test_size: float
    :param random_state: Seed for random sampling.
    :type random_state: int

    :return: Train and Test datasets.
    :rtype: pd.DataFrame, pd.DataFrame
    """

    # Get unique identifiers based on unique_id_cols
    unique_ids = df[unique_id_cols].drop_duplicates()

    # Split the unique identifiers into train and test sets
    train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=random_state)

    # Use the split identifiers to extract rows from the main dataframe
    X_train = df.merge(train_ids, on=unique_id_cols)
    X_test = df.merge(test_ids, on=unique_id_cols)

    return X_train, X_test

def data_split(config, X: pd.DataFrame):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
    val_size = config["validation"]["val_size"]

    X_train, X_val = grouped_train_test_split(
        X,
        unique_id_cols=unique_id_cols,
        test_size=val_size,
        random_state=42
    )

    logger.info("SUCCESS: successfully splitted data")

    return X_train, X_val