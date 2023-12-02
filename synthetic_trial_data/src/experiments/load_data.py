# src/data/load_data.py
import logging
import os
from omegaconf import OmegaConf
from omegaconf import DictConfig

from synthetic_trial_data.src.load_and_merge.data_loader import SAS7BDATLoader
from synthetic_trial_data.src.load_and_merge.data_merger import DataMerger
from synthetic_trial_data.src.preprocessing.preprocessors import DataTypeOptimizer

logger = logging.getLogger(__name__)


def load_data(config, dir_path):

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    selected_columns = config["dataset"]["study_data_config"]
    merge_on = config["dataset"]["merge_tabels_on"]

    # Load data
    data_dict = SAS7BDATLoader.batch_load(dir_path=dir_path, file_names=selected_columns)

    # Merge data
    data_df, dimensions_dict = DataMerger.sequential_merge(data_dict=data_dict, unique_id=merge_on)

    # Datatype correction pipeline
    X = DataTypeOptimizer(include_date_columns=False).transform(data_df)
    print(f"X_transformed.shape: {X.shape}")

    logger.info("SUCCESS: successfully loaded all datasets")
    
    return X
