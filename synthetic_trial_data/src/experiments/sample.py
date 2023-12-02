import logging
from omegaconf import DictConfig, OmegaConf

from synthetic_trial_data.src.experiments.experiment_config import STATIC_SYNTHCITY_MODELS, SDV_MODELS


logger = logging.getLogger(__name__)


def sample_starting_sequences(model, model_class: str, count: int, config):

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    
    if model_class in STATIC_SYNTHCITY_MODELS:
        X_synth_start = model.generate(count=count).data
    elif model_class in SDV_MODELS:
        new_unique_id_col = config["sdv_models"]["preprocessing"]["new_unique_id_col"]
        X_synth_start = model.sample(num_rows=count)
        X_synth_start = X_synth_start.drop(columns=[new_unique_id_col])
    
    logger.info(f"SUCCESS: Successfuly generated {count} data points from model of type {model_class}")

    return X_synth_start