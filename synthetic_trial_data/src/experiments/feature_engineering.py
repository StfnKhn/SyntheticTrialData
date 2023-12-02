import logging
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig, OmegaConf

from synthetic_trial_data.src.preprocessing.custom_transformers import IntegerDayIndexTransformer


logger = logging.getLogger(__name__)


def feature_engineering_pipeline(config, X_train, X_val):

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    config = config["dataset"]["feature_set"]
    unique_id_cols = config['unique_id_cols']
    time_col = config['time_col']

    # Create T_diff feature
    feature_engineering_pipeline = Pipeline([
        ('day_count_generation', IntegerDayIndexTransformer(timestamp_col=time_col, unique_id=unique_id_cols)),
    ])

    X_train = feature_engineering_pipeline.fit_transform(X_train)
    X_val = feature_engineering_pipeline.transform(X_val)

    logger.info("SUCCESS: feature engineering")

    return X_train, X_val