import os
import logging
import hydra
from omegaconf import DictConfig
import optuna
import mlflow

from synthetic_trial_data.src.experiments.load_data import load_data
from synthetic_trial_data.src.experiments.data_split import data_split
from synthetic_trial_data.src.utils.experiments import make_model_checkpoint_dir, check_directory_exists
from synthetic_trial_data.src.utils.mflow import save_and_log_best_model
from synthetic_trial_data.src.experiments.objectives.static_models.synthcity import objective
from synthetic_trial_data.src.experiments.objectives.static_models.sdv import sdv_objective
from synthetic_trial_data.src.experiments.auditor import StaticAuditor
from synthetic_trial_data.src.experiments.preprocessing import general_preprocessing, drop_id_columns, SDV_Preprocessor
from synthetic_trial_data.src.experiments.experiment_config import STATIC_SYNTHCITY_MODELS, SDV_MODELS
from synthetic_trial_data.src.utils.dataframe_handling import intersection_exists


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config_static_only")
def main(cfg: DictConfig):

    # Set path to data
    original_cwd = hydra.utils.get_original_cwd()
    path_to_data = os.path.abspath(os.path.join(original_cwd, cfg.dataset.path_to_data))

    # Step 1: Load data
    data = load_data(cfg, path_to_data)

    # Step 2: Datasplit
    X_train, X_val = data_split(cfg, X=data)

    # Step 3: Preprocessing
    X_train, X_val = general_preprocessing(X_train, X_val)

    # SDV static model specific preprocessing
    sdv_models_exists = intersection_exists(
        cfg.hyper_parameter_tuning.static_model.model_classes,
        SDV_MODELS
    )
    if sdv_models_exists:
        X_train_sdv, metadata = SDV_Preprocessor(cfg).run(X_train)

    # Drop id column
    X_train = drop_id_columns(cfg, X_train)
    X_val = drop_id_columns(cfg, X_val)
    
    # Step 4: Model selection static
    logger.info("START: hyper-parameter tuning of static models")

    # Step 4.1: Set up TrustIndex auditor
    auditor = StaticAuditor(config=cfg, X_train=X_train, X_val=X_val, cwd=original_cwd).set_up()

    # Step 4.2: Setup mlflow experiment
    mlflow.set_tracking_uri(os.path.join(original_cwd, cfg.mlflow.tracking_uri))
    mlflow.set_experiment(cfg.mlflow.static_model.experiment_name)

    # Check if the base directory and model class directory exist, if not create them
    base_checkpoints_dir = os.path.join(original_cwd, cfg.model_checkpoints_dir)
    check_directory_exists(base_checkpoints_dir, "Please provide a valid base directory for model checkpoints.")

    # Step 4.3: Run hyperparameter tuning
    for model_class in cfg.hyper_parameter_tuning.static_model.model_classes:
        logger.info(f"START HYPER PARAMETER TUNING: model_class {model_class}")

        # Create study
        study = optuna.create_study(direction="maximize")

        # optimize study
        if model_class in STATIC_SYNTHCITY_MODELS:
            study.optimize(
                lambda trial: objective(trial, auditor, X_train, X_val, model_class, cfg), 
                n_trials=cfg.hyper_parameter_tuning.n_trials
            )
        elif model_class in SDV_MODELS:
            # Optimize study
            study.optimize(
                lambda trial: sdv_objective(trial, auditor, X_train_sdv, metadata, model_class, cfg), 
                n_trials=cfg.hyper_parameter_tuning.n_trials
            )
        else:
            raise ValueError(f"Unsupported model type: {model_class}")
        
        # Save model
        model_class_dir = os.path.join(base_checkpoints_dir, model_class)
        checkpoint_dir = make_model_checkpoint_dir(model_class_dir)
        save_and_log_best_model(study=study, checkpoint_dir=checkpoint_dir)
        
        logger.info(f"END HYPER PARAMETER TUNING: model_class {model_class}")
    logger.info("SUCCESS: Finished hyper-parameter tuning of static models")


if __name__ == "__main__":
    main()
