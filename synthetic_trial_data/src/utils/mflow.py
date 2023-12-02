import os
import logging
from datetime import datetime
import mlflow
import torch
from synthcity.utils.serialization import save_to_file

from synthetic_trial_data.src.utils.experiments import save_as_pickle
from synthetic_trial_data.src.experiments.experiment_config import STATIC_SYNTHCITY_MODELS, SDV_MODELS

logger = logging.getLogger(__name__)


def log_trial(trial, run_name=None):
    if not run_name:
        time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_class = trial.user_attrs["model_class"]
        run_name = f"{time_stamp}_trial_{trial.number}_{model_class}"
    
    trial.set_user_attr("run_name", run_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(trial.params)
        mlflow.log_param("model_class", trial.user_attrs["model_class"])
        mlflow.log_metric("trust_index", trial.user_attrs["trust_index"])
        mlflow.log_metric("fidelity", trial.user_attrs["fidelity"])
        mlflow.log_metric("privacy", trial.user_attrs["privacy"])
        if model_class != "PARSynthesizer":
            mlflow.log_metric("trust_index_variance", trial.user_attrs["trust_index_variance"])
            mlflow.log_metric("fidelity_variance", trial.user_attrs["fidelity_variance"])
            mlflow.log_metric("privacy_variance", trial.user_attrs["privacy_variance"])
        
        # Set run_id as trial attribute
        trial.set_user_attr("run_id", run.info.run_id)

def save_and_log_best_model(study, checkpoint_dir):
    try:
        best_trial = study.best_trial
    except Exception as e:  # invalid set of params
        logger.warning(f"{type(e).__name__}: {e}")
        logger.warning(f"Due to above error no valid model was being found")
        return None

    model_class = best_trial.user_attrs["model_class"]
    best_run_id = best_trial.user_attrs["run_id"]
    best_run_name = best_trial.user_attrs["run_name"]

    # save snythetic dataset
    X_synth = best_trial.user_attrs["X_synth"]
    X_synth_file_path = os.path.join(checkpoint_dir, "X_synth.pkl")
    save_as_pickle(X_synth, X_synth_file_path)

    # Save model_file
    model_file_path = os.path.join(checkpoint_dir, "model.pkl")
    if model_class in STATIC_SYNTHCITY_MODELS:
        save_to_file(model_file_path, best_trial.user_attrs["model"])
    elif model_class in SDV_MODELS:
        save_to_file(model_file_path, best_trial.user_attrs["model"])
    elif model_class in ["PARSynthesizer"]:
        save_to_file(model_file_path, best_trial.user_attrs["model"])
    elif model_class in ["TabFormerGPT2"]:
        model_file_path = os.path.join(checkpoint_dir, "model")
        best_trial.user_attrs["model"].save_pretrained(model_file_path)
    
    # Log the model in MLflow
    with mlflow.start_run(run_id=best_run_id):
        # Log vocabulary if available
        if model_class == "TabFormerGPT2":
            mlflow.log_artifact(os.path.join(checkpoint_dir, "vocab.nb"))

        # Log model
        mlflow.log_artifact(model_file_path)

        # Log synthetic dataset
        mlflow.log_artifact(X_synth_file_path)

    logger.info(f"SUCCESS: Logged best model from run: {best_run_name} with id {best_run_id}")

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    