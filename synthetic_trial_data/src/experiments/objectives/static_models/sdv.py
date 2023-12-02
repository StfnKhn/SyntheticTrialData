import logging
import optuna
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer

from synthetic_trial_data.src.datasets.synthetic_dataset import SyntheticDataset
from synthetic_trial_data.src.utils.mflow import log_trial
from synthetic_trial_data.src.utils.dataframe_handling import ensure_list



logger = logging.getLogger(__name__)

def sdv_objective(trial: optuna.Trial, auditor, X_train, metadata, model_class, config):

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    ID = f"trial_{trial.number}"

    # Get parameters from config
    new_unique_id_col = config["sdv_models"]["preprocessing"]["new_unique_id_col"]

    # Customize hyperparam space
    if model_class == "GaussianCopular":
        default_distribution = config["sdv_models"]["GaussianCopular"]["training_params"]["default_distribution"]
        params = {
            "default_distribution": trial.suggest_categorical(name="default_distribution", choices=default_distribution)
        }
    if model_class == "TVAE":
        epochs = config["sdv_models"]["TVAE"]["training_params"]["epochs"]
        params = {
            "epochs": trial.suggest_categorical(name="epochs", choices=epochs)
        }

    #try:
    # Get model and set hyperparameters
    if model_class == "GaussianCopular":
        model = GaussianCopulaSynthesizer(
            metadata,
            enforce_min_max_values=True,
            enforce_rounding=True,
            default_distribution=params["default_distribution"]
        )
    if model_class == "TVAE":
        model = TVAESynthesizer(
            metadata,
            enforce_rounding=True,
            epochs=params["epochs"]
        )
    model.fit(X_train)
    
    # Generate synthetic data for different seeds and compute trust index
    trust_index_list = []
    fidelity_list = []
    privacy_list = []
    sampling_seeds = ensure_list(config["sampling"]["seeds"]["static_models"])
    for seed in sampling_seeds:
        X_synth = model.sample(num_rows=X_train.shape[0])
        X_synth = X_synth.drop(columns=[new_unique_id_col])
        
        # Compute TrustIndex
        X_synth = SyntheticDataset(data=X_synth, id=trial.number, experiment_id=trial.number)
        auditor_output = auditor.compute_trust_index(X_synth)
        trust_index_list.append(auditor_output["trust_index"])
        fidelity_list.append(auditor_output["Fidelity"])
        privacy_list.append(auditor_output["Privacy"])
        
    # except Exception as e:  # invalid set of params
    #     print(f"{type(e).__name__}: {e}")
    #     print(params)
    #     raise optuna.TrialPruned()

    # Compute means of the trust dimensions
    mean_trust_index = np.mean(trust_index_list)
    mean_fidelity = np.mean(fidelity_list)
    mean_privacy= np.mean(privacy_list)

    # Compute variance of the trust dimensions
    variance_of_trust_index = np.var(trust_index_list)
    variance_of_fidelity = np.var(fidelity_list)
    variance_of_privacy = np.var(privacy_list)

    score = mean_trust_index
    logger.info(f"{ID}: trust_index={score}, fidelity={mean_fidelity}, Privacy={mean_privacy}")

    trial.set_user_attr("model_class", model_class)
    trial.set_user_attr("model", model)
    trial.set_user_attr("params", params)
    trial.set_user_attr("fidelity", mean_fidelity)
    trial.set_user_attr("privacy", mean_privacy)
    trial.set_user_attr("trust_index", score)
    trial.set_user_attr("fidelity_variance", variance_of_fidelity)
    trial.set_user_attr("privacy_variance", variance_of_privacy)
    trial.set_user_attr("trust_index_variance", variance_of_trust_index)
    trial.set_user_attr("X_synth", X_synth)

    # Log trial using mlflow
    log_trial(trial)

    return score
