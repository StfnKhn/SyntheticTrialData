import logging
import optuna
import numpy as np
from synthcity.utils.optuna_sample import suggest_all
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins.core.distribution import CategoricalDistribution
from omegaconf import DictConfig, OmegaConf


from synthetic_trial_data.src.models.GMM.model import GMM
from synthetic_trial_data.src.datasets.synthetic_dataset import SyntheticDataset
from synthetic_trial_data.src.utils.mflow import log_trial
from synthetic_trial_data.src.utils.experiments import get_batch_sizes
from synthetic_trial_data.src.utils.dataframe_handling import ensure_list


logger = logging.getLogger(__name__)

def objective(trial: optuna.Trial, auditor, X_train, X_val, plugin, config):
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    model_class = plugin
    ID = f"trial_{trial.number}"
    N_samples = X_train.shape[0]

    # Customize hyperparam space
    if plugin == "gmm":
        covariance_type = ensure_list(config["synthcity_models"]["gmm"]["training_params"]["covariance_type"])
        n_components = config["synthcity_models"]["gmm"]["training_params"]["n_components"]
        max_iter = config["synthcity_models"]["gmm"]["training_params"]["max_iter"]
        params = {
            "covariance_type": trial.suggest_categorical(name="covariance_type", choices=covariance_type),
            "n_components": trial.suggest_int(name="n_components", low=n_components["low"], high=n_components["high"]),
            "max_iter": trial.suggest_int(name="max_iter", low=max_iter["low"], high=max_iter["high"])
        }
    else:
        hp_space = Plugins().get(model_class).hyperparameter_space()
        if plugin == "pategan":
            # Adjust hp_space for batch_size
            hp_space[13].choices = get_batch_sizes(low=16, high=N_samples)

            # Adjust n_iter
            n_iter_bounds = config["synthcity_models"]["pategan"]["training_params"]["n_iter"]
            hp_space[0].high = n_iter_bounds["high"]
            hp_space[0].low = n_iter_bounds["low"]
        if plugin == "ddpm":
            # Adjust hp_space for batch_size
            batch_sizes = get_batch_sizes(low=16, high=N_samples)
            hp_space[1] = CategoricalDistribution(name="batch_size", choices=batch_sizes)

        params = suggest_all(trial, hp_space)
    print(params)

    try:
        # Get model and set hyperparameters
        if model_class == "gmm":
            model = GMM(**params)
            model.fit(X_train)
        else:
            model = Plugins().get(model_class, **params)
            X_train_loaded = GenericDataLoader(X_train)
            model.fit(X_train_loaded)
        
        # Generate synthetic data for different seeds and compute trust index
        trust_index_list = []
        fidelity_list = []
        privacy_list = []
        sampling_seeds = ensure_list(config["sampling"]["seeds"]["static_models"])
        for seed in sampling_seeds: 
            if model_class == "gmm":
                X_synth = model.generate(count=N_samples, random_state=seed)
            else:
                X_synth = model.generate(count=N_samples, random_state=seed)
                X_synth = X_synth.data
            
            # Compute TrustIndex
            X_synth = SyntheticDataset(data=X_synth, id=trial.number, experiment_id=trial.number)
            auditor_output = auditor.compute_trust_index(X_synth)
            trust_index_list.append(auditor_output["trust_index"])
            fidelity_list.append(auditor_output["Fidelity"])
            privacy_list.append(auditor_output["Privacy"])
        
    except Exception as e:  # invalid set of params
        print(f"{type(e).__name__}: {e}")
        print(params)
        raise optuna.TrialPruned()

    # Compute means of the scores
    mean_trust_index = np.mean(trust_index_list)
    mean_fidelity = np.mean(fidelity_list)
    mean_privacy= np.mean(privacy_list)

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
    trial.set_user_attr("X_synth", X_synth)

    # Log trial using mlflow
    log_trial(trial)

    return score
