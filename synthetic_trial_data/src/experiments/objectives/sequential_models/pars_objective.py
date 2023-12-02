import logging
import optuna
from sdv.sequential import PARSynthesizer
from omegaconf import DictConfig, OmegaConf

from synthetic_trial_data.src.datasets.synthetic_dataset import SyntheticDataset
from synthetic_trial_data.src.utils.mflow import log_trial
from synthetic_trial_data.src.utils.dataframe_handling import ensure_list
from synthetic_trial_data.src.models.PARSynthesizer.utils import ensure_same_size
from synthetic_trial_data.src.experiments.postprocessing import SequentialPostprocessor


logger = logging.getLogger(__name__)

PLUGIN = "PARSynthesizer"

def pars_objective(trial: optuna.Trial, auditor, X_train, metadata, config):
    ID = f"trial_{trial.number}"

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Read parameters from config file
    static_columns = config["dataset"]["feature_set"]["static_columns"]
    unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
    sequence_length = config["preprocessing"]["sequential"]["max_event_count"]

    # Get training params
    param_config = config["PARSynthesizer"]["training_params"]
    epochs = ensure_list(param_config["epochs"])
    sample_size_high = param_config["sample_size"]["high"]
    sample_size_low = param_config["sample_size"]["low"]

    # Set hyperparameter space and obtain next param values
    params = {
        "sample_size": trial.suggest_int(name="sample_size", low=sample_size_low, high=sample_size_high),
        "epochs": trial.suggest_categorical(name="epochs", choices=epochs)
    }

    try:
        # Initialize model
        context_columns = list(set(static_columns) - set(unique_id_cols))
        model = PARSynthesizer(
            metadata,
            context_columns=context_columns,
            verbose=True,
            sample_size=params["sample_size"],
            epochs=params["epochs"],
        )

        # Train model
        model.fit(X_train)

        # Sample ssequences
        num_sequences = X_train[unique_id_cols].drop_duplicates().shape[0]
        logger.info(f"Generation of {num_sequences} sequences with length of {sequence_length} records")
        X_synth = model.sample(num_sequences=num_sequences, sequence_length=sequence_length)
        X_synth = SyntheticDataset(data=X_synth, id=trial.number, experiment_id=trial.number)

    except Exception as e:  # invalid set of params
        print(f"{type(e).__name__}: {e}")
        print(params)
        raise optuna.TrialPruned()

    # Postprocess data types
    postprocessor = SequentialPostprocessor(config)
    X_synth = postprocessor.postprocess(X_synth, model_type="PARSynthesizer", N_samples_max=X_train.shape[0])

    # Compute TrustIndex
    auditor_output = auditor.compute_trust_index(X_synth)

    score = auditor_output["trust_index"]
    fidelity = auditor_output["Fidelity"]
    privacy = auditor_output["Privacy"]
    logger.info(f"{ID}: trust_index={score}, fidelity={fidelity}, Privacy={privacy}")

    trial.set_user_attr("model_class", PLUGIN)
    trial.set_user_attr("model", model)
    trial.set_user_attr("params", params)
    trial.set_user_attr("fidelity", fidelity)
    trial.set_user_attr("privacy", privacy)
    trial.set_user_attr("trust_index", score)
    trial.set_user_attr("X_synth", X_synth)


    # Log trial using mlflow
    log_trial(trial)

    return score
