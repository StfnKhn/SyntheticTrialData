import os
import logging
from mlflow.tracking import MlflowClient

from synthetic_trial_data.src.utils.experiments import load_from_pickle, check_path_exists


logger = logging.getLogger(__name__)

def get_best_model(
    experiment_name: str,
    model_name: str = "model.pkl",
    return_model_class: bool = True
):

    # Initialize the client
    client = MlflowClient()

    # Get the experiment ID from its name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        assert f"No experiments for given experiment name {experiment_name}"

    # Query the runs sorted by preferred metric
    runs = client.search_runs([experiment_id], order_by=["metrics.trust_index DESC"])

    # Iterate over runs to find the model artifact
    model = None
    for run in runs:
        run_id = run.info.run_id

        # Construct full artifact URI
        artifact_base_uri = run.info.artifact_uri
        artifact_uri = os.path.join(artifact_base_uri, model_name)

        # Check if the artifact exists
        if os.path.exists(artifact_uri):
            model = load_from_pickle(artifact_uri)
            logger.info(f"SUCCESS: Successfully loaded model from experiment {experiment_name} with run_id={run_id}")
            break

    # Handle case where no model artifact is found
    if model is None:
        raise ValueError(f"No model artifact found for any run in experiment {experiment_name}")

    # If we did not pick the best model, log a warning
    if run != runs[0]:
        logger.warning(f"WARNING: Multiple models in experiment {experiment_name} have the same trust_index. "
                       f"Loaded model is not from the top run. Consider reducing the number of trials to avoid "
                       f"unnecessary computation. Model class: {run.data.params['model_class']}.")

    if return_model_class:
        model_class = run.data.params["model_class"]
        return model, model_class

    return model