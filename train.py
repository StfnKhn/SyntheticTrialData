import os
import logging
import hydra
from omegaconf import DictConfig
import optuna
import mlflow

from synthetic_trial_data.src.experiments.load_data import load_data
from synthetic_trial_data.src.experiments.data_split import data_split
from synthetic_trial_data.src.experiments.feature_engineering import feature_engineering_pipeline
from synthetic_trial_data.src.experiments.model_handling import get_best_model
from synthetic_trial_data.src.experiments.sample import sample_starting_sequences
from synthetic_trial_data.src.utils.experiments import make_model_checkpoint_dir, check_directory_exists
from synthetic_trial_data.src.utils.mflow import save_and_log_best_model
from synthetic_trial_data.src.experiments.objectives.sequential_models.pars_objective import pars_objective
from synthetic_trial_data.src.experiments.objectives.sequential_models.gpt_objective import gpt_objective
from synthetic_trial_data.src.experiments.objectives.static_models.synthcity import objective
from synthetic_trial_data.src.experiments.objectives.static_models.sdv import sdv_objective
from synthetic_trial_data.src.experiments.auditor import StaticAuditor, SequentialAuditor
from synthetic_trial_data.src.experiments.preprocessing import (
    general_preprocessing,
    get_start_sequences,
    PARSynthesizerPreprocessor,
    TabFormerGPT2Preprocessor,
    SDV_Preprocessor
)
from synthetic_trial_data.src.experiments.experiment_config import STATIC_SYNTHCITY_MODELS, SDV_MODELS, SEQUENTIAL_MODELS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
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
    X_train, X_val = feature_engineering_pipeline(cfg, X_train, X_val)

    # Step 4: Select only the start of event records with the corresponding static info
    X_starting_seq_train = get_start_sequences(cfg, X_train)
    X_starting_seq_val = get_start_sequences(cfg, X_val)

    # Step 5: Model selection static
    if cfg.action.tuning.tune_static_model:
        logger.info("START: hyper-parameter tuning of static models")

        # Step 5.1: Set up TrustIndex auditor
        auditor = StaticAuditor(config=cfg, X_train=X_starting_seq_train, X_val=X_starting_seq_val, cwd=original_cwd).set_up()

        # Step 5.2: Setup mlflow experiment
        mlflow.set_tracking_uri(os.path.join(original_cwd, cfg.mlflow.tracking_uri))
        mlflow.set_experiment(cfg.mlflow.static_model.experiment_name)

        # Check if the base directory and model class directory exist, if not create them
        base_checkpoints_dir = os.path.join(original_cwd, cfg.model_checkpoints_dir)
        check_directory_exists(base_checkpoints_dir, "Please provide a valid base directory for model checkpoints.")

        # Step 5.3: Run hyperparameter tuning
        for model_class in cfg.hyper_parameter_tuning.static_model.model_classes:
            logger.info(f"START HYPER PARAMETER TUNING: model_class {model_class}")

            # Create study
            study = optuna.create_study(direction="maximize")

            # optimize study
            if model_class in STATIC_SYNTHCITY_MODELS:
                study.optimize(
                    lambda trial: objective(trial, auditor, X_starting_seq_train, X_starting_seq_val, model_class, cfg), 
                    n_trials=cfg.hyper_parameter_tuning.n_trials
                )
            elif model_class in SDV_MODELS:
                X_train_pre = get_start_sequences(cfg, X_train, drop_id_columns=False)
                X_val_pre = get_start_sequences(cfg, X_val, drop_id_columns=False)

                # SDV static model specific preprocessing
                X_train_pre, metadata = SDV_Preprocessor(cfg).run(X_train_pre)

                # Optimize study
                study.optimize(
                    lambda trial: sdv_objective(trial, auditor, X_train_pre, metadata, model_class, cfg), 
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


    # Step 6: Model selection dynamic model
    if cfg.action.tuning.tune_sequential_model:
        base_checkpoints_dir = os.path.join(original_cwd, cfg.model_checkpoints_dir)
        check_directory_exists(base_checkpoints_dir, "Please provide a valid base directory for model checkpoints.")

        mlflow.set_tracking_uri(os.path.join(original_cwd, "mlruns"))
        mlflow.set_experiment(cfg.mlflow.sequential_model.experiment_name)

        # Get input sequences with defined max length
        X_train_seq, metadata = PARSynthesizerPreprocessor.run(cfg, X_train)
        X_val_seq, _ = PARSynthesizerPreprocessor.run(cfg, X_val)

        # Pretrain model auditor
        seq_auditor = SequentialAuditor(config=cfg, X_train=X_train_seq, X_val=X_val_seq, cwd=original_cwd).set_up()

        # Hyper parameter tuning
        for model_class in cfg.hyper_parameter_tuning.sequential_model.model_classes:
            if model_class in SEQUENTIAL_MODELS:
                logger.info(f"START HYPER PARAMETER TUNING: model_class {model_class}")
            else: 
                raise ValueError(f"Unsupported model type: {model_class}")

            # Generate model and run specific checkpoint dir
            model_class_dir = os.path.join(base_checkpoints_dir, model_class)
            checkpoint_dir = make_model_checkpoint_dir(model_class_dir)
            
            # Overwrite
            if model_class == "PARSynthesizer":
                # Create and optimize study
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial: pars_objective(trial, seq_auditor, X_train_seq, metadata, cfg), 
                    n_trials=cfg.hyper_parameter_tuning.sequential_model.n_trials
                )
            
            if model_class == "TabFormerGPT2":
                # Get best static model
                model, static_model_class = get_best_model(experiment_name=cfg.mlflow.static_model.experiment_name)

                # Sample starting sequences from best model
                X_synth_start = sample_starting_sequences(model, static_model_class, count=X_starting_seq_train.shape[0], config=cfg)

                # Perform TabFormerGPT2 specific preprocessing
                tabgpt_preprocessor = TabFormerGPT2Preprocessor(
                    cfg,
                    X_train=X_train,
                    X_synth_start=X_synth_start,
                    checkpoint_dir=checkpoint_dir,
                    X_val=X_val,
                    use_secondary_id_groups=cfg.dataset.sequence_schema.use_secondary_id
                )
                dataset, vocab = tabgpt_preprocessor.transform()
                X_synth_start = tabgpt_preprocessor.X_synth_start

                # Set the max number of samples so that it is consistent with X_train_seq, which with it
                # is being compared in the computation of the trust_index
                N_samples_max = X_train_seq.shape[0]

                # Create and optimize study
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial: gpt_objective(trial, seq_auditor, dataset, vocab, X_synth_start, checkpoint_dir, cfg, N_samples_max), 
                    n_trials=cfg.hyper_parameter_tuning.sequential_model.n_trials
                )
            
            # Save model
            save_and_log_best_model(study=study, checkpoint_dir=checkpoint_dir)
            logger.info(f"END HYPER PARAMETER TUNING: model_class {model_class}")


if __name__ == "__main__":
    main()
