import os
import logging
import optuna
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datetime import datetime

from synthetic_trial_data.src.models.TabFormerGPT.modules import TabFormerGPT2
from synthetic_trial_data.src.datasets.synthetic_dataset import SyntheticDataset
from synthetic_trial_data.src.utils.mflow import set_seed
from synthetic_trial_data.src.utils.dataframe_handling import ensure_list
from synthetic_trial_data.src.preprocessing.tokenization import StartSequences
from synthetic_trial_data.src.models.TabFormerGPT.sample import ClinicalSequenceGenerator
from synthetic_trial_data.src.utils.dataframe_handling import convert_columns_dtype
from synthetic_trial_data.src.experiments.postprocessing import SequentialPostprocessor


logger = logging.getLogger(__name__)


def gpt_objective(trial: optuna.Trial, auditor, dataset, vocab, start_sequences, checkpoint_dir, config, N_samples_max):

    ID = f"trial_{trial.number}"

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # Static parameters
    logging_steps = config["TabFormerGPT2"]["training_params"]["logging_steps"]

    # Get data set schema
    unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
    static_columns = config["dataset"]["feature_set"]["static_columns"]
    event_columns = config["dataset"]["feature_set"]["event_columns"]
    primary_id_col = config["dataset"]["sequence_schema"]["primary_id"]
    num_cols = config["dataset"]["feature_set"]["num_cols"]

    # Preprocessing
    max_event_count = config["preprocessing"]["sequential"]["max_event_count"]
    min_seq_len = config["preprocessing"]["sequential"]["min_seq_len"]
    sequencing_strategy = config["preprocessing"]["sequential"]["sequencing_strategy"]
    N_samples = config["preprocessing"]["sequential"]["N_samples"]

    # Get sampling parameters
    decoding_strategy = ensure_list(config["TabFormerGPT2"]["sampling"]["decoding_strategy"])
    N_events = config["TabFormerGPT2"]["sampling"]["N_events"]
    batch_size = config["TabFormerGPT2"]["sampling"]["batch_size"]

    # Get hp space
    param_config = config["TabFormerGPT2"]["training_params"]
    epochs = ensure_list(param_config["epochs"])
    lr_scheduler_type = ensure_list(param_config["lr_scheduler_type"])
    lr = param_config["learning_rate"]
    temp = config["TabFormerGPT2"]["sampling"]["temperature"]
    warmup_ratio = ensure_list(param_config["warmup_ratio"])
    params = {
        "learning_rate": trial.suggest_float(name="learning_rate", low=lr["low"], high=lr["high"]),
        "lr_scheduler_type": trial.suggest_categorical(name="lr_scheduler_type", choices=lr_scheduler_type),
        "epochs": trial.suggest_categorical(name="epochs", choices=epochs),
        "warmup_ratio": trial.suggest_categorical(name="warmup_ratio", choices=warmup_ratio),
        "sampling_temperature": trial.suggest_float(name="sampling_temperature", low=temp["low"], high=temp["high"]),
        "decoding_strategy": trial.suggest_categorical(name="decoding_strategy", choices=decoding_strategy),
    }
    logger.info(f"START TRIAL: param = {params}")

    # Set up mlflow run name
    model_class = "TabFormerGPT2"
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{time_stamp}_trial_{trial.number}_{model_class}"
    trial.set_user_attr("run_name", run_name)
    trial.set_user_attr("model_class", model_class)

    with mlflow.start_run(run_name=run_name) as run:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Initialize TabFormerGPT2 instance
        tab_net = TabFormerGPT2(
            special_tokens=vocab.get_special_tokens(),
            vocab=vocab,
            field_ce=True,
            flatten=True,
        )
        
        # Set up Trainer
        collactor_cls = "DataCollatorForLanguageModeling"
        data_collator = eval(collactor_cls)(
            tokenizer=tab_net.tokenizer, mlm=False
        )
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=params["epochs"],
            save_steps=False,
            do_train=True,
            prediction_loss_only=False,
            overwrite_output_dir=True,
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            logging_dir=os.path.join(checkpoint_dir, "logs"),
            learning_rate=params["learning_rate"],
            lr_scheduler_type=params["lr_scheduler_type"],
            warmup_ratio=params["warmup_ratio"],
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=tab_net.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset.train_dataset,
            eval_dataset=dataset.val_dataset,
        )

        #try: 
        # Train model
        trainer.train()
        logger.info("SUCCESS: Successfully trained TabFormerGPT2")
        model = tab_net.model

        # Define Start Sequence for the ClinicalSequenceGenerator
        start_sequences = StartSequences(
            data=start_sequences,
            unique_id=unique_id_cols,
            static_columns=static_columns,
            event_columns=event_columns,
            vocab=vocab,
            max_event_count=max_event_count,
            min_seq_len=min_seq_len,
            sequencing_strategy=sequencing_strategy,
            N_samples=N_samples
        )


        # Generate synthetic data for different seeds and compute trust index
        trust_index_list = []
        fidelity_list = []
        privacy_list = []
        for seed in ensure_list(config["sampling"]["seeds"]["sequential_models"]):
            # Set seed for torch
            set_seed(seed)

            # Sample data
            sequence_generator = ClinicalSequenceGenerator(
                tab_net=tab_net,
                dataset=dataset,
                decoding=params["decoding_strategy"],
                N_events=N_events,
                temperature=params["sampling_temperature"],
                batch_size=batch_size,
                unique_id_col=primary_id_col
            )

            # Postprocess data types
            X_synth = sequence_generator.generate_sequences()
            X_synth = convert_columns_dtype(df=X_synth, columns=num_cols, dtype=float)
            X_synth = SyntheticDataset(data=X_synth, id=trial.number, experiment_id=trial.number)
            logger.info(f"SUCCESS: Generation of {X_synth.shape[0]} record with {N_events} per subject")

            postprocessor = SequentialPostprocessor(config)
            X_synth = postprocessor.postprocess(X_synth, model_type="TabFormerGPT", N_samples_max=N_samples_max)
            logger.info(f"SUCCESS: Postprocessing applied to synthetic samples")

            # Compute TrustIndex
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

        trial.set_user_attr("model", tab_net.model)
        trial.set_user_attr("params", params)
        trial.set_user_attr("fidelity", mean_fidelity)
        trial.set_user_attr("privacy", mean_privacy)
        trial.set_user_attr("trust_index", score)
        trial.set_user_attr("fidelity_variance", variance_of_fidelity)
        trial.set_user_attr("privacy_variance", variance_of_privacy)
        trial.set_user_attr("trust_index_variance", variance_of_trust_index)
        trial.set_user_attr("X_synth", X_synth)

        mlflow.log_params(trial.params)
        mlflow.log_param("model_class", trial.user_attrs["model_class"])
        mlflow.log_metric("trust_index", trial.user_attrs["trust_index"])
        mlflow.log_metric("fidelity", trial.user_attrs["fidelity"])
        mlflow.log_metric("privacy", trial.user_attrs["privacy"])
        mlflow.log_metric("trust_index_variance", trial.user_attrs["trust_index_variance"])
        mlflow.log_metric("fidelity_variance", trial.user_attrs["fidelity_variance"])
        mlflow.log_metric("privacy_variance", trial.user_attrs["privacy_variance"])

        # Set run_id as trial attribute
        trial.set_user_attr("run_id", run.info.run_id)

    return score
