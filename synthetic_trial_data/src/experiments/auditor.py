import os
import logging
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from typing import List
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import load_from_file, save_to_file
from sdv.sequential import PARSynthesizer

from synthetic_trial_data.src.datasets.synthetic_dataset import SyntheticDataset
from synthetic_trial_data.src.evaluators.trust_index import TrustIndexAuditor, TrustDimension
from synthetic_trial_data.src.utils.experiments import mix_datasets, load_from_pickle, save_as_pickle
from synthetic_trial_data.src.models.PARSynthesizer.utils import get_metadata, ensure_same_size
from synthetic_trial_data.src.experiments.postprocessing import SequentialPostprocessor


logger = logging.getLogger(__name__)

def get_blended_synth_dataset(X_synth: pd.DataFrame, X_train: pd.DataFrame, fractions: List[float]):
    """
    Blend synthetic data with real training data.

    :param X_synth: The synthetic dataset.
    :type X_synth: pd.DataFrame
    :param X_train: The real training dataset.
    :type X_train: pd.DataFrame
    :param fractions: Fractions to mix real data into the synthetic dataset.
    :type fractions: List[float]

    :return: List of blended synthetic datasets.
    :rtype: list
    """
        
    X_synth_blended_list = [
        SyntheticDataset(mix_datasets(X_synth, X_train, frac), id=i, experiment_id=1) 
        for i, frac in enumerate(fractions)
    ]
    
    return X_synth_blended_list


class BaseAuditor:
    """
    Base auditor class for the pretraining of a auditor. 
    The auditor need to be trained on a variety of synthetic data sets to 
    be able to compute the density funciton for each metric that is used to 
    scale and normalize the individual metrics considerd in the auditor.

    :param config: Configuration dictionary.
    :type config: dict
    :param X_train: Training dataset.
    :type X_train: pd.DataFrame
    :param X_val: Validation dataset.
    :type X_val: pd.DataFrame
    :param auditor_type: Type of auditor (e.g., "static_data").
    :type auditor_type: str
    :param cwd: Current working directory.
    :type cwd: str, optional
    """

    def __init__(self, config, X_train, X_val, auditor_type, cwd=None):
        self.cwd = cwd
        self.X_train = X_train
        self.X_val = X_val

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        
        self.config = config
        self.auditor_type = auditor_type
        self.auditor_config = config["auditor"][auditor_type]

        self.path_to_auditor = self.auditor_config["path_to_auditor"]
        self.path_to_model = self.auditor_config["path_to_model"]

        if cwd:
            self.path_to_model = os.path.join(cwd, self.auditor_config["path_to_model"])
            self.path_to_auditor = os.path.join(cwd, self.auditor_config["path_to_auditor"])

    def set_up(self):
        """
        Fits the auditor. 

        :return: The auditor object.
        :rtype: TrustIndexAuditor
        """
        if self.config["action"]["auditor"][self.auditor_type]["use_existing_auditor"]:
            logger.info(f"SUCCESS: Successfully load auditor from {self.path_to_auditor}")
            return load_from_pickle(file_path=self.path_to_auditor)
        
        X_synth = self.generate_synthetic_dataset()
        self.auditor = self.fit_auditor(X_synth)
        save_as_pickle(self.auditor, self.path_to_auditor)
        logger.info("SUCCESS: Successfully pretrained auditor")

        return self.auditor
    

    def fit_auditor(self, X_synth):
        """
        Fit the auditor using synthetic and real data.

        :param X_synth: Synthetic dataset.
        :type X_synth: pd.DataFrame

        :return: The auditor object.
        :rtype: TrustIndexAuditor
        """
        X_synth_blended_list = get_blended_synth_dataset(X_synth, self.X_train, self.auditor_config["fractions"])

        fidelity_evaluators = self.auditor_config["trust_dimensions"]["fidelity_evaluators"]
        privacy_evaluators = self.auditor_config["trust_dimensions"]["privacy_evaluators"]

        fidelity_dimension = TrustDimension(name="Fidelity", evaluators=fidelity_evaluators)
        privacy_dimension = TrustDimension(name="Privacy", evaluators=privacy_evaluators)

        auditor = TrustIndexAuditor(
            self.X_train,
            self.X_val,
            QID=self.config["dataset"]["privacy_audit"]["quasi_identifier_cols"],
            S=self.config["dataset"]["privacy_audit"]["sensitive_cols"],
            reference_size=self.auditor_config["optional_params"].get("reference_size", None),
            unique_id_cols=self.config["dataset"]["feature_set"]["unique_id_cols"],
            event_columns=self.config["dataset"]["feature_set"]["event_columns"],
            cross_corr_col_pairs=self.config["dataset"]["fidelity_audit"]["cross_corr_col_pairs"]
        )

        fidelity_weight = self.auditor_config["trust_dimensions"]["weights"]["fidelity"]
        privacy_weight = self.auditor_config["trust_dimensions"]["weights"]["privacy"]

        auditor.add_dimension(fidelity_dimension, weight=fidelity_weight)
        auditor.add_dimension(privacy_dimension, weight=privacy_weight)
        auditor.fit(X_synth_blended_list)
        auditor.compute_trust_index(X_synth)
        return auditor

    def generate_synthetic_dataset(self):
        """
        Generate a synthetic dataset.

        :raises NotImplementedError: This method needs to be implemented in a child class.
        """
        raise NotImplementedError("This method needs to be implemented in a child class.")


class StaticAuditor(BaseAuditor):
    """
    Framework that allows to set up and pretrain an TrustFormer Auditor for
    static synthetic data.

    The auditor need to be trained on a variety of synthetic data sets to 
    be able to compute the density funciton for each metric that is used to 
    scale and normalize the individual metrics considerd in the auditor.

    :param config: Configuration dictionary.
    :type config: dict
    :param X_train: Training dataset.
    :type X_train: pd.DataFrame
    :param X_val: Validation dataset.
    :type X_val: pd.DataFrame
    :param cwd: Current working directory.
    :type cwd: str, optional

    """

    def __init__(self, config, X_train, X_val, cwd=None):
        super().__init__(config, X_train, X_val, "static", cwd)

    def generate_synthetic_dataset(self):
        """
        Generate static synthetic dataset that should be used to 
        pretrain the auditor.

        :return: Synthetic dataset.
        :rtype: SyntheticDataset
        """
        if self.config["action"]["auditor"]["static"]["use_existing_model"]:
            model = load_from_file(self.path_to_model)

        else:
            private_models = Plugins(categories=["privacy"]).list()
            model_plugin = self.auditor_config["model_plugin"]["name"]
            plugin_params = self.auditor_config["model_plugin"]["params"]
            if model_plugin not in private_models:
                raise ValueError(f"Model {model_plugin} is not supported. Only the following models {private_models}")
            
            model = Plugins().get(model_plugin, **plugin_params)
            loader = GenericDataLoader(self.X_train)
            model.fit(loader)
            save_to_file(self.path_to_model, model)

        X_synth = model.generate(count=self.X_train.shape[0], random_state=42).data
        X_synth = SyntheticDataset(data=X_synth, id=0, experiment_id=1)
        
        return X_synth

    
class SequentialAuditor(BaseAuditor):
    """
    Framework that allows to set up and pretrain an TrustFormer Auditor for
    sequential synthetic data.

    The auditor need to be trained on a variety of synthetic data sets to 
    be able to compute the density funciton for each metric that is used to 
    scale and normalize the individual metrics considerd in the auditor.

    :param config: Configuration dictionary.
    :type config: dict
    :param X_train: Training dataset.
    :type X_train: pd.DataFrame
    :param X_val: Validation dataset.
    :type X_val: pd.DataFrame
    :param cwd: Current working directory.
    :type cwd: str, optional
    """

    def __init__(self, config, X_train, X_val, cwd=None):
        super().__init__(config, X_train, X_val, "sequential", cwd)

    def generate_synthetic_dataset(self):
        """
        Generate sequential synthetic dataset that should be used to 
        pretrain the auditor.
        """
        if self.config["action"]["auditor"]["sequential"]["use_existing_model"]:
            model = load_from_file(self.path_to_model)

        else:
            # Get metadata for the PARSynthesizer model
            time_col = self.config["dataset"]["feature_set"]["time_col"]
            unique_id_cols = self.config["dataset"]["feature_set"]["unique_id_cols"]
            metadata = get_metadata(self.X_train, unique_id_cols, time_col)

            # Initialize model
            static_columns = self.config["dataset"]["feature_set"]["static_columns"]
            context_columns = list(set(static_columns) - set(unique_id_cols))
            model = PARSynthesizer(
                metadata,
                context_columns=context_columns,
                verbose=True,
                sample_size=self.auditor_config["model_plugin"]["params"]["sample_size"],
                epochs=self.auditor_config["model_plugin"]["params"]["epochs"],
            )

            # Train model
            model.fit(self.X_train)

            # Save model
            save_to_file(self.path_to_model, model)

        # Sample sequences
        num_sequences = self.X_train[unique_id_cols].drop_duplicates().shape[0]
        sequence_length = self.config["preprocessing"]["sequential"]["max_event_count"]
        logger.info(f"Generation of {num_sequences} sequences with length of {sequence_length} records")
        X_synth = model.sample(num_sequences=num_sequences, sequence_length=sequence_length)
        X_synth = SyntheticDataset(data=X_synth, id=0, experiment_id=1)

        # Apply postprocessing
        postprocessor = SequentialPostprocessor(self.config)
        X_synth = postprocessor.postprocess(X_synth, model_type="PARSynthesizer", N_samples_max=self.X_train.shape[0])
        self.X_train = postprocessor.postprocess(self.X_train, model_type="PARSynthesizer")

        return X_synth