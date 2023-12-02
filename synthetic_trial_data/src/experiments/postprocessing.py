
import logging
from omegaconf import DictConfig, OmegaConf

from synthetic_trial_data.src.utils.dataframe_handling import convert_columns_dtype


logger = logging.getLogger(__name__)


class SequentialPostprocessor:
    def __init__(self, config):
        self.config = config
        self.count_variables = config["dataset"]["feature_set"]["count_variables"]
        self.continous_cols = config["dataset"]["feature_set"]["continous_cols"]
        self.time_col = config["dataset"]["feature_set"]["time_col"]
        self.use_secondary_id = config["dataset"]["sequence_schema"]["use_secondary_id"]
        self.secondary_id_col = config["dataset"]["sequence_schema"]["secondary_id"]
        self.secondary_id_options = config["dataset"]["sequence_schema"]["secondary_id_options"]

        if isinstance(config, DictConfig):
            self.config = OmegaConf.to_container(config, resolve=True)

    def postprocess(self, X_synth, model_type, N_samples_max=None):
        # Ensure correct dtypes
        X_synth = self.ensure_dtypes(X_synth)

        if N_samples_max != None:
            # Crop X_synth to size of X_real
            X_synth = X_synth.iloc[0:N_samples_max]
        
        # Model-specific processing
        if model_type == "TabFormerGPT":
            X_synth = self.postprocess_tabformer_gpt(X_synth)
        elif model_type == "PARSynthesizer":
            X_synth = self.postprocess_parsynthesizer(X_synth)
        
        return X_synth

    def ensure_dtypes(self, X_synth):
        X_synth = convert_columns_dtype(df=X_synth, columns=self.continous_cols, dtype=float)
        X_synth = convert_columns_dtype(df=X_synth, columns=self.count_variables, dtype=int)
        return X_synth

    def postprocess_tabformer_gpt(self, X_synth):
        # Initialize secondary_id column just as a dummy column so that outputs of PARSynthesizer
        # and TabFormerGPT have same format
        if self.use_secondary_id:
            X_synth[self.secondary_id_col] = self.secondary_id_options[0]
        return X_synth

    def postprocess_parsynthesizer(self, X_synth):
        # Add logic specific to PARSynthesizer here
        X_synth.drop(self.time_col, axis=1, inplace=True)

        return X_synth
