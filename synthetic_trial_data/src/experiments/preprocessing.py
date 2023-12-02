# src/data/load_data.py
import os
import logging
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline
from sdv.metadata import SingleTableMetadata

from synthetic_trial_data.src.preprocessing.preprocessors import Imputer
from synthetic_trial_data.src.preprocessing.data_sequencing import PARSynthesizerInputSequencer, StartSequencer
from synthetic_trial_data.src.utils.trial_data import convert_numeric_static_to_categories
from synthetic_trial_data.src.models.PARSynthesizer.utils import get_metadata
from synthetic_trial_data.src.utils.trial_data import assign_ids_by_grouping, add_unique_id_column
from synthetic_trial_data.src.models.TabFormerGPT.vocab import Vocabulary, VocabularyBasisData
from synthetic_trial_data.src.preprocessing.tokenization import ClinicalTrialDataset
from synthetic_trial_data.src.preprocessing.custom_transformers import ContinuousBinner, CountVariableTransformer
from synthetic_trial_data.src.utils.dataframe_handling import merge_columns_to_string


logger = logging.getLogger(__name__)

def general_preprocessing(X_train, X_val):

    # Impute missing data
    imputer = Imputer()
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    logger.info("SUCCESS: preprocessing finished")

    return X_train, X_val

def drop_id_columns(config: dict, X: pd.DataFrame):
    
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    static_columns = config["dataset"]["feature_set"]["static_columns"]
    X = X[static_columns]
    logger.info(f"Select only the following columns for the starting sequences: {X.columns}")

    return X

def get_start_sequences(config: dict, X: pd.DataFrame, drop_id_columns=True):

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    static_columns = config["dataset"]["feature_set"]["static_columns"]
    event_columns = config["dataset"]["feature_set"]["event_columns"]
    unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
    time_col = config["dataset"]["feature_set"]["time_col"]
    first_event_key_col = config["dataset"]["sequence_schema"]["first_event_key_col"]
    first_event_key_value = config["dataset"]["sequence_schema"]["first_event_key_value"]

    # Get only start of sequence records
    start_sequencer = StartSequencer(
        static_columns=static_columns,
        event_columns=event_columns,
        unique_id_cols=unique_id_cols,
        datetime_col=time_col,
        first_event_key_col=first_event_key_col,
        first_event_key_value=first_event_key_value,
        only_keep_key_value=True,
    )
    start_sequences = start_sequencer.create_sequences(X, drop_id_columns)

    return start_sequences


class SDV_Preprocessor:
    """
    A preprocessor for the SDV model.

    This class contains utility methods to process data for input to the SDV model.
    """

    def __init__(self, config: dict):
        self.config = config
        if isinstance(config, DictConfig):
            self.config = OmegaConf.to_container(config, resolve=True)

        self.new_unique_id_col = self.config["sdv_models"]["preprocessing"]["new_unique_id_col"]

    def merge_unique_id_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Merge unique ID columns into one column, if more than unique_id is built up 
        from multiple columns.

        :param X: Input dataset.
        :type X: pd.DataFrame

        :return: DataFrame with merged unique ID columns.
        :rtype: pd.DataFrame
        """
        X = X.copy()
        unique_id_cols = self.config["dataset"]["feature_set"]["unique_id_cols"]
        if len(unique_id_cols) > 1:
            X = merge_columns_to_string(
                X,
                cols=unique_id_cols,
                merged_col_name=self.new_unique_id_col,
                separator='_',
                drop_columns=True
            )
        else:
            X = X.rename(columns={unique_id_cols[0]: self.new_unique_id_col})
        
        return X

    def get_metadata(self, X: pd.DataFrame) -> SingleTableMetadata:
        """
        Extract metadata from the given dataset for the SDV.

        :param X: Input dataset.
        :type X: pd.DataFrame

        :return: Metadata for the provided dataset.
        :rtype: SingleTableMetadata
        """
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=X)
        
        metadata.update_column(column_name=self.new_unique_id_col, sdtype='id')
        metadata.set_primary_key(column_name=self.new_unique_id_col)
        
        return metadata

    def run(self, X: pd.DataFrame) -> (pd.DataFrame, SingleTableMetadata):
        """
        Execute the complete preprocessing pipeline for the SDV.

        :param X: Input dataset.
        :type X: pd.DataFrame

        :return: Tuple of processed DataFrame and its associated metadata.
        :rtype: (pd.DataFrame, SingleTableMetadata)
        """
        # Merge unique ID columns
        processed_data = self.merge_unique_id_columns(X)
        
        # Get meta data
        metadata = self.get_metadata(processed_data)
        
        return processed_data, metadata


class PARSynthesizerPreprocessor:
    """
    A preprocessor for the PARSynthesizer model.

    This class contains various utility methods to process data for input to the PARSynthesizer model.
    """

    @staticmethod
    def get_start_sequences(config: dict, X: pd.DataFrame):
        """
        Extract sequences with defined maximum sequence length from the given DataFrame.

        :param config: Configuration dictionary.
        :type config: dict
        :param X: Input dataset.
        :type X: pd.DataFrame

        :return: Processed DataFrame with sequences.
        :rtype: pd.DataFrame
        """

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        static_columns = config["dataset"]["feature_set"]["static_columns"]
        event_columns = config["dataset"]["feature_set"]["event_columns"]
        unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
        time_col = config["dataset"]["feature_set"]["time_col"]
        max_event_count = config["preprocessing"]["sequential"]["max_event_count"]
        min_seq_len = config["preprocessing"]["sequential"]["min_seq_len"]

        # Get only start of sequence records
        start_sequencer = PARSynthesizerInputSequencer(
            static_columns=static_columns,
            event_columns=event_columns,
            unique_id_cols=unique_id_cols,
            datetime_col=time_col,
            max_event_count=max_event_count,
            min_seq_len=min_seq_len,
        )
        start_sequences = start_sequencer.create_sequences(X)

        logger.info("SUCCESS: generated input sequences of length for PARSynthesizer")

        return start_sequences
    
    @staticmethod
    def numerical_static_column_handling(config: dict, X: pd.DataFrame):
        """
        Convert numerical static columns to categories.

        As the PARSynthesizer works only with categorical static data, this method 
        transforms the already binned numerical columns to categorical datatype.

        :param config: Configuration dictionary.
        :type config: dict
        :param X: Input dataset.
        :type X: pd.DataFrame

        :return: DataFrame with numerical static columns converted to categorical.
        :rtype: pd.DataFrame
        """
        unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
        X = convert_numeric_static_to_categories(X, unique_id_cols)
        return X

    @staticmethod
    def get_metadata(config, X: pd.DataFrame):
        """
        Extract metadata from the given dataset for the PARSynthesizer.

        :param config: Configuration dictionary.
        :type config: dict
        :param X: Input dataset.
        :type X: pd.DataFrame

        :return: Metadata for the provided dataset.
        :rtype: SingleTableMetadata
        """

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        
        unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
        time_col = config["dataset"]["feature_set"]["time_col"]

        # Define Metadata for sequential dataset
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(X)

        # Set sequence keys
        for col in unique_id_cols:
            metadata.update_column(column_name=col, sdtype='id')
        metadata.set_sequence_key(column_name=tuple(unique_id_cols))

        # Set sequence/time index
        metadata.set_sequence_index(column_name=time_col)

        return metadata

    @staticmethod
    def run(config, X: pd.DataFrame):
        """
        Execute the complete preprocessing pipeline for the PARSynthesizer.

        :param config: Configuration dictionary.
        :type config: dict
        :param X: Input dataset.
        :type X: pd.DataFrame

        :return: Tuple of processed DataFrame and its associated metadata.
        :rtype: (pd.DataFrame, SingleTableMetadata)
        """
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        
        # get starting sequences with finite length
        sequences = PARSynthesizerPreprocessor.get_start_sequences(config, X)

        # Convert numerical static colums to dtype category
        sequences = PARSynthesizerPreprocessor.numerical_static_column_handling(config, sequences)

        # get meta data
        unique_id_cols = config["dataset"]["feature_set"]["unique_id_cols"]
        time_col = config["dataset"]["feature_set"]["time_col"]
        metadata = get_metadata(sequences, unique_id_cols, time_col)

        return sequences, metadata


class TabFormerGPT2Preprocessor:

    def __init__(
        self,
        config,
        X_train: pd.DataFrame,
        X_synth_start: pd.DataFrame,
        checkpoint_dir: str,
        X_val: pd.DataFrame = None,
        use_secondary_id_groups = True,
    ):
        self.config = config
        self.X_train = X_train
        self.X_synth_start = X_synth_start
        self.X_val = X_val
        self.use_secondary_id_groups = use_secondary_id_groups
        self.checkpoint_dir = checkpoint_dir

        if isinstance(config, DictConfig):
            self.config = OmegaConf.to_container(config, resolve=True)
    
    def transform(self):
        # Step 1: Data discretization
        if self.X_val is not None:
            self.X_train, self.X_synth_start, self.X_val = self.get_binned_data()
        else:
            self.X_train, self.X_synth_start = self.get_binned_data()

        # Step 2: Assign unique ids
        if self.use_secondary_id_groups:
            # Create multi-level sequences by clusters if required
            self.X_synth_start = self.get_secondary_id_groups()
        else:
            # Assign single unique_id column
            primary_id_col = self.config["dataset"]["sequence_schema"]["primary_id"]
            self.X_synth_start = add_unique_id_column(self.X_synth_start, primary_id_col)

        # Step 3: Get and initialize vocabulary
        self.vocab = self.get_vocabulary()

        # Step 4: Tokenization 
        self.dataset = self.get_tokenized_dataset(vocab=self.vocab)

        logger.info(f"SUCCESS: Successfully finished preprocessing for TabFormerGPT2 model")

        return self.dataset, self.vocab

    
    def get_binned_data(self):
        """
        Return preprocessed data that where the continous_cols are being binned and the
        count variables are being rounded and transformed to integers if required.
        """

        continous_cols = self.config["dataset"]["feature_set"]['continous_cols']
        count_variables = self.config["dataset"]["feature_set"]['count_variables']
        n_bins = self.config["preprocessing"]["sequential"]["n_bins"]


        # Create and fit the pipeline
        binning_pipeline = Pipeline([
            ('binner', ContinuousBinner(nbins=n_bins, columns=continous_cols)),
            ('count_vars', CountVariableTransformer(columns=count_variables)),
        ])

        X_train = binning_pipeline.fit_transform(self.X_train)
        X_synth_start = binning_pipeline.transform(self.X_synth_start)

        if self.X_val is not None:
            X_val = binning_pipeline.transform(self.X_val)
            return X_train, X_synth_start, X_val
        
        return X_train, X_synth_start

    def get_secondary_id_groups(self):
        """
        """

        primary_id_col = self.config["dataset"]["sequence_schema"]["primary_id"]
        secondary_id = self.config["dataset"]["sequence_schema"]["secondary_id"]
        secondary_id_options = self.config["dataset"]["sequence_schema"]["secondary_id_options"]
        elements_per_primary_id = self.config["dataset"]["sequence_schema"]["elements_per_primary_id"]
        quasi_identifier_cols = self.config["dataset"]["privacy_audit"]["quasi_identifier_cols"]

        X_synth_start = assign_ids_by_grouping(
            self.X_synth_start,
            QID=quasi_identifier_cols,
            primary_id_col=primary_id_col,
            secondary_id_col=secondary_id,
            secondary_ids=secondary_id_options,
            seq_len_per_primary_id=elements_per_primary_id
        )

        return X_synth_start
    
    def get_vocabulary(self):
        """
        """
        # Load parameters from config
        vocab_config = self.config["vocabulary"]

        # List all available real data
        if self.X_val is not None:
            X_real = [self.X_train, self.X_val]
        else:
            X_real = self.X_train

        # Combine synthetic start sequences and available real data set
        vocab_basis_data = VocabularyBasisData(
            X_real=X_real,
            X_synth=self.X_synth_start,
            static_columns=self.config["dataset"]["feature_set"]['static_columns'],
            event_columns=self.config["dataset"]["feature_set"]['event_columns'],
            count_variables=self.config["dataset"]["feature_set"]['count_variables']
        )

        # Initialize Vocabulary
        vocab = Vocabulary(
            vocab_dir=self.checkpoint_dir ,
            unk_token=vocab_config["unk_token"],
            sep_token=vocab_config["sep_token"],
            pad_token=vocab_config["pad_token"],
            bos_token=vocab_config["bos_token"],
            eos_token=vocab_config["eos_token"],
            missing_token=vocab_config["missing_token"],
            special_field_tag=vocab_config["special_field_tag"],
        )
        vocab.initialize_vocabulary(vocab_basis_data)

        return vocab
    
    def get_tokenized_dataset(self, vocab: Vocabulary):
        """
        """
        static_columns = self.config["dataset"]["feature_set"]['static_columns']
        event_columns = self.config["dataset"]["feature_set"]['event_columns']
        unique_id_cols = self.config["dataset"]["feature_set"]['unique_id_cols']
        max_event_count = self.config["preprocessing"]["sequential"]['max_event_count']
        min_seq_len = self.config["preprocessing"]["sequential"]['min_seq_len']
        sequencing_strategy = self.config["preprocessing"]["sequential"]['sequencing_strategy']
        N_samples = self.config["preprocessing"]["sequential"]['N_samples']
        val_size = self.config["TabFormerGPT2"]["val_size"]

        dataset = ClinicalTrialDataset(
            data=self.X_train,
            unique_id=unique_id_cols,
            static_columns=static_columns,
            event_columns=event_columns,
            max_event_count=max_event_count,
            min_seq_len=min_seq_len,
            vocab_dir=self.checkpoint_dir,
            val_size=val_size,
            vocab=vocab,
            sequencing_strategy=sequencing_strategy,
            N_samples=N_samples
        )
        return dataset