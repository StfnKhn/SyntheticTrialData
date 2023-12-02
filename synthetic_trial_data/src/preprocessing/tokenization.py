import os
import logging
import pandas as pd
import numpy as np
import torch
from typing import Union, List
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer

from synthetic_trial_data.src.preprocessing.data_sequencing import create_sequences
from synthetic_trial_data.src.models.TabFormerGPT.vocab import Vocabulary
from synthetic_trial_data.src.models.TabFormerGPT.utils import random_split_dataset


logger = logging.getLogger(__name__)


class Tokenizer(PreTrainedTokenizer):
    def __init__(self, parent):
        self.parent = parent
        self.vocab = parent.vocab
        
        super().__init__(bos_token=parent.bos_token, eos_token=parent.eos_token, unk_token=parent.unk_token)

    def encode(self, sequence: List[str], field_name_mask: List[str]) -> List[int]:
        """
        Convert a single sequence to its corresponding token IDs.
        """
        token_ids = []

        for in_seq_idx, token in enumerate(sequence):
            # Get respective field_name
            field_name = field_name_mask[in_seq_idx]

            # Handle nan edge_case       
            if isinstance(token, (float, np.float64)) and np.isnan(token):
                token = "nan"

            # Get vocab_id from vovabulary instance
            vocab_id = self.vocab.get_id(token=token, field_name=field_name)
            token_ids.append(vocab_id)

        return token_ids

    def decode(self, ids: List[int]) -> List[str]:
        """
        Convert a list of token IDs back to its corresponding sequence.
        """
        sequence = [self.vocab.id2token[id][0] for id in ids]
        return sequence


class BaseDataset(Dataset):
    def __init__(
        self,
        unique_id: Union[str, List[str]],
        static_columns: List[str],
        event_columns: List[str],
        max_event_count: int,
        min_seq_len:int,
        return_labels=False,
        sequencing_strategy: Union["most_recent", "sampled"] = "most_recent",
        N_samples: int = 2
    ):
        self.static_columns = static_columns
        self.event_columns = event_columns
        self.unique_id = unique_id
        self.max_event_count = max_event_count
        self.min_seq_len = min_seq_len
        self.return_labels = return_labels
        self.sequencing_strategy = sequencing_strategy
        self.N_samples = N_samples

        # Define special Tokens
        self.missing_token = "[MISSING]"
        self.unk_token = "[UNK]"
        self.sep_token = "[START_EVENT]"
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        
        self.special_field_tag = "SPECIAL"
        self.special_tokens = [
            self.missing_token,
            self.unk_token,
            self.sep_token,
            self.pad_token,
            self.bos_token,
            self.eos_token 
        ]

    def __getitem__(self, index):
        """
        Get an item from the dataset by its index.
        
        :param index: The index of the desired item.
        :type index: int
        :return: Tokenized data for the given index. If `return_labels` is True, returns data and labels.
        """
        return_data = torch.tensor(self.samples[index], dtype=torch.long)
        
        if self.return_labels:
            target = self.targets[index]
            return_data = return_data, torch.tensor(target, dtype=torch.float32)
    
        return return_data

    def __len__(self):
        """
        Get the length of the dataset.
        
        :return: The number of items in the dataset.
        :rtype: int
        """
        return len(self.samples)

    def _add_bos_eos_tokens(self, sequence):
        """
        Add BOS (Beginning Of Sequence) and EOS (End Of Sequence) tokens to a sequence.
        
        :param sequence: The input sequence to which BOS and EOS tokens are to be added.
        :type sequence: List[str]
        :return: The sequence with added BOS and EOS tokens.
        :rtype: List[str]
        """
        return [self.bos_token] + sequence + [self.eos_token]

    def _data_sequencing(self, data):
        """
        Convert the input DataFrame into sequences.

        Given a pandas DataFrame, this function iterates over the rows and converts each row into 
        a sequence of tokens. Each row in the DataFrame corresponds to a clinical tabulated data entry,
        and is transformed into a list of tokens, effectively "flattening" the structured data into sequences.
        
        :param data: Input data in the form of a pandas DataFrame containing clinical information.
        :type data: pd.DataFrame
        :return: List of sequences where each sequence represents a row from the input DataFrame.
        :rtype: List[List[str]]
        """
        sequences, N_sentences = create_sequences(
            df=data,
            static_columns=self.static_columns,
            event_columns=self.event_columns,
            unique_id=self.unique_id,
            max_event_count=self.max_event_count,
            min_seq_len=self.min_seq_len,
            sequence_format="flat",
            use_separator_token=True,
            separator_token=self.sep_token,
            convert_to_str=False,
            strategy=self.sequencing_strategy,
            N_samples=self.N_samples
        )

        return sequences

    def _prepare_samples(self):
        """
        Prepare the samples by converting each sequence in the dataset to token IDs.

        :param sequences: List of sequences to be processed.
        :type sequences: List[List[str]]
        """
        self.samples = [self._convert_sequence_to_ids(idx, seq) for idx, seq in enumerate(self.sequences)]

    def _convert_sequence_to_ids(self, seq_idx: int, sequence: List[str]) -> List[int]:
        """
        Convert a single sequence to its corresponding token IDs.

        :param seq_idx: Index of sequence in the list of sequences
        :type idx: int
        :param sequence: List of tokens to be processed.
        :type sequence: List[str]
        :return: Token IDs for the sequence.
        :rtype: List[int]
        """
        token_ids = []

        for in_seq_idx, token in enumerate(sequence):
            # Get respective field_name
            field_name = self.field_name_masks[seq_idx][in_seq_idx]

            # Handle nan edge_case       
            if isinstance(token, (float, np.float64)) and np.isnan(token):
                token = "nan"

            # Get vocab_id from vovabulary instance
            vocab_id = self.vocab.get_id(token=token, field_name=field_name)
            token_ids.append(vocab_id)

        return token_ids

    def _generate_field_name_mask(self, seq):
        """
        Generate a field name mask based on the provided sequence.

        The function follows these rules:
        1. The sequence always starts with a BOS token.
        2. Then, the tokens for static columns appear.
        3. After the static columns, events are separated by a `sep_token` (e.g., "[START_EVENT]").
        4. Between each `sep_token`, the event tokens always appear in the same order.

        :param seq: The input sequence for which the mask has to be generated.
        :type seq: list of str
        :return: A mask list with field names corresponding to each token in the input sequence.
        :rtype: list of str

        Example:
        .. code-block:: python
            tokenizer = ClinicalTabTokenizer(
                static_columns = ["Sex", "Gender"],
                event_columns = ["Type", "Size"]
            )
            seq = ["[BOS]", "Female", "White", "[START_EVENT]", "A", "38", "[START_EVENT]", "A", "37", "[PAD]"]
            mask = tokenizer.generate_field_name_mask(seq)
            print(mask)
            # Expected output: ['SPECIAL', 'Sex', 'Gender', 'SPECIAL', 'Type', 'Size', 'SPECIAL', 'Type', 'Size', 'SPECIAL']

        """
        mask = []
        static_idx = 0
        event_idx = 0
        in_event = False
        
        for token in seq:
            if token in self.special_tokens:
                mask.append(self.special_field_tag)
                if token == self.sep_token:
                    in_event = True
                    event_idx = 0
                continue
            
            if in_event:
                if event_idx < len(self.event_columns):
                    mask.append(self.event_columns[event_idx])
                    event_idx += 1
                else:
                    # Reset for a new event after the last event column token
                    in_event = False
                    event_idx = 0
                    mask.append(self.static_columns[static_idx])
                    static_idx += 1
            else:
                if static_idx < len(self.static_columns):
                    mask.append(self.static_columns[static_idx])
                    static_idx += 1
                else:
                    # If you reach here, there might be an issue with the sequence or the static columns
                    mask.append("UNKNOWN")

        return mask


class ClinicalTrialDataset(BaseDataset):
    """
    A class that represents clinical tabular datasets that could be used. Initializing this 
    ClinicalTrialDataset performs thw whole tokenization process that is required to fed
    the data into the TabFormerGPT2 model.

    :param data: The input DataFrame with clinical data.
    :type data: pd.DataFrame
    :param unique_id: Unique identifier for each record. It can also be a list of strings
    :type unique_id: str or List[str]
    :param static_columns: Static columns in the input data.
    :type static_columns: List[str]
    :param event_columns: Event columns in the input data.
    :type event_columns: List[str]
    :param max_event_count: Maximum count of events to consider.
    :type max_event_count: int
    :param min_seq_len: Minimum sequence length.
    :type min_seq_len: int
    :param vocab_dir: Directory to store the vocabulary.
    :type vocab_dir: str
    :param test_size: Fraction of data to be used for testing.
    :type test_size: float
    :param val_size: Fraction of data to be used for validation.
    :type val_size: float
    :param sequencing_strategy: If "most_recent" the most recent events are being considered,
        if sequence length is greater than the max_event_count. When "sampled" then the
        sequences are being sampled from the number of sequences available for this sequence
        until the max_event_count is reached. First and last samples are always used. The order
        is being preserved in the sampling process. Note that this strategy increases
        the number of training points. No duplicates are being added.
    :type sequencing_strategy: Union["most_recent", "sampled"]
    :param N_samples: number of samples per long sequence patient.
    :type N_samples: int
    :param return_labels: Flag to determine if labels should be returned. Default is False.
    :type return_labels: bool
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unique_id: Union[str, List[str]],
        static_columns: List[str],
        event_columns: List[str],
        max_event_count: int,
        min_seq_len:int,
        vocab_dir:str,
        val_size: float,
        test_size: float = None,
        vocab=None,
        return_labels=False,
        sequencing_strategy: Union["most_recent", "sampled"] = "most_recent",
        N_samples: int = 2
    ):
        super().__init__(
            unique_id,
            static_columns,
            event_columns,
            max_event_count,
            min_seq_len,
            return_labels,
            sequencing_strategy,
            N_samples
        )

        # Set test_size and val_size for data split
        self.test_size = test_size
        self.val_size = val_size

        # Generate vocabulary instance and unify the special tokens
        self.vocab_dir = vocab_dir
        if not vocab:
            self.vocab = Vocabulary(
                vocab_dir=self.vocab_dir,
                unk_token=self.unk_token,
                sep_token=self.sep_token,
                pad_token=self.pad_token,
                bos_token=self.bos_token,
                eos_token=self.eos_token,
                missing_token=self.missing_token,
                special_field_tag = self.special_field_tag,
            )
            # Initialize vocabularys
            self.vocab._initialize_vocabulary(data)
        else:
            self.vocab = vocab

        # Set up a tokenizer for this class
        self.tokenizer = Tokenizer(self)

        # Run the tokenization and the creation of the vocabulary
        self._run(data)

    def _run(self, data):
        """
        Main processing function that orchestrates the tokenization process for the input data.

        This function performs the following operations in order:
        1. Converts the input data into sequences.
        2. Adds BOS and EOS tokens to each sequence.
        3. Includes padding tokens to unify sequence lengths.
        4. Maps field names to tokens for each sequence.
        5. Converts sequences into token IDs.
        6. Splits the dataset into train, test, and optional validation subsets.

        :param data: Input data in the form of a pandas DataFrame.
        :type data: pd.DataFrame
        """

        # Generate sequences of data of the from
        # [static_col_1, static_col_2, ..., "[START_EVENT]", "event_A_1",  "event_B_1", "[START_EVENT]", ...] 
        self.sequences = self._data_sequencing(data)

        # Add BOS and EOS token
        for idx, seq in enumerate(self.sequences):
            self.sequences[idx] = self._add_bos_eos_tokens(seq)

        # Include PAD tokens to unify the sequence lengths
        self.attention_masks = []
        max_seq_length = max([len(seq) for seq in self.sequences])
        for idx, seq in enumerate(self.sequences):
            self.sequences[idx], attention_mask = ClinicalTrialDataset.include_padding_tokens(
                sequence=seq,
                max_seq_length=max_seq_length,
                pad_token=self.pad_token
            )
            self.attention_masks.append(attention_mask)

        # Generate mask that maps field_names to tokens by sequence index
        self.field_name_masks = []
        for idx, seq in enumerate(self.sequences): 
            self.field_name_masks.append(self._generate_field_name_mask(seq))

        # converting each sequence in the dataset to token IDs.        
        self._prepare_samples()

        # Perform train_test(_val) split
        self._train_test_split(self.test_size, self.val_size)

    def _train_test_split(self, test_size: float, val_size: float = None):
        """
        Splits the current dataset into train, validation, and test subsets based
        on the specified proportions.

        :param test_size: Proportion of the dataset to be used as the test set.
        :type test_size: float
        :param val_size: Proportion of the dataset to be used as the validation set. 
            If not specified, the remaining data (after allocating to the test set) 
            is used entirely as the training set.
        :type val_size: Optional[float]
        :return: Split datasets (train, validation, test).
        """
        totalN = len(self)

        if test_size is not None:
            testN = int(test_size * totalN)
        else:
            testN = 0

        if val_size is not None:
            valN = int(val_size * totalN)
        else:
            valN = 0

        trainN = totalN - valN - testN

        assert totalN == trainN + valN + testN, "Mismatch in dataset split counts."

        logger.info(f"Data split into: train [{trainN}]  valid [{valN}]  test [{testN}]")

        if val_size == None:
            lengths = [trainN, testN]
            self.train_dataset, self.test_dataset = random_split_dataset(self, lengths)
        else:
            lengths = [trainN, valN, testN]
            self.train_dataset, self.val_dataset, self.test_dataset = random_split_dataset(self, lengths)
            print(self.train_dataset)
        
    @staticmethod
    def include_padding_tokens(sequence, max_seq_length, pad_token):
        """
        Add padding tokens to a sequence until it reaches the specified maximum sequence length.
        
        :param sequence: The input sequence to be padded.
        :type sequence: list
        :param max_seq_length: The desired maximum length for the sequence after padding.
        :type max_seq_length: int
        :param pad_token: The token used to pad the sequence.
        :type pad_token: str
        :return: The padded sequence.
        :rtype: list
    
        Example:
        .. code-block:: python
    
            sequence = [1, 2, 3]
            max_seq_length = 5
            pad_token = 0
            padded_sequence = include_padding_tokens(sequence, max_seq_length, pad_token)
            print(padded_sequence)
            # Expected output: [1, 2, 3, 0, 0]
    
        """
        while len(sequence) < max_seq_length:
            sequence.append(pad_token)
    
        attention_mask = [0 if token == pad_token else 1 for token in sequence]
        
        return sequence, attention_mask


class StartSequences(BaseDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        unique_id: Union[str, List[str]],
        static_columns: List[str],
        event_columns: List[str],
        vocab: Vocabulary,
        max_event_count: int,
        min_seq_len:int,
        return_labels=False,
        sequencing_strategy: Union["most_recent", "sampled"] = "most_recent",
        N_samples: int = 2
    ):
        super().__init__(
            unique_id,
            static_columns,
            event_columns,
            max_event_count,
            min_seq_len,
            return_labels,
            sequencing_strategy,
            N_samples
        )

        self.data = data
        self.vocab = vocab

        # Generate sequences of tabular sequential data set
        self.sequences = self._data_sequencing(data)

        # Add BOS and EOS token
        for idx, seq in enumerate(self.sequences):
            self.sequences[idx] = [self.bos_token] + seq

        # Generate mask that maps field_names to tokens by sequence index
        self.field_name_masks = []
        for idx, seq in enumerate(self.sequences): 
            self.field_name_masks.append(self._generate_field_name_mask(seq))

        # converting each sequence in the dataset to token IDs.        
        self._prepare_samples()
