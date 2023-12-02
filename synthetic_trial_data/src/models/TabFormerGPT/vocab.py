import os
import logging
from typing import List, Union
from collections import OrderedDict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class VocabularyBasisData:
    """
    Instances of the VocabularyBasisData represent a dataset that combines the
    synthetic start sequences with the real data, that is used to generate a common
    vocabulary across both datasets.
    
    :param X_real: The actual dataset or list of datasets.
    :type X_real: Union[pd.DataFrame, List[pd.DataFrame]]
    :param X_synth: The synthetic start sequences.
    :type X_synth: pd.DataFrame
    :param static_columns: Columns in the data that remain unchanged.
    :type static_columns: List[str]
    :param event_columns: Columns in the data representing events.
    :type event_columns: List[str]
    :ivar data: A combined DataFrame of X_real and X_synth.
    :vartype data: pd.DataFrame
    """

    def __init__(
        self,
        X_real: Union[pd.DataFrame, List[pd.DataFrame]],
        X_synth: pd.DataFrame,
        static_columns: List[str], 
        event_columns: List[str],
        count_variables: List[str] = None,
    ):
        self.X_real = X_real
        self.X_synth = X_synth
        self.static_columns = static_columns
        self.event_columns = event_columns
        self.count_variables = count_variables

        # Combine synthetic start sequences and available real data set
        if isinstance(X_real, pd.DataFrame):
            self.data = pd.concat([X_real, X_synth], ignore_index=True)
        else:  # X_real is a list of dataframes
            self.data = pd.concat(X_real + [X_synth], ignore_index=True)


class Vocabulary:
    """
    Vocabulary class for tabular data tokenization.

    :param vocab_dir: Directory for the vocabulary.
    :type vocab_dir: str
    :param unk_token: Token for unknown values.
    :type unk_token: str
    :param sep_token: Token for separating sequences.
    :type sep_token: str
    :param pad_token: Token for padding.
    :type pad_token: str
    :param cls_token: Token for the start of a sequence.
    :type cls_token: str
    :param mask_token: Token for masking.
    :type mask_token: str
    :param bos_token: Token for the beginning of a sequence.
    :type bos_token: str
    :param eos_token: Token for the end of a sequence.
    :type eos_token: str
    :param missing_token: Token for missing values.
    :type missing_token: str
    :param special_field_tag: Tag for special fields.
    :type special_field_tag: str
    :ivar token2id: Dictionary mapping tokens to their global and local IDs.
    :vartype token2id: dict
    :ivar id2token: Dictionary mapping IDs to their tokens and fields.
    :vartype id2token: dict
    :ivar field_keys: Dictionary holding field keys.
    :vartype field_keys: dict
    """

    def __init__(
        self,
        vocab_dir,
        unk_token = "[UNK]",
        sep_token = "[SEP]",
        pad_token = "[PAD]",
        cls_token = "[CLS]",
        mask_token = "[MASK]",
        bos_token = "[BOS]",
        eos_token = "[EOS]",
        missing_token = "[MISSING]",
        special_field_tag = "SPECIAL"
    ):

        #self.adap_sm_cols = set()

        self.special_field_tag = special_field_tag

        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.missing_token = missing_token

        self.special_tokens = [
            self.unk_token, 
            self.sep_token,
            self.pad_token,
            self.cls_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
            self.missing_token
        ]

        self.token2id = OrderedDict()  # {field: {token: id}, ...}
        self.id2token = OrderedDict()  # {id : [token,field]}
        self.field_keys = OrderedDict()
        self.token2id[self.special_field_tag] = OrderedDict()

        # Vocabulary
        self.vocab_dir = vocab_dir
        self.vocab_filename = 'vocab.nb' # this field is set in the `save_vocab` method

        for token in self.special_tokens:
            global_id = len(self.id2token)
            local_id = len(self.token2id[self.special_field_tag])

            self.token2id[self.special_field_tag][token] = [global_id, local_id]
            self.id2token[global_id] = [token, self.special_field_tag, local_id]
        
    def initialize_vocabulary(self, dataset: VocabularyBasisData):
        """
        Construction and initialization of the vocabulary of a given dataset
        Initializes the vocabulary for the dataset.

        The method performs the following operations:
        1. Sets Field Keys: Treats the columns as "field keys" for the vocabulary.
        2. Populates the Vocabulary: For each column, identifies unique values sorted by frequency 
           and adds them to the vocabulary.
        3. Logs Information: Reports columns used, total vocabulary size, and per-column vocabulary size.
    
        This ensures each unique value in the dataset columns gets a unique ID, enabling the 
        conversion of raw tabular data into sequences of token IDs for model ingestion.
        """
        # Consider only columns that are in either event_columns or static_columns
        cols = [
            col for col in dataset.data.columns 
            if col in dataset.event_columns 
            or col in dataset.static_columns
        ]
        self.static_columns = dataset.static_columns
        self.event_columns = dataset.event_columns
        self.set_field_keys(cols)

        for column in cols:
            if column in dataset.count_variables:
                # Get the min and max values for the count variable
                min_val = int(dataset.data[column].min())
                max_val = int(dataset.data[column].max())
                
                # Ensure the vocabulary contains all integers between min and max
                for val in range(min_val, max_val + 1):
                    self.set_id(val, column)
            else:
                unique_values = dataset.data[column].value_counts(sort=True, dropna=False).to_dict()
                for val in unique_values:
                    self.set_id(val, column)

        logger.info(f"columns used for vocab: {list(cols)}")
        logger.info(f"total vocabulary size: {len(self.id2token)}")

        for column in cols:
            vocab_size = len(self.token2id[column])
            logger.info(f"column : {column}, vocab size : {vocab_size}")
        
        # Save vocabulary
        file_name = os.path.join(self.vocab_dir, self.vocab_filename)
        logger.info(f"saving vocab at {file_name}")
        self.save_vocab(file_name)

    def set_id(self, token, field_name, return_local=False):
        """
        Assigns a global and local ID to the token.

        :param token: The token to set an ID for.
        :type token: str
        :param field_name: The field name associated with the token.
        :type field_name: str
        :param return_local: If True, returns the local ID.
        :type return_local: bool
        :return: The global or local ID for the token.
        :rtype: int
        """
        global_id, local_id = None, None

        if isinstance(token, (float, np.float64)) and np.isnan(token):
            token = "nan"
                
        if token not in self.token2id[field_name]:
            global_id = len(self.id2token)
            local_id = len(self.token2id[field_name])

            self.token2id[field_name][token] = [global_id, local_id]
            self.id2token[global_id] = [token, field_name, local_id]
        else:
            global_id, local_id = self.token2id[field_name][token]

        if return_local:
            return local_id

        return global_id

    def get_id(self, token, field_name="", special_token=False, return_local=False):
        """
        Retrieves the ID associated with a token.

        :param token: The token to get the ID for.
        :type token: str
        :param field_name: The field name associated with the token.
        :type field_name: str
        :param special_token: If True, searches within special tokens.
        :type special_token: bool
        :param return_local: If True, returns the local ID.
        :type return_local: bool
        :return: The ID for the token.
        :rtype: int
        """
        global_id, local_id = None, None
        if special_token:
            field_name = self.special_field_tag

        if token in self.token2id[field_name]:
            global_id, local_id = self.token2id[field_name][token]
        else:
            raise Exception(f"token {token} not found in field: {field_name}")

        if return_local:
            return local_id

        return global_id

    def set_field_keys(self, keys):
        """
        Sets field keys for the vocabulary.

        :param keys: The list of field keys.
        :type keys: List[str]
        """
        for key in keys:
            self.token2id[key] = OrderedDict()
            self.field_keys[key] = None

        self.field_keys[self.special_field_tag] = None  # retain the order of columns

    def get_field_ids(self, field_name, return_local=False):
        """
        Retrieves the field keys from the vocabulary.

        :param remove_target: If True, removes target field key.
        :type remove_target: bool
        :param ignore_special: If True, ignores special field keys.
        :type ignore_special: bool
        :return: List of field keys.
        :rtype: List[str]
        """
        if field_name in self.token2id:
            ids = self.token2id[field_name]
        else:
            raise Exception(f"field name {field_name} is invalid.")

        selected_idx = 0
        if return_local:
            selected_idx = 1
        return [ids[idx][selected_idx] for idx in ids]

    def get_from_local_ids(self, field_name, local_ids, what_to_get='global_ids'):
        """
        Retrieves the corresponding global IDs or tokens based on the provided local IDs.
        
        :param field_name: The field name associated with the local IDs.
        :type field_name: str
        :param local_ids: Array-like container of local IDs to retrieve information for.
        :type local_ids: Any array-like data structure (e.g., list, ndarray)
        :param what_to_get: Determines what to retrieve, either 'global_ids' or 'tokens'.
        :type what_to_get: str
        :return: Array-like container with global IDs or tokens corresponding to the provided local IDs.
        :rtype: Same type as input local_ids (e.g., list, ndarray)
        """
        device = local_ids.device

        def map_local_ids_to_global_ids(lid):
            return self.get_field_ids(field_name)[lid] if lid != -100 else -100

        def map_local_ids_to_tokens(lid):
            gid = map_local_ids_to_global_ids(lid)
            return f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'

        if what_to_get == 'global_ids':
            return local_ids.cpu().apply_(map_local_ids_to_global_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_local_ids_to_tokens)
            new_array_for_tokens = local_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Only 'global_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def get_from_global_ids(self, global_ids, what_to_get='local_ids', with_field_name=True):
        """
        Retrieves the corresponding local IDs or tokens based on the provided global IDs.
        
        :param global_ids: Array-like container of global IDs to retrieve information for.
        :type global_ids: Any array-like data structure (e.g., list, ndarray)
        :param what_to_get: Determines what to retrieve, either 'local_ids' or 'tokens'.
        :type what_to_get: str
        :param with_field_name: If True, tokens retrieved will include field name. 
            Only applicable when what_to_get is 'tokens'.
        :type with_field_name: bool
        :return: Array-like container with local IDs or tokens corresponding to the 
            provided global IDs.
        :rtype: Same type as input global_ids (e.g., list, ndarray)
        """
        device = global_ids.device

        def map_global_ids_to_local_ids(gid):
            return self.id2token[gid][2] if gid != -100 else -100

        def map_global_ids_to_tokens(gid):
            full_token =  f'{self.id2token[gid][1]}_{self.id2token[gid][0]}' if gid != -100 else '-'
            if with_field_name:
                return full_token
            else:
                return self.id2token[gid][0]

        if what_to_get == 'local_ids':
            return global_ids.cpu().apply_(map_global_ids_to_local_ids).to(device)
        elif what_to_get == 'tokens':
            vectorized_token_map = np.vectorize(map_global_ids_to_tokens)
            new_array_for_tokens = global_ids.detach().clone().cpu().numpy()
            return vectorized_token_map(new_array_for_tokens)
        else:
            raise ValueError("Only 'local_ids' or 'tokens' can be passed as value of the 'what_to_get' parameter.")

    def save_vocab(self, fname):
        """
        Saves the vocabulary to a file.

        :param fname: The filename to save the vocabulary to.
        :type fname: str
        """
        self.filename = fname
        with open(fname, "w") as fout:
            for idx in self.id2token:
                token, field, _ = self.id2token[idx]
                token = "%s_%s" % (field, token)
                fout.write("%s\n" % token)

    def get_field_keys(self, remove_target=True, ignore_special=False, ignore_static=True):
        keys = list(self.field_keys.keys())

        if ignore_special:
            keys.remove(self.special_field_tag)

        if ignore_static:
            exclude_keys = self.static_columns
            keys = [item for item in keys if item not in exclude_keys]

        return keys

    def get_special_tokens(self):
        special_tokens_map = {}
        keys = ["unk_token", "sep_token", "pad_token", "cls_token", "mask_token", "bos_token", "eos_token"]
        for key, token in zip(keys, self.special_tokens):
            token = "%s_%s" % (self.special_field_tag, token)
            special_tokens_map[key] = token

        return AttrDict(special_tokens_map)

    def __len__(self):
        return len(self.id2token)

    def __str__(self):
        """
        Returns a string representation of the vocabulary.

        :return: String representation of the vocabulary.
        :rtype: str
        """
        str_ = 'vocab: [{} tokens]  [field_keys={}]'.format(len(self), self.field_keys)
        return str_
