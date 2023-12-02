from abc import ABC, abstractmethod
import logging
import random
import pandas as pd
from typing import List, Union
from itertools import groupby


logger = logging.getLogger(__name__)


class BaseSequencer(ABC):

    def __init__(
        self,
        static_columns: List[str],
        event_columns: List[str],
        unique_id_cols: List[str],
        datetime_col: str,
    ):
        self.static_columns = static_columns
        self.event_columns = event_columns
        self.unique_id_cols = unique_id_cols
        self.datetime_col = datetime_col

    @abstractmethod
    def create_sequences(self):
        raise NotImplementedError


class StartSequencer(BaseSequencer):

    def __init__(
        self,
        static_columns: List[str],
        event_columns: List[str],
        unique_id_cols: List[str],
        datetime_col: str,
        first_event_key_col: str,
        first_event_key_value,
        only_keep_key_value:  bool = False
    ):
        super().__init__(
            static_columns=static_columns,
            event_columns=event_columns,
            unique_id_cols=unique_id_cols,
            datetime_col =datetime_col 
        )
        self.columns = static_columns + event_columns
        self.first_event_key_col = first_event_key_col
        self.first_event_key_value = first_event_key_value
        self.only_keep_key_value = only_keep_key_value

    def _select_start_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group the DataFrame by `unique_id_cols` and select rows based on a specific priority.
    
        This function is designed to filter data in a DataFrame according to a hierarchy of selection 
        criteria. Initially, it tries to select rows based on the presence of a key value in a specific column. 
        If this is not available, it selects the earliest available date. If neither is available, it will simply 
        select the first row from the group.
    
        :param df: The input DataFrame to be processed.
        :type df: pd.DataFrame
        :return: A DataFrame after applying the described selection criteria.
        :rtype: pd.DataFrame    
        """
        # Ensure date_col is in datetime format
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors='coerce')

        if self.only_keep_key_value:
            df_with_key_val = df[df[self.first_event_key_col] == self.first_event_key_value]
            return df_with_key_val.groupby(self.unique_id_cols, group_keys=False).first().reset_index()
    
        def select_row(group):
            # Check for 'Screening' in visit column
            row_with_key_element = group[group[self.first_event_key_col] == self.first_event_key_value]
            if not row_with_key_element.empty:
                selected_row = row_with_key_element
            # If no 'Screening' value, select the row with the earliest date
            elif group[self.datetime_col].notna().any():
                selected_row = group[group[self.datetime_col] == group[self.datetime_col].min()]
            # If no date, select the first row
            else:
                selected_row = group.iloc[[0]]
    
            if selected_row.shape[0] > 1:
                return selected_row.iloc[[0]]
            else:
                return selected_row
    
        return df.groupby(self.unique_id_cols, group_keys=False).apply(select_row).reset_index()
    
    def create_sequences(self, X: pd.DataFrame, drop_id_columns=True):
        # Select start sequences from real dataset
        self.X = self._select_start_sequences(df=X)

        # Only select given columns
        if drop_id_columns:
            self.X = self.X[self.columns]
            logger.info(f"Select only the following columns for the starting sequences: {self.X.columns}")
        else:
            self.X = self.X[self.unique_id_cols + self.columns]

        return self.X
    

class PARSynthesizerInputSequencer(BaseSequencer):

    def __init__(
        self,
        static_columns: List[str],
        event_columns: List[str],
        unique_id_cols: List[str],
        datetime_col: str,
        max_event_count: int,
        min_seq_len: int,

    ):
        super().__init__(
            static_columns=static_columns,
            event_columns=event_columns,
            unique_id_cols=unique_id_cols,
            datetime_col =datetime_col 
        )
        self.max_event_count = max_event_count
        self.min_seq_len = min_seq_len

    def create_sequences(self, X: pd.DataFrame):
        X = X.copy()
        self.X = create_vertical_sequences(
            df=X,
            static_columns=self.static_columns,
            event_columns=self.event_columns,
            time_col=self.datetime_col,
            unique_id_cols=self.unique_id_cols,
            max_event_count=self.max_event_count,
            min_seq_len=self.min_seq_len # The PARSynthesizer does not work for sequences with length 1
        )

        return self.X


def create_sequences(
    df: pd.DataFrame,
    static_columns: List[str],
    event_columns: List[str],
    unique_id: str,
    max_event_count: int,
    min_seq_len: int = 1,
    sequence_format: str = 'flat',
    convert_to_str: bool = True,
    use_separator_token: bool = False,
    separator_token: str = None,
    strategy: Union["most_recent", "sampled"] = "most_recent",
    N_samples: int  = 1
):
    """
    Create sequences from a dataframe containing multiple patient-specific 
    tabular time series.
    
    The function generates sequences for each patient based on their 
    static information and event data. The static information is taken 
    from the first visit of each patient, assuming it remains unchanged
    over time. The event data is appended for each visit chronologically.

    If the number of sentences exceeds the defined maximum length, 
    the function will retain the most recent events up to that maximum 
    length while preserving the static information at the beginning of 
    the sequence.

    :param df: Input dataframe where each row pertains to one 
        patient's visit at a particular time. The dataframe should be sorted
        by patient ID and then by visit time.
    :type df: pd.DataFrame
    :param static_columns: List of column names in the dataframe that 
        contain static information for patients.
    :type static_columns: list
    :param list event_columns: List of column names in the dataframe that 
        represent clinical events which might change over time.
    :type event_columns: list
    :param unique_id: column name of the unique patient ID
    :type unique_id: str
    :param max_event_count: Maximum number of records that should be considered in
         the sequenced data set in order to reduce the variance in sequence length.
    :type max_event_count: int
    :param min_seq_len: All patients with less or equal than min_seq_len
        number of sentences/records are being dropped
    :type min_seq_len: int
    :param sequence_format: Defines the format in which the event data 
        should be appended. If 'flat', the events will be added as 
        flat sequence. If 'tuple', the events will be added as tuples.
    :type sequence_format: str
    :param convert_to_str: If True, all elements of the sequence and sub sequences
        are being converted to strings, defaults True
    :type convert_to_str: bool
    :param N_samples: number of samples per long sequence patient
    :type N_samples: int

    :return: List of sequences for each patient. Each sequence is a list
        of static information followed by event data for each visit.
    :rtype: list
    """
    
    sequences = []
    N_sentences = []

    # Ensure valid sequence format
    if sequence_format not in ['flat', 'tuple']:
        raise ValueError("Invalid sequence_format. Choose between 'flat' and 'tuple'.")
    
    # Group data by patient
    grouped = df.groupby(unique_id)

    for _, group in grouped:
        sequence = []

        # Compute number of records/sentences per patient
        N = group[event_columns].shape[0]

        # Take static information from the first visit (assuming it doesn't change)
        static_data = group[static_columns].iloc[0].tolist()
        sequence.extend(static_data)

        # Add separator token between static data and event data if enabled
        if use_separator_token:
            sequence.append(separator_token)

        if N > max_event_count:
            # Add the first record as base line
            base_line_event = group[event_columns].iloc[0].tolist()
            sequence.extend(base_line_event)

            # Add separator between events if the format is 'flat' and separator is enabled
            if sequence_format == 'flat' and use_separator_token:
                sequence.append(separator_token)

            if strategy == "most_recent":
                # Add the most recent records
                most_recent_records = group[event_columns].iloc[-max_event_count-1:].values

                # Append most recent event data
                sequence = append_event_data_to_sequence(
                    sequence, 
                    most_recent_records, 
                    sequence_format, 
                    use_separator_token, 
                    separator_token
                )

                # Convert elements to strings if required
                if convert_to_str:
                    sequence = convert_to_strings(sequence)

                if N > min_seq_len:
                    sequences.append(sequence)
                    N_sentences.append(max_event_count)

            if strategy == "sampled":
                sampled_indices_list = []
                for sample_count in range(0, N_samples):
                    current_sequence = sequence.copy()
                    event_data = []
                    # Sample and sort indexes for events except the first and last
                    remaining_indices = list(range(1, N - 1))
                    sampled_indices = sorted(random.sample(remaining_indices, max_event_count - 2))
                    
                    for index in sampled_indices:
                        event_data.append(group[event_columns].iloc[index].values)#.tolist())

                    # Add end of sequence
                    end_of_sequence = group[event_columns].iloc[-1:].values[0]#.tolist()
                    event_data.append(end_of_sequence)
                
                    # Append most recent event data
                    sampled_sequence = append_event_data_to_sequence(
                        current_sequence, 
                        event_data, 
                        sequence_format, 
                        use_separator_token, 
                        separator_token
                    )

                    # Convert elements to strings if required
                    if convert_to_str:
                        sampled_sequence = convert_to_strings(sampled_sequence)

                    # Only add sampled_indices if not yet added to avoid duplicates
                    if N > min_seq_len and sampled_indices not in sampled_indices_list:
                        sequences.append(sampled_sequence)
                        N_sentences.append(max_event_count)
                        sampled_indices_list.append(sampled_indices)
                        
        else:
            # Take event data from all visits
            event_data = group[event_columns].values
            
            # Append most recent event data
            sequence = append_event_data_to_sequence(
                sequence, 
                event_data, 
                sequence_format, 
                use_separator_token, 
                separator_token
            )

            # Convert elements to strings if required
            if convert_to_str:
                sequence = convert_to_strings(sequence)

            if N > min_seq_len:
                sequences.append(sequence)
                N_sentences.append(N)
        
    return sequences, N_sentences


def convert_sequences_to_tabular(
    sequences: List[List[str]], 
    static_columns: List[str], 
    event_columns: List[str], 
    sep_token: str = "[START_EVENT]",
    unique_id_col: str = None
) -> pd.DataFrame:
    """
    Convert a list of sequences into a pandas DataFrame in a tabular format.

    Each sequence is transformed into a DataFrame where the static columns are repeated 
    for every event in the sequence, and the event columns follow the static columns in each row. 
    The events within sequences are separated by a specified separator token.

    :param sequences: The list of sequences, where each sequence contains individual tokens.
    :type sequences: List[List[str]]
    :param static_columns: The list of static column names.
    :type static_columns: List[str]
    :param event_columns: The list of event column names.
    :type event_columns: List[str]
    :param sep_token: The token used to separate events in a sequence. Default is "[START_EVENT]".
    :type sep_token: str
    :param unique_id_col: Name of a unique_id column that is being added to the generated samples
    :type unique_id_col: str

    :return: A DataFrame in tabular format, representing the sequences.
    :rtype: pd.DataFrame

    Example:
    .. code-block:: python

        sequences = [
            ["[BOS]", "patientA", "conditionX", "[START_EVENT]", "event_A_1", "event_B_1", "[START_EVENT]", "event_A_2", "event_B_2"],
            ["[BOS]", "patientB", "conditionY", "[START_EVENT]", "event_A_1", "event_B_1", "[START_EVENT]", "event_A_2", "event_B_2"]
        ]
        static_columns = ["Patient", "Condition"]
        event_columns = ["Event_A", "Event_B"]
        df = sequences_to_dataframe(sequences, static_columns, event_columns)
        print(df)

        # The output DataFrame will have columns: ["Patient", "Condition", "Event_A", "Event_B"]
        # and rows corresponding to the events for each patient.

    """
    
    data = []
    unique_ids= []  # list to store unique ids for sequences if needed
    unique_id = 0
    for sequence in sequences:
        # Remove the starting '[BOS]' token
        sequence = sequence[1:]

        # Extract static tokens from the beginning of the sequence and convert them to strings
        static_tokens = [str(token) for token in sequence[:len(static_columns)]]
        sequence = sequence[len(static_columns):]

        # Iterate over the sequence and split by separator token
        event_lists = [
            list(map(str, group)) 
            for key, group in groupby(sequence, lambda token: token != sep_token) 
            if key
        ]

        for event_tokens in event_lists:
            row_data = static_tokens + event_tokens
            data.append(row_data)
            if unique_id_col != None:
                unique_ids.append(unique_id)
        
        # increase the counter after each sequence
        if unique_id_col != None:
            unique_id += 1

    # Create DataFrame
    df = pd.DataFrame(data, columns=static_columns + event_columns)

    # If unique ids are to be added, add them to the DataFrame
    if unique_id_col != None:
        df[unique_id_col] = pd.Series(unique_ids).astype(str)
    
    return df


def append_event_data_to_sequence(
    sequence: list,
    event_data: list,
    sequence_format: Union["flat", "tuple"],
    use_separator_token: bool, 
    separator_token: str
):
    """
    Append event data to a given sequence based on the specified format.
    """
    if sequence_format == 'flat':
        for record in event_data:
            sequence.extend(record)
            if use_separator_token:
                sequence.append(separator_token)
        # Remove the last unnecessary separator token
        if use_separator_token:
            sequence.pop()
    elif sequence_format == 'tuple':
        event_data_tuples = [tuple(record) for record in event_data]
        sequence.extend(event_data_tuples)

    return sequence


def convert_to_strings(sequence):
    """
    Convert all items within a list to strings. If an item is a tuple, 
    each element inside the tuple is converted to a string.

    :param sequence: List containing elements or tuples to be converted 
        to strings.
    :type sequence: list
    :return: List where all items are strings or tuples of strings.
    :rtype: list

    Example:
    .. code-block:: python

        data = [123, (45, 67), "test", (89, "mixed")]
        converted_data = convert_to_strings(data)
        print(converted_data)  
        # Expected output: ['123', ('45', '67'), 'test', ('89', 'mixed')]

    """
    for idx, item in enumerate(sequence):
        if isinstance(item, tuple):
            sequence[idx] = tuple(str(sub_item) for sub_item in item)
        else:
            sequence[idx] = str(item)
    return sequence


def create_vertical_sequences(
    df: pd.DataFrame,
    static_columns: List[str],
    event_columns: List[str],
    time_col: str,
    unique_id_cols: List[str],
    max_event_count: int,
    min_seq_len: int = 2,
) -> pd.DataFrame:
    """
    Create vertical sequences from a dataframe containing multiple patient-specific 
    tabular time series.
    
    :param df: Input dataframe.
    :type df: pd.DataFrame
    :param static_columns: List of column names in the dataframe that contain static information for patients.
    :type static_columns: list
    :param event_columns: List of column names in the dataframe that represent clinical events.
    :type event_columns: list
    :param unique_id_cols: column name of the unique patient ID.
    :type unique_id_cols: str
    :param max_event_count: Maximum number of records that should be considered.
    :type max_event_count: int
    :param min_seq_len: Minimum number of records to be included.
    :type min_seq_len: int
    
    :return: A DataFrame with vertical sequences.
    :rtype: pd.DataFrame
    """
    
    all_sequences = []

    # Group data by patient
    grouped = df.groupby(unique_id_cols)

    for _, group in grouped:
        # Take static information from the first visit
        static_data = group[static_columns + unique_id_cols].iloc[0].to_dict()

        N = group.shape[0]
        if N < min_seq_len:
            continue

        if N > max_event_count:
            # Get the beginning of sequence event
            selected_rows = [group.iloc[0]]

            # Determine how many samples are needed in-between
            N_samples_needed = max_event_count - 2

            if N_samples_needed > 0:
                remaining_indices = list(range(1, N - 1))
                sampled_indices = sorted(random.sample(remaining_indices, N_samples_needed))
                selected_rows.extend([group.iloc[idx] for idx in sampled_indices])

            # Add end of sequence event
            selected_rows.append(group.iloc[-1])

        else:
            selected_rows = [row for _, row in group.iterrows()]

        # Convert selected rows to dictionaries and update with static data
        for row in selected_rows:
            seq_data = static_data.copy()
            seq_data.update(row[event_columns + [time_col]].to_dict())
            all_sequences.append(seq_data)

    return pd.DataFrame(all_sequences)
