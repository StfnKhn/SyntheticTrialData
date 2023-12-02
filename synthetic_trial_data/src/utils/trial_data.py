import pandas as pd
import numpy as np


def assign_unique_ids(df, static_columns, unique_id_col, secondary_id, value_order=None):
    """
    Assign unique identifiers to rows based on the uniqueness of attributes within specified columns.

    This function is designed to generate unique IDs based on a combination of static attributes 
    (columns that define a unique entity, e.g., a person) and a variant attribute (column in which 
    the variability determines the ID assignment). Optionally, the function can consider an order 
    of values for subgrouping.

    :param df: The input DataFrame to which unique IDs will be assigned.
    :type df: pd.DataFrame
    :param static_columns: List of columns that define a unique entity, such as a person.
    :type static_columns: list[str]
    :param unique_id_col: The new column name where the unique IDs will be assigned.
    :type unique_id_col: str
    :param secondary_id: Secondary column in which the attribute's variability determines the ID assignment.
    :type secondary_id: str
    :param value_order: (Optional) List of ordered values for the secondary_id for subgrouping.
    :type value_order: list, default is None
    :return: A DataFrame with a new column containing the assigned unique identifiers.
    :rtype: pd.DataFrame

    Example:
    .. code-block:: python

        df = pd.DataFrame({
            'name': ['Alice', 'Alice', 'Bob', 'Bob'],
            'birthdate': ['1990-01-01', '1990-01-01', '1985-05-05', '1985-05-05'],
            'tatlid': ['A1', 'A2', 'B1', 'B2']
        })

        new_df = assign_unique_ids(df, static_columns=['name', 'birthdate'], unique_id_col='unique_id', secondary_id='tatlid')

    Note:
    The function assumes that the `static_columns` provided include the `secondary_id`, and it is removed 
    during processing. The output DataFrame will have a new column specified by `unique_id_col` containing 
    the assigned unique IDs.
    """
    quasi_identifier_cols = static_columns.copy()
    if secondary_id in quasi_identifier_cols:
        quasi_identifier_cols.remove(secondary_id)
    
    def generate_ids(sub_df):
        sub_df.sort_values(by=secondary_id, inplace=True, key=lambda x: x.map({v: i for i, v in enumerate(value_order)}))
        sub_df[unique_id_col] = sub_df.groupby(secondary_id).cumcount().add(1)
        return sub_df

    df = df.groupby(quasi_identifier_cols).apply(generate_ids).reset_index(drop=True)
        
    group_ids = df.groupby(quasi_identifier_cols).ngroup().add(1)
    
    df[unique_id_col] = group_ids.astype(str).str.zfill(3) + '-' + df[unique_id_col].astype(str).str.zfill(3)
    
    return df


def add_unique_id_column(df: pd.DataFrame, unique_id_col: str):
    """
    Adds a column with unique integer identifiers to the input DataFrame.

    :param df: The DataFrame to add the unique identifier column to.
    :type df: pd.DataFrame
    :param unique_id_col: The name of the new column to store the unique identifiers.
    :type unique_id_col: str
    :return: The DataFrame with the added unique identifier column.
    :rtype: pd.DataFrame
    """
    if unique_id_col in df.columns:
        raise ValueError(f"The DataFrame already has a column named '{unique_id_col}'. Please choose a different name for the unique identifier column.")

    df[unique_id_col] = range(1, len(df) + 1)
    return df



def assign_ids_by_grouping(df, QID, primary_id_col, secondary_id_col, secondary_ids, seq_len_per_primary_id):
    """
    Assign primary and secondary identifiers to rows based on the uniqueness of attributes within QID columns.
    
    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param QID: Quasi-identifier columns used for grouping.
    :type QID: list[str]
    :param primary_id_col: The new column name where the primary IDs will be assigned.
    :type primary_id_col: str
    :param secondary_ids: List of secondary IDs.
    :type secondary_ids: list[str]
    :param seq_len_per_primary_id: Dictionary where keys are subgroup sizes and values are their sampling probability.
    :type seq_len_per_primary_id: dict[int, float]
    :return: A DataFrame with new columns containing the assigned primary and secondary IDs.
    :rtype: pd.DataFrame
    """
    sizes, probabilities = zip(*seq_len_per_primary_id.items())
    probabilities = np.array(probabilities) / sum(probabilities)
    
    grouped = df.groupby(QID)

    unique_group_counter = [1]  # Using a mutable type like list to modify inside the function
    
    def assign_group_ids(group):
        n_rows = len(group)
        chosen_sizes = []
        while sum(chosen_sizes) < n_rows:
            sampled_size = np.random.choice(sizes, p=probabilities)
            chosen_sizes.append(sampled_size)
        
        # Adjust if sum of chosen sizes exceeds the group size
        while sum(chosen_sizes) > n_rows:
            chosen_sizes[-1] = n_rows - sum(chosen_sizes) + chosen_sizes[-1]

        primary_ids = []
        secondary_id_list = []
        start_index = 0
        for size in chosen_sizes:
            end_index = start_index + size
            primary_id = f"{str(unique_group_counter[0]).zfill(3)}"
            primary_ids.extend([primary_id] * size)
            unique_group_counter[0] += 1
            
            for idx in range(size):
                secondary_id_list.append(secondary_ids[idx % len(secondary_ids)])
            
            start_index = end_index

        group[primary_id_col] = primary_ids
        group[secondary_id_col] = secondary_id_list
        return group

    result_df = grouped.apply(assign_group_ids).reset_index(drop=True)
    
    return result_df


def get_static_columns(df: pd.DataFrame, unique_id_cols: list) -> list:
    """
    Identify and return static data columns from a DataFrame.

    A column is considered static if, for each unique group (defined by unique_id_cols),
    it has the same value for every element within that group.

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param unique_id_cols: List of columns that define unique groups.
    :type unique_id_cols: list

    :return: List of static column names.
    :rtype: list
    """
    static_columns = []
    grouped = df.groupby(unique_id_cols)
    
    for column in df.columns:
        if column not in unique_id_cols:
            unique_values_per_group = grouped[column].nunique()
            if all(unique_values_per_group == 1):
                static_columns.append(column)

    return static_columns


def get_static_columns_cardinality(df: pd.DataFrame, unique_id_cols: list) -> dict:
    """
    Identify and return the cardinalities of static data columns from a DataFrame.

    A column is considered static if, for each unique group (defined by unique_id_cols),
    it has the same value for every element within that group.

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param unique_id_cols: List of columns that define unique groups.
    :type unique_id_cols: list

    :return: Dictionary of static column names and their cardinalities.
    :rtype: dict
    """
    static_columns_cardinality = {}
    grouped = df.groupby(unique_id_cols)
    
    for column in df.columns:
        if column not in unique_id_cols:
            unique_values_per_group = grouped[column].nunique()
            if all(unique_values_per_group == 1):
                cardinality = df[column].nunique()
                static_columns_cardinality[column] = cardinality

    return static_columns_cardinality


def convert_numeric_static_to_categories(df: pd.DataFrame, unique_id_cols: list) -> pd.DataFrame:
    """
    Convert all numeric static columns in a DataFrame to category type.

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param unique_id_cols: List of columns that define unique groups.
    :type unique_id_cols: list

    :return: DataFrame with numeric static columns converted to category type.
    :rtype: pd.DataFrame
    """
    static_columns = get_static_columns(df, unique_id_cols)
    for column in static_columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].astype('str')
    return df
