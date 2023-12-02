import pandas as pd
from sdv.metadata import SingleTableMetadata

def get_metadata(X_train, unique_id_cols, time_col):
    # Define Metadata for sequential dataset
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(X_train)
    for col in unique_id_cols:
        metadata.update_column(column_name=col, sdtype='id')
    metadata.set_sequence_key(column_name=tuple(unique_id_cols))
    metadata.set_sequence_index(column_name=time_col)
    return metadata

def ensure_same_size(X_real: pd.DataFrame, X_synth: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Ensures that two given DataFrames are of the same size by cropping the larger DataFrame 
    down to the size of the smaller one.

    Returns:
    - A tuple containing the possibly cropped DataFrames in the same order as the input.
    """
    N_min = min(len(X_real), len(X_synth))
    
    return X_real.iloc[0:N_min], X_synth.iloc[0:N_min]