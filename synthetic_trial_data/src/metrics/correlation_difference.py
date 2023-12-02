import pandas as pd

def compute_grouped_autocorrelation(df: pd.DataFrame, unique_id_cols: list, event_columns: list) -> pd.DataFrame:
    """
    Compute average autocorrelation for each event column within groups.

    :param df: Input DataFrame with event columns.
    :type df: pd.DataFrame
    :param unique_id_cols: List of columns that define unique groups.
    :type unique_id_cols: list
    :param event_columns: List of event column names that might contain temporal dependencies.
    :type event_columns: list

    :return: DataFrame with average autocorrelation values for each event column.
    :rtype: pd.DataFrame
    """
    # Filter event columns to keep only numeric ones
    event_columns = [col for col in event_columns if pd.api.types.is_numeric_dtype(df[col])]
    
    # Subset the dataframe to the required columns
    df_subset = df[unique_id_cols + event_columns]
    
    # Helper function to compute autocorrelation for a group's events
    def group_autocorr(group):
        return group.apply(lambda col: col.autocorr() if len(col) > 1 else None)  # autocorr requires len > 1

    # Compute autocorrelation for each group
    grouped_autocorr = df_subset.groupby(unique_id_cols).apply(group_autocorr)

    # Compute the mean autocorrelation for each event column
    avg_autocorr = grouped_autocorr.mean()

    return avg_autocorr.to_frame().transpose().rename(index={0: "mean_corr"})

def compute_grouped_crosscorrelation(df: pd.DataFrame, unique_id_cols: list, column_pairs: list) -> pd.DataFrame:
    """
    Compute average cross-correlation for specified column pairs within groups.

    :param df: Input DataFrame with event columns.
    :type df: pd.DataFrame
    :param unique_id_cols: List of columns that define unique groups.
    :type unique_id_cols: list
    :param column_pairs: List of tuple pairs specifying columns to compute cross-correlation.
    :type column_pairs: list of tuples

    :return: DataFrame with average cross-correlation values for each column pair.
    :rtype: pd.DataFrame
    """
    
    # Helper function to compute cross-correlation for a group's column pair
    def group_crosscorr(group, col1, col2):
        if len(group) > 1:
            return group[col1].corr(group[col2])
        return None

    results = {}
    for col1, col2 in column_pairs:
        # Compute cross-correlation for each group
        group_crosscorrs = df.groupby(unique_id_cols).apply(group_crosscorr, col1, col2)
        # Average cross-correlation across groups
        results[(col1, col2)] = group_crosscorrs.mean()

    return pd.DataFrame(results, index=["mean_cross_corr"]).transpose()

def crosscorr_mse(X_real: pd.DataFrame, X_synth: pd.DataFrame, unique_id_cols: list, column_pairs: list) -> pd.DataFrame:
    """
    Compute the difference/mse in cross-correlation between synthetic and real dataframes.

    :param synthetic_df: Synthetic DataFrame.
    :type synthetic_df: pd.DataFrame
    :param real_df: Real DataFrame.
    :type real_df: pd.DataFrame
    :param unique_id_cols: List of columns that define unique groups.
    :type unique_id_cols: list
    :param column_pairs: List of tuple pairs specifying columns to compute cross-correlation.
    :type column_pairs: list of tuples

    :return: DataFrame with the absolute difference in cross-correlation values for each column pair.
    :rtype: pd.DataFrame
    """

    synth_crosscorr = compute_grouped_crosscorrelation(X_synth, unique_id_cols, column_pairs)
    real_crosscorr = compute_grouped_crosscorrelation(X_real, unique_id_cols, column_pairs)
    
    # mse 
    mse = ((synth_crosscorr - real_crosscorr)**2).mean(axis=0)[0]

    return mse

def autocorr_mse(X_real: pd.DataFrame, X_synth: pd.DataFrame, unique_id_cols: list, event_columns: list) -> float:
    """
    Compute the difference/mse in autocorrelation between synthetic and real dataframes.

    :param X_real: Real DataFrame with event columns.
    :type X_real: pd.DataFrame
    :param X_synth: Synthetic DataFrame with event columns.
    :type X_synth: pd.DataFrame
    :param unique_id_cols: List of columns that define unique groups.
    :type unique_id_cols: list
    :param event_columns: List of event column names that might contain temporal dependencies.
    :type event_columns: list

    :return: MSE in autocorrelation values for each event column.
    :rtype: float
    """
    
    synth_autocorr = compute_grouped_autocorrelation(X_synth, unique_id_cols, event_columns)
    real_autocorr = compute_grouped_autocorrelation(X_real, unique_id_cols, event_columns)
    
    # Compute the MSE 
    mse = ((synth_autocorr - real_autocorr)**2).mean(axis=1)[0]

    return mse
