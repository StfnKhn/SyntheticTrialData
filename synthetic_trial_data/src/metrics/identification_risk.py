import os
import logging
import pandas as pd
import numpy as np
from typing import Union, List
from sklearn.cluster import KMeans
from jenkspy import JenksNaturalBreaks

from synthetic_trial_data.src.utils.dataframe_handling import categorize_columns_by_type

logger = logging.getLogger(__name__)

def assign_group_size(X, QID, group_sizes, col_name='group_size'):
    """Assigns group sizes to the DataFrame based on the provided 
    quasi-identifiers (QID) and the group sizes dictionary.

    :param X: DataFrame with the data.
    :type X: pd.DataFrame
    :param QID: List of column names to be treated as quasi-identifiers.
    :type QID: list
    :param group_sizes: Dictionary with QID tuple keys and group size values.
    :type group_sizes: dict
    :param col_name: Name of the new column that will hold the group size. Default is 
        'group_size'.
    :type col_name: str
    
    :return: DataFrame with an added column named as per 'col_name' parameter.
    :rtype: pd.DataFrame
    """
    X[col_name] = X.apply(lambda row: group_sizes.get(tuple(row[QID]), 0), axis=1)
    
    return X

def assign_indicators(X, QID, dict_real, dict_synth, col_name):
    """Checks if each key in dict_real exists in dict_synth and assigns the result 
    to the DataFrame rows that match the respective set of QID columns.

    :param X: DataFrame with the data.
    :type X: pd.DataFrame
    :param QID: List of column names to be treated as quasi-identifiers.
    :type QID: list
    :param dict_real: Dictionary with QID tuple keys from the real dataset.
    :type dict_real: dict
    :param dict_synth: Dictionary with QID tuple keys from the synthetic dataset.
    :type dict_synth: dict
    :param col_name: Name of the new column that will hold the indicator. Default is 'key_exists'.
    :type col_name: str
    
    :return: DataFrame with an added column named as per 'col_name' parameter.
    :rtype: pd.DataFrame
    """
    key_indicator_map = {key: key in dict_synth for key in dict_real}
    X[col_name] = X.apply(lambda row: key_indicator_map.get(tuple(row[QID]), 0), axis=1)
    
    return X


def optimal_k_jenks(X, max_k=20, gvf_threshold=0.99):
    """
    Determine optimal k using Jenks Natural Breaks and goodness of fit.

    :param X: Data array for which optimal k has to be determined.
    :type X: np.array
    :param max_k: Maximum number of clusters to consider. Default is 20.
    :type max_k: int
    :param gvf_threshold: Threshold for goodness of variance fit. Default is 0.99.
    :type gvf_threshold: float
    
    :return: Optimal number of clusters k.
    :rtype: int
    """
    unique_values = len(np.unique(X))
    upper_limit = min(max_k, unique_values)
    logger.debug(f"upper_limit: {upper_limit}")
    if upper_limit == 1:
        k = 1
        return k
    for k in range(2, upper_limit+1):
        jnb = JenksNaturalBreaks(k)
        jnb.fit(X)
        gvf = jnb.goodness_of_variance_fit(X)
        if gvf > gvf_threshold:
            return k
    return k


def discretize_column(df, column, k_clusters):
    """
    Discretizes continuous columns using Jenks Natural Breaks.

    :param df: DataFrame containing the continuous data column.
    :type df: pd.DataFrame
    :param column: Name of the column to be discretized.
    :type column: str
    :param k_clusters: Number of clusters for discretization.
    :type k_clusters: int
    
    :return: DataFrame with an added column for discretized data.
    :rtype: pd.DataFrame
    """
    discretized_df = df.copy()
    X = df[column].values
    
    # Use optimal breaks to bin the data
    jnb = JenksNaturalBreaks(k_clusters)
    jnb.fit(X)
    breaks = jnb.breaks_
    labels = jnb.labels_
    
    # Replace continuous values with bin labels
    discretized_df[f"cluster_{column}"] = labels
        
    return discretized_df


def compute_Rs_for_nominal(X_real, X_synth, col, QID: List[str]):
    """
    Computes `Rs` for a categorical and binary column for every real record s. 
    `Rs` is an indicator variable that takes a value of 1 if the adversary
    would learn something new from matching the records s to the synthetic data
    
    :param X_real: Real dataset with actual records.
    :type X_real: pd.DataFrame
    :param X_synth: Synthetic dataset.
    :type X_synth: pd.DataFrame
    :param col: Name of the nominal column for which `Rs` is computed.
    :type col: str
    :param QID: List of quasi-identifier columns.
    :type QID: List[str]
    
    :return: Series with `Rs` values for the given column.
    :rtype: pd.Series
    """
    X_real = X_real.copy()

    # To increase efficiency, we use only columns QID + col
    X_real = X_real[QID + [col]]

    # Calculate the proportion of records in X_real that have the same value for each sensitive variable
    X_real["p_j"] = X_real[col].map(X_real.groupby(col).size() / len(X_real))

    # Calculate dj sensitive variable
    X_real['d_j'] = 1 - X_real['p_j']
    X_real["threshold"] = np.sqrt(X_real['p_j'] * (1 - X_real["p_j"]))

    # Determine the matches between the real and synthetic dataframes based on QI
    # Only take unique combinations from both dataframes
    unique_real = X_real.drop_duplicates(subset=QID)
    unique_synth = X_synth[QID + [col]].drop_duplicates(subset=QID)
    
    matches = pd.merge(unique_real, unique_synth, on=QID, suffixes=('', '_synth'))

    # Compute the similarity for sensitive variable
    matches[f'similarity_{col}'] = matches['d_j'] * (matches[col] == matches[col + '_synth'])
    matches['Rs'] = (matches[f'similarity_{col}'] > matches['threshold']).astype(int)
    
    # Merge back Rs values based on QID
    X_real = X_real.merge(matches[QID + [col, 'Rs']], on=QID, how="left", suffixes=('', '_synth'))

    # Handle missing values
    X_real['Rs'] = X_real.apply(lambda row: 0 if row[col] == "missing" else row['Rs'], axis=1)

    return X_real["Rs"]


def compute_Rs_for_continous(X_real, X_synth, col, QID: List[str]):
    """
    Computes `Rs` for a continuous column for every real record s. 
    `Rs` is an indicator variable that takes a value of 1 if the adversary
    would learn something new from matching the records s to the synthetic data
    
    :param X_real: Real dataset with actual records.
    :type X_real: pd.DataFrame
    :param X_synth: Synthetic dataset.
    :type X_synth: pd.DataFrame
    :param col: Name of the continuous column for which `Rs` is computed.
    :type col: str
    :param QID: List of quasi-identifier columns.
    :type QID: List[str]
    
    :return: DataFrame with an added column for `Rs` values.
    :rtype: pd.DataFrame
    """
    X = X_real[col].values
    
    # Determine optimal k using Jenks Natural Breaks and goodness of fit
    optimal_clusters = optimal_k_jenks(X)
    logger.debug(f"Optimal number of clusters for column '{col}': k = {optimal_clusters}")

    # Discretize sensitive column using Jenks Natural Breaks
    X_real = discretize_column(X_real, col, optimal_clusters)

    # Calculate cluster sizes and proportions for real data
    cluster_sizes = X_real.groupby(f'cluster_{col}').size()
    X_real[f'Cs_{col}'] = X_real[f'cluster_{col}'].map(cluster_sizes)

    # Compute proportion of records p_s
    X_real[f'ps_{col}'] = X_real[f'Cs_{col}'] / len(X_real)

    # Compute distance d_s
    X_real[f'ds_{col}'] = X_real[f'ps_{col}']

    # Calculate MAD for real data
    mad = 1.48 * np.median(np.abs(X - np.median(X)))
    
    # Calculate weighted difference between real and synthetic
    Yt = X_synth[col].values
    X_real[f'difference_{col}'] = X_real[f'ds_{col}'] * np.abs(X - Yt)
    
    # Compare with MAD
    X_real[f'Rs_{col}'] = (X_real[f'difference_{col}'] < mad).astype(int)

    # Drop columns with interim results
    X_real.drop(
        columns=[
            f'cluster_{col}',
            f'Cs_{col}',
            f'ps_{col}',
            f'ds_{col}',
            f'difference_{col}',
        ], inplace=True
    )

    return X_real


def identification_risk(X_real: pd.DataFrame, X_synth: pd.DataFrame, QID: List[str], S: List[str]):
    """
    Compute the identification risk for given sensitive columns in the real dataset 
    against the synthetic dataset. The methodology is proposed by :cite:`Emam2020` and 
    enhanced by :cite:`Mendelevitch2021`. The enhancements made by :cite:`Mendelevitch2021`
    are all used in this implementation. It is important to note that the adjustment factor
    $\lambda_s^'$ is fixed to 1 in this implementation, which corresponds to the most
    conservative assumption for this factor.
    
    :param X_real: Real dataset with actual records.
    :type X_real: pd.DataFrame
    :param X_synth: Synthetic dataset.
    :type X_synth: pd.DataFrame
    :param QID: List of quasi-identifier columns.
    :type QID: List[str]
    :param S: List of sensitive columns for which the risk is to be calculated.
    :type S: List[str]
    
    :return: Dictionary with the calculated risk per sensitive column.
    :rtype: dict[str, float]

    Example:
    .. code-block:: python

        # Create some sample data
        data_real = {
            "age": [25, 30, 35, 40, 45],
            "gender": ["male", "female", "male", "female", "male"],
            "income": [50000, 55000, 60000, 65000, 70000],
            "married": [True, False, True, False, True]
        }

        data_synth = {
            "age": [26, 31, 36, 41, 46],
            "gender": ["male", "female", "male", "female", "male"],
            "income": [51000, 56000, 61000, 66000, 71000],
            "married": [True, False, True, False, True]
        }

        df_real = pd.DataFrame(data_real)
        df_synth = pd.DataFrame(data_synth)

        # Define QID and S
        QID = ["age", "gender"]
        S = ["income", "married"]

        # Compute the identification risk
        risk = identification_risk(df_real, df_synth, QID, S)
        print(risk)
    
    .. [Emam2020] El Emam, K., Mosquera, L., & Bass, J. (2020). 
        Evaluating identity disclosure risk in fully synthetic health data: 
        model development and validation. Journal of medical Internet 
        research, 22(11), e23139.

    .. [Mendelevitch2021] Mendelevitch, O., & Lesh, M. D. (2021). 
        Fidelity and privacy of synthetic medical data. 
        arXiv preprint arXiv:2101.08658.
    """
    X_real = X_real.copy()
    
    # Number of records in the true population
    N = X_real.shape[0]

    # Compute Equivalence class group size in the real sample
    groups_real = dict(X_real[QID].value_counts())
    groups_synth = dict(X_synth[QID].value_counts())

    # I_s: Add indicators
    X_real = assign_indicators(X_real, QID, groups_real, groups_synth, col_name='I_s')

    # f_s:  group size in the real sample for a particular record s in the real sample
    X_real = assign_group_size(X=X_real, QID=QID, group_sizes=groups_real, col_name="f_s")

    # F_s: group size in the population that has the same QIDs values as record s in the real sample
    X_real = assign_group_size(X=X_real, QID=QID, group_sizes=groups_synth, col_name="F_s")

    # categorize columns by type
    cols_by_type = categorize_columns_by_type(X_real[S])
    
    # Compute Rs for numeric columns
    for col in cols_by_type["numeric"]:
        X_real = compute_Rs_for_continous(X_real, X_synth, col=col, QID=QID)

    # Compute Rs for nominal columns
    for col in cols_by_type["categorical"] + cols_by_type["boolean"]:
        X_real[f"Rs_{col}"] = compute_Rs_for_nominal(X_real, X_synth, col=col, QID=QID)

    # Compute overall attribute disclosure risk per variable
    risk_per_col = {}
    for col in S:
        f_s = X_real["f_s"]
        I_s = X_real["I_s"]
        F_s = X_real["F_s"]
        R_s = X_real[f"Rs_{col}"]
        N_synth = X_synth.shape[0]

        # population-to-sample attack
        A = 1/N * (1/f_s * I_s * R_s).sum()
        # sample-to-population attack
        B = 1/N_synth * (1/F_s * I_s * R_s).sum()

        # Compute risk per column
        risk_per_col[col] = max(np.round(A, 4), np.round(B, 4))

    return risk_per_col
