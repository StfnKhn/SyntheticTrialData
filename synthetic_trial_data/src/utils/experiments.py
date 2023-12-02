import os
from datetime import datetime
import pandas as pd
import pickle
import math


def make_model_checkpoint_dir(path_to_dir: os.path) -> str:
    """
    Create a new directory within the specified directory, named with the current datetime stamp.
    
    :param path_to_dir: Parent directory where the new directory will be created.
    :type path_to_dir: os.path
    :return: Full path to the newly created directory.
    :rtype: str
    
    Example:
    .. code-block:: python

        parent_dir = "/path/to/parent_directory"
        checkpoint_dir = make_model_checkpoint_dir(parent_dir)
        print(checkpoint_dir)
        # Expected output: "/path/to/parent_directory/%Y%m%d_%H%M%S

    """
    # Create a new directory with the current datetime stamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(path_to_dir, timestamp)
    
    # Ensure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    return checkpoint_dir


def check_directory_exists(path_to_dir: str, additional_msg: str = None) -> None:
    """
    Check if the specified directory exists and raise an AssertionError if it doesn't.

    :param path_to_dir: The path to the directory that needs to be checked.
    :type path_to_dir: str
    :raises AssertionError: If the directory does not exist.
    """
    error_msg = f"Directory {path_to_dir} does not exist!"
    if additional_msg != None:
        error_msg = error_msg + " " + additional_msg
    assert os.path.exists(path_to_dir) and os.path.isdir(path_to_dir), error_msg


def check_path_exists(path: str, additional_msg: str = None) -> None:
    """
    Check if the specified path exists and raise an AssertionError if it doesn't.

    :param path_to_dir: The path that needs to be checked.
    :type path_to_dir: str
    :raises AssertionError: If the directory does not exist.
    """
    error_msg = f"Path {path} does not exist!"
    if additional_msg != None:
        error_msg = error_msg + " " + additional_msg
    assert os.path.exists(path), error_msg


def mix_datasets(X_synth: pd.DataFrame, X_real: pd.DataFrame, fraction):
    """
    Mix synthetic and real datasets based on the specified fraction to create a blended dataset.

    This function takes in both a synthetic dataset and a real dataset, then replaces a fraction 
    of the synthetic dataset with a corresponding fraction of the real dataset. This is useful for 
    scenarios where the aim is to augment real data with synthetic data while retaining some real-world patterns.

    :param X_synth: The input DataFrame containing synthetic data.
    :type X_synth: pd.DataFrame
    :param X_real: The input DataFrame containing real-world data.
    :type X_real: pd.DataFrame
    :param fraction: The fraction of synthetic data entries to replace with entries from the real data. 
                     The value should be between 0 and 1.
    :type fraction: float
    :return: A DataFrame containing the blended data resulting from the combination of synthetic 
             and real data based on the specified fraction.
    :rtype: pd.DataFrame

    Raises:
    - AssertionError: If the columns of synthetic_data and real_data do not match.
    """

    # Ensure the datasets have the same columns
    assert all(X_synth.columns == X_real.columns), "Datasets must have the same columns."

    # Number of records to replace
    num_to_replace = int(len(X_synth) * fraction)

    # Randomly sample from real data
    real_sample = X_real.sample(n=num_to_replace, replace=False)

    # Drop records from synthetic data
    synthetic_remaining = X_synth.drop(X_synth.sample(n=num_to_replace, replace=False).index)

    # Concatenate the remaining synthetic data with the real data sample
    blended_data = pd.concat([synthetic_remaining, real_sample], ignore_index=True).sample(frac=1).reset_index(drop=True)

    return blended_data


def save_as_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_from_pickle(file_path: str):
    """
    Load a serialized object from a pickle file.
    
    :param file_path: The path to the pickle file.
    :type file_path: str
    :return: Deserialized object from the pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)


def get_batch_sizes(low:int, high:int):
    max_i = int(math.log(high, 2))
    min_i = int(math.log(low, 2))
    return [2 ** i for i in range(min_i, max_i)]
