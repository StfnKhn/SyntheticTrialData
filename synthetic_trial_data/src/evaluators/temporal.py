from typing import Union, List, Tuple
import numpy as np
import pandas as pd

from synthetic_trial_data.src.evaluators.base_evaluator import BaseEvaluator
from synthetic_trial_data.src.utils.dataframe_handling import to_dataframe
from synthetic_trial_data.src.metrics.correlation_difference import crosscorr_mse, autocorr_mse


class CrossCorrelationDifference(BaseEvaluator):
    """
    Computes the difference in crosscorelation of the synthetic and the real data set
    using the mean squared error
    """

    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "CrossCorrelationDifference"
    
    @staticmethod
    def metrics() -> List[str]:
        return [__class__.name()]

    @staticmethod
    def direction() -> str:
        return "minimize"
    
    @staticmethod
    def polarity() -> int:
        return {__class__.name(): -1}
        
    def evaluate(
        self,
        X_real: Union[np.ndarray, pd.DataFrame],
        X_synth: Union[np.ndarray, pd.DataFrame],
        unique_id_cols: List[str], 
        cross_corr_col_pairs: List[Tuple[str]]
    ):
        """
        Computes the difference in crosscorrelation of two data sets.

        :param X_real: The original (real) data
        :type X_real: np.array
        :param X_synth: The synthetic data
        :type X_synth: np.array
        :param kwargs: Other optional parameters
        :type kwargs: dict, optional
        """

        # Convert data to pd.DataFrame if necessary
        X_real = to_dataframe(X_real)
        X_synth = to_dataframe(X_synth)

        # Compute metric
        mse = crosscorr_mse(
            X_real=X_real,
            X_synth=X_synth,
            unique_id_cols=unique_id_cols,
            column_pairs=cross_corr_col_pairs
        )

        result = {
            __class__.name(): mse
        }
        
        return result


class AutoCorrelationDifference(BaseEvaluator):
    """
    Computes the difference in autocorrelation of the synthetic and the real data set
    using the mean squared error
    """

    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "AutoCorrelationDifference"
    
    @staticmethod
    def metrics() -> List[str]:
        return [__class__.name()]

    @staticmethod
    def direction() -> str:
        return "minimize"
    
    @staticmethod
    def polarity() -> int:
        return {__class__.name(): -1}
        
    def evaluate(
        self,
        X_real: Union[np.ndarray, pd.DataFrame],
        X_synth: Union[np.ndarray, pd.DataFrame],
        unique_id_cols: List[str], 
        event_columns: List[str]
    ):
        """
        Computes the difference in autocorrelation of two data sets.

        :param X_real: The original (real) data
        :type X_real: np.array
        :param X_synth: The synthetic data
        :type X_synth: np.array
        :param kwargs: Other optional parameters
        :type kwargs: dict, optional
        """

        # Convert data to pd.DataFrame if necessary
        X_real = to_dataframe(X_real)
        X_synth = to_dataframe(X_synth)

        # Compute metric
        mse = autocorr_mse(
            X_real=X_real,
            X_synth=X_synth,
            unique_id_cols=unique_id_cols,
            event_columns=event_columns
        )

        result = {
            __class__.name(): mse
        }
        
        return result