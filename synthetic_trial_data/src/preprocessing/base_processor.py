from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from typeguard import typechecked

from synthetic_trial_data.src.utils.dataframe_handling import categorize_columns_by_type


@typechecked
class BaseProcessor(ABC, BaseEstimator, TransformerMixin):

    def __init__(self):
        self.categorical_col = None
        self.numeric_col = None
        self.boolean_col = None
        self._column_order = None
        self.categorical_transformer = None
        self.numeric_transformer = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> BaseProcessor:
        """Fits the Processor to a passed pd.DataFrame.

        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """Transforms the passed pd.DataFrame with the fit DataProcessor.
        """
        raise NotImplementedError

    def _check_is_fitted(self):
        """Checks if the processor is fitted by testing the numerical pipeline.
        Raises NotFittedError if not."""
        if self.categorical_transformer is None and self.numeric_transformer is None:
            raise NotFittedError("This data processor has not yet been fitted.")
    
    def _set_columns_by_type(self, X: pd.DataFrame):
        self.column_by_types = categorize_columns_by_type(X)
        self.categorical_col = self.column_by_types["categorical"]
        self.numeric_col = self.column_by_types["numeric"]
        self.boolean_col = self.column_by_types["boolean"]

    def _set_column_order(self, X):
        self._column_order = self.categorical_col + self.numeric_col + self.boolean_col
        return X[self._column_order]

