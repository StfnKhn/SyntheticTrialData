from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typeguard import typechecked

@typechecked
class BaseModel(ABC, BaseEstimator, TransformerMixin):

    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: pd.DataFrame):
        """Fits the Model to a passed pd.DataFrame.

        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, X: DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """
        """
        raise NotImplementedError