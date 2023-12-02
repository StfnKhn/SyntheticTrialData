from __future__ import annotations

import os
import logging
import pandas as pd
import numpy as np
import pyarrow as pa

from typing import Union, List, Set
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from synthetic_trial_data.src.preprocessing.custom_transformers import (
    Object2BooleanTransformer,
    String2DateTransformer,
    Date2ContinousTransformer,
)
from synthetic_trial_data.src.utils.dataframe_handling import (
    categorize_columns_by_type
)

from synthetic_trial_data.src.preprocessing.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


def remove_prefixes(X: pd.DataFrame, prefixes:list):
    """
    This function removes given prefixes from the column names of a dataframe.

    :param df: The input dataframe.
    :type df: pd.DataFrame
    :param prefixes: The prefixes to be removed from the column names.
    :type prefixes: list of str
    :return: The dataframe with updated column names.
    :rtype: pd.DataFrame
    """
    X.columns = X.columns.str.replace('|'.join([f"^{prefix}__" for prefix in prefixes]), '', regex=True)
    return X

class Imputer(BaseProcessor):
    def __init__(
        self,
        use_prefix: bool = False,
    ):
        super().__init__()
        self.use_prefix = use_prefix

    def _create_numeric_transformer(self):
        self.numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(missing_values=pd.NA, strategy='mean')),
            ]
        )

    def _create_boolean_transformer(self):
        self.boolean_transformer = Pipeline(
            steps=[
                ("to_numeric", FunctionTransformer(lambda X: X.astype(float))),
                ("imputer", SimpleImputer(missing_values=pd.NA, strategy='most_frequent')),
                ("to_bool", FunctionTransformer(lambda X: X.astype(bool))),
            ]
        )

    def _create_categorical_transformer(self):
        # FunctionTransformer that removes all strings that only contain spaces
        remove_empty_strings = FunctionTransformer(lambda X: X.replace(r'^\s*$', pd.NA, regex=True))

        # Definition of imputer Step
        imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value='missing')

        # Definition of categorical transformation pipeline
        self.categorical_transformer = Pipeline(steps=[
            ("remove_empty_str", remove_empty_strings),
            ("imputer", imputer),
        ])
    
    def _create_pipeline(self):
        self._create_numeric_transformer()
        self._create_boolean_transformer()
        self._create_categorical_transformer()
        column_transformer = ColumnTransformer(
            transformers=[
                ("cat", self.categorical_transformer, self.categorical_col),
                ("num", self.numeric_transformer, self.numeric_col),
                ("bool", self.boolean_transformer, self.boolean_col),
            ]
        ).set_output(transform="pandas")

        self.pipeline = Pipeline(steps=[('impute', column_transformer)])

    def fit(self, X: pd.DataFrame) -> Imputer:

        # Group the columns by their data type
        super()._set_columns_by_type(X)

        # Set reprocucible column order
        X = super()._set_column_order(X)

        # Create pipeline
        self._create_pipeline()

        # Fit pipeline
        self.pipeline.fit(X)

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        X = self.pipeline.transform(X)
        
        if not self.use_prefix:
            X = remove_prefixes(X=X, prefixes=['cat', 'num', 'bool'])
            
        return X

    def inverse_transform(self, X) -> pd.DataFrame:
        """Reverts encoding if used
        """
        pass


class Encoder(BaseProcessor):
    def __init__(
        self,
        cat_encoder: Union[None, "ohe"] = "ohe",
        to_numpy: bool= False,
        scale_numeric: bool=False
    ):
        super().__init__()
        self.cat_encoder = cat_encoder
        self.scale_numeric = scale_numeric

    def _create_numeric_transformer(self):
        if self.scale_numeric:
            self.numeric_transformer = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler())
                ]
            ).set_output(transform="pandas")
        else:
            self.numeric_transformer = Pipeline(
                steps=[
                    ("passthrough", FunctionTransformer(lambda X: X)),
                ]
            ).set_output(transform="pandas")

    def _create_boolean_transformer(self):
        self.boolean_transformer = Pipeline(
            steps=[
                ("passthrough", FunctionTransformer(lambda X: X)),
            ]
        ).set_output(transform="pandas")

    def _create_categorical_transformer(self):
        imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value='missing')

        if self.cat_encoder == "ohe":
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.categorical_transformer = Pipeline(steps=[
                    ("encoder", ohe),
            ]).set_output(transform="pandas")
        else:
            raise ValueError("Only supportet encoders are ['ohe']")

    def _create_pipeline(self):
        self._create_numeric_transformer()
        self._create_boolean_transformer()
        self._create_categorical_transformer()

    def fit(self, X: pd.DataFrame) -> Imputer:

        # Group the columns by their data type
        super()._set_columns_by_type(X)

        # Set reprocucible column order
        X = super()._set_column_order(X)

        # Create pipeline
        self._create_pipeline()

        # Fit pipeline
        if self.categorical_col:
            self.categorical_transformer.fit(X[self.categorical_col])
        if self.numeric_col:
            self.numeric_transformer.fit(X[self.numeric_col])
        if self.boolean_col:
            self.boolean_transformer.fit(X[self.boolean_col])

        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()

        if self.categorical_col:
            X_cat = self.categorical_transformer.transform(X[self.categorical_col]).reset_index(drop=True)
        else:
            X_cat = pd.DataFrame(0, index=np.arange(len(X)), columns=[]).reset_index(drop=True)
        if self.numeric_col:
            X_num = self.numeric_transformer.transform(X[self.numeric_col]).reset_index(drop=True)
        else:
            X_num = pd.DataFrame(0, index=np.arange(len(X)), columns=[]).reset_index(drop=True)
        if self.boolean_col:
            X_bool = self.boolean_transformer.transform(X[self.boolean_col]).reset_index(drop=True)

        else:
            X_bool = pd.DataFrame(0, index=np.arange(len(X)), columns=[]).reset_index(drop=True)

        X_transformed = pd.concat([X_cat, X_num, X_bool], axis=1).reset_index(drop=True)
            
        return X_transformed

    def inverse_transform(self, X) -> pd.DataFrame:
        """Reverts encoding if used
        """
        self._check_is_fitted()

        self._cat_col_idx_ = len(self.categorical_transformer.get_feature_names_out())
        self._num_col_idx_ = len(self.numeric_transformer.get_feature_names_out()) + self._cat_col_idx_
        self._bol_col_idx_ = len(self.boolean_col) + self._cat_col_idx_ + self._num_col_idx_

        X_cat_encoded, X_num_encoded, X_bool_encoded, _ = np.split(
            X, [self._cat_col_idx_, self._num_col_idx_, self._bol_col_idx_], axis=1
        )

        if self.categorical_col:
            X_cat = self.categorical_transformer.inverse_transform(X_cat_encoded)
        else:
            X_cat = np.zeros([len(X), 0])
        if self.numeric_col:
            X_num = self.numeric_transformer.inverse_transform(X_num_encoded)
        else: 
            X_num = np.zeros([len(X), 0])
        if self.boolean_col:
            X_bool = X_bool_encoded
        else: 
            X_bool = np.zeros([len(X), 0])

        X_transformed = pd.concat([
            pd.DataFrame(X_cat, columns=self.categorical_col),
            pd.DataFrame(X_num, columns=self.numeric_col),
            pd.DataFrame(X_bool, columns=self.boolean_col)
        ], axis=1)

        return X_transformed


class DataTypeOptimizer(BaseProcessor):

    def __init__(self, include_date_columns=True):
        self.pipeline = None
        self.include_date_columns = include_date_columns

    def _create_pipeline(self):
        """Create a pipeline that applies a series of transformations 
        to correct the data types of DataFrame columns.

        The pipeline includes the following steps:
        - Convert object columns to boolean (if applicable)
        - Convert string columns to date (if applicable)
        - Convert date columns to continuous (unix timestamp)

        :return: A pipeline that applies data type corrections.
        :rtype: sklearn.pipeline.Pipeline
        """

        if self.include_date_columns:
            self.pipeline = Pipeline(steps=[
                    ("object_to_bool", Object2BooleanTransformer()),
                    ("string_to_date", String2DateTransformer()),
                    ("date_to_continous", Date2ContinousTransformer())
                ])
        else:
            self.pipeline = Pipeline(steps=[
                ("object_to_bool", Object2BooleanTransformer())
            ])


        

    def fit(self, X, y=None):
        """Fit the transformer. Since this transformer is stateless, 
        this method just returns self.

        :param X: The DataFrame to be transformed.
        :type X: pd.DataFrame
        :param y: Ignored for this transformer.
        :return: self
        """
        return self

    def transform(self, X: pd.DataFrame):
        self._create_pipeline()
        return self.pipeline.transform(X)


def create_preprocessing_pipeline(column_by_types):
    """Create a preprocessing pipeline that applies encoding to categorical features.

    :param column_by_types: A dictionary that maps column types to lists of column names.
    :type column_by_types: dict
    :return: A pipeline that applies preprocessing.
    :rtype: sklearn.pipeline.Pipeline

    :Example:

    >>> column_by_types = {
    >>>     "categorical": ["cat_col1", "cat_col2"],
    >>>     "numeric": ["num_col1", "num_col2"],
    >>>     "boolean": ["bool_col1", "bool_col2"]
    >>> }
    >>> preprocessing_pipeline = create_preprocessing_pipeline(column_by_types)
    >>> processed_df = preprocessing_pipeline.fit_transform(df)
    """

    # Impute Null values
    imputer = SimpleImputer(missing_values=pd.NA, strategy='constant', fill_value='missing')

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", imputer),
            # ("encoder", OneHotEncoder(
            #     handle_unknown="infrequent_if_exist",
            #     sparse_output=False
            # )),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy='mean')),
        ]
    )

    boolean_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy='most_frequent')),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, column_by_types["categorical"]),
            ("num", numeric_transformer, column_by_types["numeric"]),
            ("bool", boolean_transformer, column_by_types["boolean"]),
        ]
    ).set_output(transform="pandas")

    preprocessessing_pipe = Pipeline(steps=[
        ('encoding', column_transformer),
    ])
    
    return preprocessessing_pipe
