import os
import re
import logging
import pandas as pd
import numpy as np
import polars as pl

from typing import Union, List, Set
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer

from synthetic_trial_data.src.utils.dataframe_handling import (
    categorize_columns_by_type
)

logger = logging.getLogger(__name__)


class Object2BooleanTransformer(BaseEstimator, TransformerMixin):
    """Transforms object columns in a dataframe into boolean columns based on
    predefined possible mappings.

    :param possible_mappings: List of dictionaries containing mapping from object to boolean.
    :type possible_mappings: List[Set], optional
    :param null_value_indicators: List of values indicating null values in the data.
    :type null_value_indicators: List, optional
    :param columns_to_transform: List of columns to be transformed in the DataFrame.
    :type columns_to_transform: list, optional
    """

    def __init__(
        self, 
        possible_mappings: List[Set] = None,
        null_value_indicators: List = None,
        columns_to_transform: list = None
    ) -> None:
        self.possible_mappings = possible_mappings
        self.null_value_indicators = null_value_indicators
        self.columns_to_transform = columns_to_transform

        if not possible_mappings:
            self.possible_mappings = [
                {"Yes": True, "No": False},
                {"YES": True, "NO": False},
                {"True": True, "False": False},
                {"true": True, "false": False},
                {"TRUE": True, "FALSE": False},
                {"1": True, "0": False},
            ]

        if not null_value_indicators:
            self.null_value_indicators = ["", np.nan, " ", None]

    def check_boolean(self, df: pl.DataFrame, col: str):
        """Checks and transforms a single column in a dataframe into boolean type.
        
        :param df: DataFrame to be processed.
        :type df: polars.DataFrame
        :param col: Column in df to be checked and potentially transformed.
        :type col: str
        :return: DataFrame with the processed column.
        :rtype: polars.DataFrame
        """
        
        for map_dict in self.possible_mappings:
            complete_list = list(map_dict.keys())
            complete_list.extend(self.null_value_indicators)
            unique_values = df[col].unique()
                
            # Check if unique_values is a subset of complete_list
            is_subset = set(unique_values).issubset(set(complete_list))
            
            if is_subset:
                # Handle missing values
                modified_col = df[col].set(df[col].is_in(self.null_value_indicators), None)
                df = df.replace(col, modified_col)            
                # Convert to boolean according to map_dict
                df = df.replace(col, df[col].map_dict(map_dict))
                
                return df
                
        return df
    
    def fit(self, X, y=None):
        """Fit the transformer. Since this transformer is stateless, 
        this method just returns self.

        :param X: The DataFrame to be transformed.
        :type X: pd.DataFrame
        :param y: Ignored for this transformer.
        :return: self
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check each column in the provided DataFrame, if a column contains
        only two unique values (except the options given in self.null_value_indicators), 
        that are included in possible_mappings, then this column will be converted 
        to boolean. 
        
        :param df: The DataFrame to process.
        :type df: pandas.DataFrame
        :return: DataFrame with converted columns.
        :rtype: pandas.DataFrame
        """

        # If no columns specified check all categorical columns
        if not self.columns_to_transform:
            self.columns_to_transform = categorize_columns_by_type(df)["categorical"]

        df_pl = pl.from_pandas(df)

        for col in self.columns_to_transform:
            df_pl = self.check_boolean(df=df_pl, col=col)
        
        logger.info(f"Columns {self.columns_to_transform} will be converted from string to bool")

        df = df_pl.to_pandas(use_pyarrow_extension_array=False)
        df = df.where(pd.notnull(df), np.nan)

        return df


class String2DateTransformer(BaseEstimator, TransformerMixin):
    """A class used to transform string columns to date columns in the 
    provided format.
    
    :param format: Desired date format (default is "%Y-%m-%d")
    :type format: str, optional
    :param date_delimiter: Delimiter used in date representation (default is "-")
    :type date_delimiter: str, optional
    """
    
    def __init__(
        self, 
        format="%Y-%m-%d",
        date_delimiter: str = "-"
    ):
        self.format = format
        self.date_delimiter = date_delimiter

    @staticmethod
    def __date_indicator_map(string, delimiter="-"):
        """Private static method to check if a string follows a specific pattern.

        :param string: String to check.
        :type string: str
        :param delimiter: Delimiter used in pattern (default is "-")
        :type delimiter: str, optional
        :return: Boolean indicating if the string follows the pattern.
        :rtype: bool
        """
        pattern = r"^\d+{0}\d+{0}\d+$".format(re.escape(delimiter))
        if bool(re.match(pattern, string)) and len(string) == 10:
            return True
        return False

    def is_date_format(self, column: pl.Series):
        """Checks if a given Polars Series follows the defined date format.

        :param column: A Polars Series.
        :type column: pl.Series
        :return: Boolean indicating if the series follows the date format.
        :rtype: bool
        """
        # Sets all elements to None that do not contain len(format) characters 
        column = column.apply(
            lambda val: val if self.__date_indicator_map(val, self.date_delimiter) else None, 
            return_dtype=pl.datatypes.Utf8
        )

        # Count non-null values in the series before the parse
        non_null_count_before = column.is_not_null().sum()

        # Attempt to parse column as date with given format
        parsed_column = column.str.to_date(format=self.format, strict=False)

        # Count non-null values in the parsed series
        non_null_count_after = parsed_column.is_not_null().sum()
        
        if non_null_count_before == non_null_count_after and non_null_count_before>0:
            return True
        else:
            return False
    
    def get_hidden_date_columns(self, df: Union[pl.DataFrame, pl.Series]) -> list:
        """Detects if columns in a DataFrame or elements in a Series 
        are dates in the provided format.

        :param df: DataFrame or Series containing the data to be checked.
        :type df: Union[pl.DataFrame, pl.Series]
        :return: List of columns (in case of DataFrame) or names (in case of Series)
            that match the date format.
        :rtype: list
        :raises ValueError: If the provided df is not of type {pl.DataFrame, pl.Series}.
        """
        date_cols = []
        
        columns_to_check = categorize_columns_by_type(df)["categorical"]
        df = df[columns_to_check]

        if isinstance(df, pl.DataFrame):
            for col_name in df.columns:
                if self.is_date_format(df[col_name]):
                    date_cols.append(col_name)     
        elif isinstance(df, pl.Series):
            if self.is_date_format(df):
                date_cols.append(df.name)   
        else:
            raise ValueError("df needs to be of type {pl.DataFrame, pl.Series}")

        return date_cols

    def fit(self, X, y=None):
        """Fit the transformer. Since this transformer is stateless,
        this method just returns self.

        :param X: The DataFrame to be transformed.
        :type X: pd.DataFrame
        :param y: Ignored for this transformer.
        :return: self
        """
        return self

    def transform(
        self,
        df: Union[pd.DataFrame, pd.Series],
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Transform specified columns in a DataFrame to dates in the provided format.

        :param df: DataFrame containing the data
        :type df: polars.DataFrame
        :param columns: List of column names to transform
        :type columns: list
        :return: DataFrame with transformed columns.
        :rtype: pl.DataFrame
        :raises ValueError: If the provided df is not of type {pl.DataFrame, pl.Series}.
        """
        df = pl.from_pandas(df)
        if not columns:
            columns = self.get_hidden_date_columns(df)
            logger.info(f"Columns {columns} will be converted from string to datetime.date")
            
        if isinstance(df, pl.DataFrame):
            for col_name in columns:
                transformed_column = df[col_name].str.to_date(format=self.format, strict=False)
                df = df.with_columns(transformed_column.alias(col_name))
                
        elif isinstance(df, pl.Series):
            df = df.str.to_date(format=self.format, strict=False)
            
        df = df.to_pandas(use_pyarrow_extension_array=False)
        df = df.where(pd.notnull(df), np.nan)

        return df


class Date2ContinousTransformer(BaseEstimator, TransformerMixin):
    """Transformer for converting date columns in a DataFrame to Unix timestamps, 
    making them continuous and suitable for use in many machine learning algorithms.

    This class inherits from the BaseEstimator and TransformerMixin classes from sklearn.base,
    which provides standard methods for the pipeline fitting process.

    :param date_cols: The list of column names to convert. If not provided, 
        all datetime columns will be converted.
    :type date_cols: List[str]
    """

    def __init__(self, date_cols: List[str] = None):
        self.date_cols = date_cols  

    def fit(self, X, y=None):
        """Fit the transformer. Since this transformer is stateless,
        this method just returns self.

        :param X: The DataFrame to be transformed.
        :type X: pd.DataFrame
        :param y: Ignored for this transformer.
        :return: self
        """
        return self

    def transform(self, X):
        """Transform the specified date columns in X into Unix timestamps.

        :param X: The DataFrame to be transformed.
        :type X: pd.DataFrame
        :return: DataFrame with the date columns converted.
        :rtype: pd.DataFrame
        """
        if not self.date_cols:
            colums_by_type = categorize_columns_by_type(X)
            self.date_cols = colums_by_type["datetime"]

        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col]).dt.normalize()
            X[col] = X[col].astype(int) / 10**9

        return X


class ContinuousBinner(BaseEstimator, TransformerMixin):
    """
    Transformer for binning continuous data.

    This transformer computes bins for continuous data columns and assigns each 
    value to its respective bin. This is useful for making continuous features 
    categorical and potentially beneficial for certain machine learning models.

    :param nbins: Number of bins to use for quantization.
    :type nbins: int
    :param columns: List of columns to apply binning. If None, applies to all columns.
    :type columns: list or None
    """
    def __init__(self, nbins, columns, strategy="kmeans"):
        self.nbins = nbins
        self.columns = columns
        self.strategy = strategy
        self.bin_params = {}

    def fit(self, X, y=None):
        """
        Compute bin edges, centers, and widths for specified columns in the dataframe.

        :param X: DataFrame containing continuous data.
        :type X: pd.DataFrame
        :param y: Ignored.
        :return: self.
        """
        for column in self.columns:
            discretizer = KBinsDiscretizer(
                n_bins=self.nbins,
                encode='ordinal',
                strategy=self.strategy,
                subsample=None
            )
            discretizer.fit(X[column].to_numpy().reshape(-1, 1))
            self.bin_params[column] = discretizer
        return self
    
    def transform(self, X, y=None):
        """
        Transform the dataframe by quantizing specified columns using pre-computed bin parameters.

        :param X: DataFrame containing continuous data.
        :type X: pd.DataFrame
        :param y: Ignored.
        :return: Transformed DataFrame.
        :rtype: pd.DataFrame
        """
        X_binned = X.copy()

        for column, discretizer in self.bin_params.items():
            X_binned[column] = discretizer.transform(X[column].to_numpy().reshape(-1, 1))
        return X_binned

    def inverse_transform(self, X):
        """
        Inverse transform the binned data back to continuous form using bin centers.

        :param X: DataFrame containing binned data.
        :type X: pd.DataFrame
        :return: Approximate continuous data DataFrame.
        :rtype: pd.DataFrame
        """
        X_continuous = X.copy()

        for column, discretizer in self.bin_params.items():
            X_continuous[column] = discretizer.inverse_transform(X[column].to_numpy().reshape(-1, 1))
        return X_continuous


class CountVariableTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """
        Placeholder, since no fitting step is required for this transformation
        :param X: DataFrame
        :type X: pd.DataFrame
        :param y: Ignored.
        :return: self
        """
        return self
    
    def transform(self, X, y=None):
        """
        Converts all count from float to integer variables to integers

        :param X: DataFrame continuous count data.
        :type X: pd.DataFrame
        :param y: Ignored.
        :return: Transformed DataFrame.
        :rtype: pd.DataFrame
        """
        X = X.copy()
        for column in self.columns:
            X[column] = X[column].round().astype(int)
    
        return X

    def inverse_transform(self, X):
        """
        Placeholder, no inverse transformation required

        :param X: DataFrame containing binned data.
        :type X: pd.DataFrame
        :return: Approximate continuous data DataFrame.
        :rtype: pd.DataFrame
        """
        return X


class TimeDifference(BaseEstimator, TransformerMixin):

    def __init__(
        self, 
        timestamp_col: str,
        unique_id: str,
        unit: Union['days', 'seconds', 'raw']
    ):
        """
        Initializes the transformer with the date column, patient ID column, and the desired unit.

        :param timestamp_col: The column containing the date information.
        :type timestamp_col: str
        :param unique_id: The column containing the unique patient IDs.
        :type unique_id: str
        :param unit: The unit for time difference, one of 'days', 'seconds', or 'raw'.
        :type unit: str
        """
        self.timestamp_col = timestamp_col
        self.unique_id = unique_id
        self.unit = unit
        assert unit in ['days', 'seconds', 'raw'], "Invalid unit. Choose one of 'days', 'seconds', or 'raw'."

    def fit(self, X, y=None):
        """
        Placeholder, since no fitting step is required for this transformation
        :param X: DataFrame
        :type X: pd.DataFrame
        :param y: Ignored.
        :return: self
        """
        return self

    def transform(self, X, y=None):
        """
        Introduce the feature that measures the time difference in the specified unit 
        to the previous record of each patient.

        :param X: DataFrame containing patient records with date information.
        :type X: pd.DataFrame
        :param y: Ignored.
        :return: Transformed DataFrame with an additional column for time difference.
        :rtype: pd.DataFrame
        """
        X = X.copy()
        
        # Convert the date column to datetime type if it's not already
        X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col], errors='coerce')

        # Sort data within each patient group by the timestamp
        X.sort_values(by=[self.unique_id, self.timestamp_col], inplace=True)
        
        # Calculate the time difference based on the desired unit
        diff = X.groupby(self.unique_id)[self.timestamp_col].diff()
        if self.unit == 'days':
            X['T_diff'] = diff.dt.days
        elif self.unit == 'seconds':
            X['T_diff'] = diff.dt.seconds
        else:
            X['T_diff'] = diff
        
        # Replace NaN values (for the first record of each patient) with 0 or equivalent representation
        X['T_diff'].fillna(pd.Timedelta(seconds=0) if self.unit == 'raw' else 0, inplace=True)
        X['T_diff'] = X['T_diff'].astype(int)
        
        return X

    def inverse_transform(self, X):
        """
        Placeholder, no inverse transformation required

        :param X: DataFrame containing the T_diff column.
        :type X: pd.DataFrame
        :return: DataFrame without the T_diff column.
        :rtype: pd.DataFrame
        """
        return X #X.drop('T_diff', axis=1)


class NaNInsertTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer to insert NaN values into a target column based on the missing values in a source column.

    :param source_col: Name of the column to check for missing values.
    :type source_col: str
    :param target_col: Name of the column to insert np.nan values based on the source column.
    :type target_col: str

    Example:
    .. code-block:: python

        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': ['a', 'b', 'c', 'd']
        })

        transformer = NaNInsertTransformer(source_col='A', target_col='B')
        transformed_df = transformer.fit_transform(df)
        print(transformed_df)
        # Expected output:
        #     A    B
        # 0  1.0    a
        # 1  2.0    b
        # 2  NaN  NaN
        # 3  4.0    d

    """

    def __init__(self, source_col, target_col):
        self.source_col = source_col
        self.target_col = target_col
    
    def fit(self, X, y=None):
        """
        Fit method for the transformer. This transformer does not need to learn anything 
        from the data, so it just returns itself.
        
        :param X: The input data to be transformed.
        :type X: pd.DataFrame
        :param y: Target values. Not used in this transformer.
        :type y: array-like, default=None
        :return: The transformer instance.
        :rtype: NaNInsertTransformer
        """
        return self
    
    def transform(self, X):
        """
        Apply the transformation on the data based on the source and target columns.
        
        :param X: The input data to be transformed.
        :type X: pd.DataFrame
        :return: Transformed data with NaN values inserted into the target column based on the missing values in the source column.
        :rtype: pd.DataFrame
        """
        # Ensure the input is a DataFrame
        X = pd.DataFrame(X)
        
        # Set target column to np.nan where source column is missing
        X.loc[X[self.source_col].isnull(), self.target_col] = np.nan
        
        return X


class IntegerDayIndexTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that calculates an integer index for each timestamp 
    relative to the first timestamp in each group, representing the difference in days. 
    This is used to create a consistent time series index for irregularly spaced data.

    :param timestamp_col: The column containing the date information.
    :type timestamp_col: str
    :param unique_id: The column containing the unique patient IDs.
    :type unique_id: str
    :param unit: The unit for time difference, one of 'days', 'seconds', or 'raw'.
    :type unit: str
    """
    
    def __init__(self, timestamp_col, unique_id):
        self.timestamp_col = timestamp_col
        self.unique_id = unique_id

    def fit(self, X, y=None):
        """
        Fit method is used to fit the transformer but does not have any effect for this transformer.

        :param X: Input DataFrame.
        :type X: pd.DataFrame
        :param y: Target values. (unused)
        :type y: Any
        :return: self
        :rtype: IntegerDayIndex
        """
        return self

    def transform(self, X, y=None):
        """
        Transforms the given DataFrame by creating a new column 'day_count' 
        which has an integer index for each timestamp relative to the first timestamp in each group.

        :param X: Input DataFrame.
        :type X: pd.DataFrame
        :param y: Target values. (unused)
        :type y: Any
        :return: Transformed DataFrame.
        :rtype: pd.DataFrame
        """
        X = X.copy()
        
        # Convert the date column to datetime type if it's not already
        X[self.timestamp_col] = pd.to_datetime(X[self.timestamp_col], errors='coerce')
        
        # Compute the difference in days for each timestamp relative to the first timestamp in its group
        X['day_count'] = X.groupby(self.unique_id)[self.timestamp_col].transform(lambda x: (x - x.min()).dt.days if not x.isna().all() else np.nan)
    
        # Filter out groups with only NaN values for timestamp_col and find the index of the first timestamp for the rest
        mask = X.groupby(self.unique_id)[self.timestamp_col].transform(lambda x: not x.isna().all())
        filtered_X = X[mask]
        first_idx = filtered_X.groupby(self.unique_id)[self.timestamp_col].idxmin()
        X.loc[first_idx, 'day_count'] = 0
    
        # Attempt linear interpolation within groups
        X['day_count'] = X.groupby(self.unique_id)['day_count'].transform(lambda x: x.interpolate())
    
        # For NaNs that couldn't be interpolated, compute the mean of nth values from other groups and fill NaNs
        def impute_with_group_means(group):
            for idx, isna in enumerate(group['day_count'].isna()):
                if isna:
                    nth_values = X.groupby(self.unique_id).nth(idx)['day_count'].dropna()
                    group.iloc[idx, group.columns.get_loc('day_count')] = nth_values.mean()
            return group
    
        X = X.groupby(self.unique_id).apply(impute_with_group_means)

        X['day_count'] = X['day_count'].astype(int)
    
        return X

