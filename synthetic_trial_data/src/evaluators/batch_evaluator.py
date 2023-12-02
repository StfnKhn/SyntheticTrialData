import logging
import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

from synthetic_trial_data.src.preprocessing.preprocessors import Encoder
from synthetic_trial_data.src.evaluators.precision_recall import PrecisionRecall
from synthetic_trial_data.src.evaluators.privacy import Identifiability, AttributeDisclosureRisk, DomiasKDE, DomiasBNAF
from synthetic_trial_data.src.evaluators.statistical_tests import MMD_Agg
from synthetic_trial_data.src.evaluators.temporal import CrossCorrelationDifference, AutoCorrelationDifference
from synthetic_trial_data.src.evaluators.distance_metrics import (
    WassersteinDistance,
    JensenShannonDistance,
    MaximumMeanDiscrepancy,
    KLDivergence
)

EVALUATORS = [
    PrecisionRecall,
    Identifiability,
    AttributeDisclosureRisk,
    DomiasKDE,
    DomiasBNAF,
    WassersteinDistance,
    JensenShannonDistance,
    MaximumMeanDiscrepancy,
    KLDivergence,
    MMD_Agg,
    CrossCorrelationDifference,
    AutoCorrelationDifference
]

class BaseBatchEvaluator(ABC):
    def __init__(
        self,
        X_real: pd.DataFrame,
        X_real_val: pd.DataFrame,
        X_synth_data: Union[pd.DataFrame, List[pd.DataFrame]],
        evaluators: List[str],
        QID: List[str] = None,
        S: List[str] = None,
        reference_size: int = None,
        unique_id_cols: List[str] = None, 
        cross_corr_col_pairs: List[Tuple[str]] = None,
        event_columns: List[str] = None
    ):
        self.X_real = X_real
        self.X_real_val = X_real_val
        self.X_synth_data = X_synth_data
        self.evaluators = evaluators
        self.evaluator_objects = EVALUATORS

        # Define optional input data
        self.QID = QID
        self.S = S
        self.reference_size = reference_size
        self.unique_id_cols = unique_id_cols
        self.cross_corr_col_pairs = cross_corr_col_pairs
        self.event_columns = event_columns

        # Initialize metric names
        self.jsd = "JSD"
        self.wasserstein_distance = "Wasserstein Distance"
        self.mmd = "MMD"
        self.kl_div = "KL-Divergence"
        self.pr = "precision_recall"
        self.identifiability = "epsilon-identifiability"
        self.attr_disclosure_risk = "Attribute Disclosure Risk"
        self.cross_corr_difference = "CrossCorrelationDifference"
        self.auto_corr_difference = "AutoCorrelationDifference"
        self.domias = "DomiasKDE"
        self.domiasBNAF = "domiasBNAF"
        self.MMDAgg = "MMDAgg-test"

        # Handle both single DataFrame or List of DataFrames
        if isinstance(X_synth_data, pd.DataFrame):
            self.X_synth_list = [X_synth_data]
        elif isinstance(X_synth_data, list):
            self.X_synth_list = X_synth_data
        else:
            raise ValueError("X_synth_data must be either a DataFrame or a list of DataFrames")

        # Check if given evaluators are supported
        self._check_if_evaluators_are_supported()

        # Initialize metrics dict
        self.metrics = self._get_metrics_for_given_evaluators()
        self.metrics_dict = {metric: [] for metric in self.metrics}

        self.polarity = self._get_polarity_dict()

        # Fit encoder (Some metrics need to work with encoded data)
        self.encoder = Encoder(cat_encoder="ohe", scale_numeric=True)
        self.X_real_encoded = self.encoder.fit_transform(self.X_real)
        self.X_real_val_encoded = self.encoder.transform(self.X_real_val)

    def _available_evaluators(self):
        return [evaluator.name() for evaluator in self.evaluator_objects]

    def _get_evaluator_by_name(self, name: str):
        for evaluator in self.evaluator_objects:
            if evaluator.name() == name:
                return evaluator
        raise ValueError(f"No evaluator found with the name: {name}. Only use supported metrics: {self._available_evaluators}")

    def _check_if_evaluators_are_supported(self):
        for evaluator in self.evaluators:
            self._get_evaluator_by_name(evaluator)

    def _get_metrics_for_given_evaluators(self):
        """
        Required because some evaluators output multiple metrics
        """
        metrics = []
        for evaluator in self.evaluator_objects:
            if evaluator.name() in self.evaluators:
                metrics.extend(evaluator.metrics())
        return metrics

    def _get_polarity_dict(self):
        polarity = {}
        for name in self.evaluators:
            obj = self._get_evaluator_by_name(name)
            polarity.update(obj.polarity())
        return polarity

    def _evaluate_single(self, evaluator, X_synth):
        """
        Evaluate a single metric for a single synthetic dataset.

        :param evaluator: Name of the evaluator specifying the metric.
        :type evaluator: str
        :param X_synth: The synthetic dataset for evaluation.
        :type X_synth: pd.DataFrame
        :return: Dictionary containing evaluation results for the specified metric.
        :rtype: dict
        """
        # Encode Synthetic data        
        X_synth_encoded = self.encoder.transform(X_synth)

        # Get evaluator object
        if evaluator == self.MMDAgg:
            X_synth_encoded = self.encoder.transform(X_synth)
            output_dict = MMD_Agg().evaluate(
                X_real=self.X_real_encoded,
                X_synth=X_synth_encoded,
                X_real_val=self.X_real_val_encoded
            )
            return output_dict
        elif evaluator == self.cross_corr_difference:
            output_dict = CrossCorrelationDifference().evaluate(
                self.X_real,
                X_synth,
                self.unique_id_cols,
                self.cross_corr_col_pairs
            )
            return output_dict
        elif evaluator == self.auto_corr_difference:
            output_dict = AutoCorrelationDifference().evaluate(
                self.X_real,
                X_synth,
                self.unique_id_cols,
                self.event_columns
            )
            return output_dict
        elif evaluator == self.identifiability:
            output_dict = Identifiability().evaluate(self.X_real, X_synth)
            return output_dict
        elif evaluator == self.attr_disclosure_risk :
            output_dict = AttributeDisclosureRisk(QID=self.QID, S=self.S).evaluate(self.X_real, X_synth)
            return output_dict
        elif evaluator == self.domias:
            output_dict = DomiasKDE().evaluate(
                X_real_val=self.X_real_val,
                X_synth=X_synth,
                X_real_train=self.X_real,
                reference_size=self.reference_size
            )
            return output_dict
        elif evaluator == self.domiasBNAF:
            X_synth_train, X_synth_val = train_test_split(X_synth, test_size=0.1)
            output_dict = DomiasBNAF().evaluate(
                X_real_train=self.X_real,
                X_real_val=self.X_real_val,
                X_synth=X_synth_train,
                X_synth_val=X_synth_val,
                reference_size=self.reference_size
            )
            return output_dict
        
        # Compute standard evaluators that use encoded data
        evaluator_object = self._get_evaluator_by_name(evaluator)()
        output_dict = evaluator_object.evaluate(self.X_real_encoded, X_synth_encoded)

        return output_dict

class BatchEvaluator(BaseBatchEvaluator):
    def __init__(
        self,
        X_real: pd.DataFrame,
        X_real_val: pd.DataFrame,
        X_synth_list: List[pd.DataFrame],
        evaluators: List[str],
        QID: List[str] = None,
        S: List[str] = None,
        reference_size: int = None,
        unique_id_cols: List[str] = None, 
        cross_corr_col_pairs: List[Tuple[str]] = None,
        event_columns: List[str] = None
    ):
        super().__init__(
            X_real=X_real,
            X_real_val=X_real_val,
            X_synth_data=X_synth_list,
            evaluators=evaluators,
            QID=QID,
            S=S,
            reference_size=reference_size,
            unique_id_cols=unique_id_cols,
            cross_corr_col_pairs=cross_corr_col_pairs,
            event_columns=event_columns
        )
    
    def evaluate(self):
        """
        Evaluate the specified metrics for all synthetic datasets in the list.

        :return: Dictionary with metrics as keys and lists of evaluation results as values.
        :rtype: dict
        """
        for evaluator in self.evaluators:
            for X_synth in self.X_synth_data:
                results = self._evaluate_single(evaluator, X_synth)
                for key, value in results.items():
                    self.metrics_dict[key].append(value)
           
        return self.metrics_dict
    

class SingleSetBatchEvaluator(BaseBatchEvaluator):
    def __init__(
        self,
        X_real: pd.DataFrame,
        X_real_val: pd.DataFrame,
        X_synth: List[pd.DataFrame],
        evaluators: List[str],
        QID: List[str] = None,
        S: List[str] = None,
        reference_size: int = None,
        unique_id_cols: List[str] = None, 
        cross_corr_col_pairs: List[Tuple[str]] = None,
        event_columns: List[str] = None
    ):  
        super().__init__(
            X_real=X_real,
            X_real_val=X_real_val,
            X_synth_data=X_synth,
            evaluators=evaluators,
            QID=QID,
            S=S,
            reference_size=reference_size,
            unique_id_cols=unique_id_cols,
            cross_corr_col_pairs=cross_corr_col_pairs,
            event_columns=event_columns
        )
    
    def evaluate(self):
        """
        Evaluate the specified metrics for all synthetic datasets in the list.

        :return: Dictionary with metrics as keys and lists of evaluation results as values.
        :rtype: dict
        """
        for evaluator in self.evaluators:
            results = self._evaluate_single(evaluator, self.X_synth_data)
            for key, value in results.items():
                self.metrics_dict[key] = value
           
        return self.metrics_dict