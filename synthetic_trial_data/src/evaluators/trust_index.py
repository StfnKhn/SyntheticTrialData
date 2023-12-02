from typing import Union, List
import pandas as pd
import numpy as np
from math import exp, log
from statsmodels.distributions.empirical_distribution import ECDF

from synthetic_trial_data.src.evaluators.batch_evaluator import SingleSetBatchEvaluator

class TrustDimension:
    """
    Class for evaluating and aggregating metrics corresponding to a specific trust dimension.

    :param name: Name of the trust dimension.
    :type name: str
    :param evaluators: List of evaluator names specifying metrics for evaluation.
    :type evaluators: List[str]

    Trust dimensions aim to assess quantitatively various risks associated with synthetic datasets,
    relative to real datasets. Metrics under a trust dimension measure different aspects and have
    different monotonicity. This class normalizes the monotonicity such that higher metric values
    indicate better conformity with the trust dimension.

    Evaluations are performed on various metrics like Fidelity, Privacy, Utility, Fairness, and Robustness.
    The goodness of conformity of each metric is increased in its numerical values by adjusting its polarity.

    A detailded explanation of the implementaion can be found in :cite:`[Belgodere2023]`

    .. [Belgodere2023] Belgodere, B., Dognin, P., Ivankay, A., Melnyk, I., Mroueh, Y., Mojsilovic, A., ... & Young, R. A. (2023).
        Auditing and Generating Synthetic Data with Controllable Trust Trade-offs. arXiv preprint arXiv:2304.10819.
    """
    def __init__(self, name: str, evaluators: List[str]):
        self.name = name
        self.metrics = []
        self.evaluators = evaluators
        self._is_fitted = False
    
    def _check_fitted(self):
        """Check if the dimension is fitted."""
        if not self._is_fitted:
            raise ValueError("TrustDimension is not fitted. Please call fit() before using this method.")
    
    def evaluate_metrics(
        self, 
        X_real,
        X_real_val,
        X_synth,
        QID=None,
        S=None,
        reference_size=None,
        unique_id_cols=None,
        cross_corr_col_pairs=None,
        event_columns=None
    ):
        """
        Evaluate metrics for a single synthetic dataset against the real dataset.

        :param X_real: The real-world DataFrame.
        :type X_real: pd.DataFrame
        :param X_real_val: Validation DataFrame based on the real dataset.
        :type X_real_val: pd.DataFrame
        :param X_synth: The synthetic dataset for evaluation.
        :type X_synth: pd.DataFrame
        :param QID: Quasi-Identifier columns for some evaluators. Only relevant for computation
            of the `Attrubute Disclosure Risk`
        :type QID: List[str]
        :param S: Sensitive attribute for some evaluators. Only relevant for computation
            of the `Attrubute Disclosure Risk`
        :type S: str
        :param reference_size: Size of the reference sample for some evaluators. Only relevant for computation
            of DomiasKDE and Domias BNAF.
        :type reference_size: int
        :return: Aligned metric values after adjusting for polarity.
        :rtype: pd.Series
        """
        # Initialization of Evaluator that computes all selected evaluators in a batch
        evaluator = SingleSetBatchEvaluator(
            X_real=X_real,
            X_real_val=X_real_val,
            X_synth=X_synth,
            evaluators=self.evaluators,
            QID=QID,
            S=S,
            reference_size=reference_size,
            unique_id_cols=unique_id_cols,
            cross_corr_col_pairs=cross_corr_col_pairs,
            event_columns=event_columns
        )
        self.metrics = evaluator.metrics

        # Perform evaluations
        metrics_dict = evaluator.evaluate()

        # Compute aligned value
        polarity_dict = evaluator.polarity
        metrics_df = pd.DataFrame({'metric_value': metrics_dict, 'polarity': polarity_dict})
        metrics_df["aligned_value"] = metrics_df["metric_value"] * metrics_df["polarity"]

        return metrics_df["aligned_value"]
    
    def fit(
        self,
        X_real,
        X_real_val,
        X_synth_reference,
        QID,
        S,
        reference_size,
        unique_id_cols,
        cross_corr_col_pairs,
        event_columns
    ):
        reference_values = {
            X_synth_ref.id: self.evaluate_metrics(
                X_real,
                X_real_val,
                X_synth_ref,
                QID,
                S,
                reference_size,
                unique_id_cols,
                cross_corr_col_pairs,
                event_columns
            )
            for X_synth_ref in X_synth_reference
        }        
        # ECDF Evaluation and Score Normalization
        self.ecdf_dict = {}
        self.reference_values_per_metric = {}
        self.normed_metrics = {}
        for i, metric in enumerate(self.metrics):
            self.reference_values_per_metric[metric] = [ref_values[metric] for ref_values in reference_values.values()]
            ecdf = ECDF(self.reference_values_per_metric[metric])

            # For the purpose of plotting the results
            self.ecdf_dict[metric] = ecdf
            self.normed_metrics[metric] = [ecdf(ref_values[metric]) for ref_values in reference_values.values()]
        
        # Set the flag after fitting
        self._is_fitted = True 
        
    def aggregate(
        self,
        X_real,
        X_real_val,
        X_synth,
        QID,
        S,
        reference_size,
        unique_id_cols,
        cross_corr_col_pairs,
        event_columns,
    ):
        """
        Aggregate metric values to produce a trust dimension index using the Copula Method.

        For a trust dimension T, this function aims to aggregate the aligned metrics which are evaluated on 
        a synthetic dataset Ds against a real dataset Dr, effectively producing an index for the trust dimension T.

        Challenges:
        The metrics might have different dynamic ranges which can make a simple mean aggregation incorrect 
        as it could favor metrics with larger dynamic ranges. This function resorts to the copula aggregation method to tackle this.

        Steps:
        1. ECDF Evaluation: For each aligned metric, compute the Empirical Cumulative Distribution Function (ECDF) given 
        observations across multiple synthetic datasets.
        2. Score Normalization: Use ECDFs to map each metric's distribution to an approximately uniform distribution. 
        This normalized score indicates the quality of synthetic data with respect to the metric.
        3. Compute the Copula/Trust Dimension Index: Conformity of synthetic data to a trust dimension is aggregated 
        using normalized scores of all metrics. The aggregation uses a geometric mean copula. If there's a known priority 
        for certain metrics over others, it can be integrated through weights that reflect each metric's relative importance.

        :param X_real: The real-world DataFrame.
        :type X_real: pd.DataFrame
        :param X_real_val: Validation DataFrame based on the real dataset.
        :type X_real_val: pd.DataFrame
        :param X_synth: The synthetic dataset for evaluation.
        :type X_synth: pd.DataFrame
        :param QID: Quasi-Identifier columns for some evaluators. Only relevant for computation
            of the `Attrubute Disclosure Risk`
        :type QID: List[str]
        :param S: Sensitive attribute for some evaluators. Only relevant for computation
            of the `Attrubute Disclosure Risk`
        :type S: str
        :param reference_size: Size of the reference sample for some evaluators. Only relevant for computation
            of DomiasKDE and Domias BNAF.
        :type reference_size: int
        :return: Trust Dimension Index for the synthetic data.
        :rtype: float
        """
        self._check_fitted() 
        metric_values = self.evaluate_metrics(
            X_real,
            X_real_val,
            X_synth,
            QID,
            S,
            reference_size,
            unique_id_cols,
            cross_corr_col_pairs,
            event_columns
        )
     
        # ECDF Evaluation and Score Normalization
        u_values = []
        for i, metric in enumerate(self.metrics):
            ecdf = self.ecdf_dict[metric]
            u = ecdf(metric_values[metric])
            u_values.append(u)
        
        # Remove Zeros to avoid math domain error when computing log(0)
        u_values = [np.finfo(float).eps if u==0 else u for u in u_values] 
        
        # Compute Trust Dimension Index (using geometric mean)
        # Here we are assuming uniform weights (β weights). You can add specifics if required.
        trust_dimension_index = exp(sum(log(u) for u in u_values) / len(u_values))

        return trust_dimension_index


class TrustIndexAuditor:
    """
    Class for auditing synthetic datasets to produce a comprehensive Trust Index, introduced by :cite:`[Belgodere2023]`,
    that supports a mechanism to define trade-offs between trust dimensions by adjusting weights.
    A higher weight indicates a higher priority or importance for the associated trust dimension.

    Trust Index is derived from multiple Trust Dimensions. Each dimension's metrics are aggregated to
    yield a single score per dimension. Trust Index combines these scores using given weights.

    :param X_real: The real-world DataFrame.
    :type X_real: pd.DataFrame
    :param X_real_val: Validation DataFrame based on the real dataset.
    :type X_real_val: pd.DataFrame
    :param X_synth: The synthetic dataset for evaluation.
    :type X_synth: pd.DataFrame
    :param QID: Quasi-Identifier columns for some evaluators. Only relevant for computation
        of the `Attrubute Disclosure Risk`
    :type QID: List[str]
    :param S: Sensitive attribute for some evaluators. Only relevant for computation
        of the `Attrubute Disclosure Risk`
    :type S: str
    :param reference_size: Size of the reference sample for some evaluators. Only relevant for computation
        of DomiasKDE and Domias BNAF.
    :type reference_size: int

    .. [Belgodere2023] Belgodere, B., Dognin, P., Ivankay, A., Melnyk, I., Mroueh, Y., Mojsilovic, A., ... & Young, R. A. (2023).
        Auditing and Generating Synthetic Data with Controllable Trust Trade-offs. arXiv preprint arXiv:2304.10819.
    """

    def __init__(
        self,
        X_real,
        X_real_val,
        QID=None,
        S=None,
        reference_size=None,
        unique_id_cols=None,
        cross_corr_col_pairs=None,
        event_columns=None
    ):
        self.X_real = X_real
        self.X_real_val = X_real_val
        self.QID = QID
        self.S = S
        self.reference_size = reference_size
        self.unique_id_cols = unique_id_cols
        self.cross_corr_col_pairs = cross_corr_col_pairs
        self.event_columns = event_columns
        self._is_fitted = False

        self.dimensions = []
        self.weights = []
    
    def _check_fitted(self):
        """Check if the dimension is fitted."""
        if not self._is_fitted:
            raise ValueError("TrustIndexAuditor is not fitted. Please call fit() before using this method.")
        
    def add_dimension(self, dimension: TrustDimension, weight: float):
        """
        Add a trust dimension to the auditor and specify its weight.

        :param dimension: TrustDimension instance representing a trust dimension.
        :type dimension: TrustDimension
        :param weight: Weight for the trust dimension indicating its importance in the final Trust Index.
        :type weight: float
        """
        self.dimensions.append(dimension)
        self.weights.append(weight)
    
    def fit(self, X_synth_list):

        for dimension in self.dimensions:
            dimension.fit(
                self.X_real,
                self.X_real_val,
                X_synth_list,
                QID=self.QID,
                S=self.S,
                reference_size=self.reference_size,
                unique_id_cols=self.unique_id_cols,
                cross_corr_col_pairs=self.cross_corr_col_pairs,
                event_columns=self.event_columns
            )
        
        # Set the flag after fitting
        self._is_fitted = True 
    
    def compute_trust_index(self, X_synth):
        """
        Compute the Trust Index for the synthetic dataset based on added trust dimensions and their weights.

        :return: Dictionary containing the Trust Index and scores for each individual trust dimension.
        :rtype: dict
        """
        self._check_fitted() 

        aggregated_scores = []
        for dimension in self.dimensions:
            aggregated_scores.append(
                dimension.aggregate(
                    self.X_real,
                    self.X_real_val,
                    X_synth,
                    QID=self.QID,
                    S=self.S,
                    reference_size=self.reference_size,
                    unique_id_cols=self.unique_id_cols,
                    cross_corr_col_pairs=self.cross_corr_col_pairs,
                    event_columns=self.event_columns
                )
            )
        trust_index = exp(sum(self.weights[i] * log(score) for i, score in enumerate(aggregated_scores)))

        output_dict = {
            "trust_index": trust_index
        }
        output_dict.update({self.dimensions[i].name: score for i, score in enumerate(aggregated_scores)})
        return output_dict
