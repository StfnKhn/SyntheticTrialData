from typing import Union, List
import numpy as np
import pandas as pd
from synthcity.metrics.eval import Metrics

from synthetic_trial_data.src.evaluators.base_evaluator import BaseEvaluator
from synthetic_trial_data.src.utils.dataframe_handling import to_dataframe


class KLDivergence(BaseEvaluator):
    """
    Computes the Kullback-Leibler divergence (KL-divergence) between two probability mass functions (PMFs). 
    The KL-divergence measures the discrepancy between the PMFs and can be used to compare the distribution 
    of the synthetic data and original data. A zero KL-divergence signifies perfect identity between the two 
    PMFs, while larger values reflect greater differences.

    .. math::
        D_{KL}^{v}(P||Q) = \sum_{i=1}^{v} P(i) \log \left( \frac{P(i)}{Q(i)} \right)

    :param name: The name of the Evaluator, optional.
    :type name: str, optional
    """

    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "KL-Divergence"
    
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
        **kwargs
    ):
        """
        Computes the Kullback-Leibler divergence score

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
        result_df = Metrics.evaluate(
            X_gt=X_real,
            X_syn=X_synth,
            metrics={'stats': ["inv_kl_divergence"]}
        )

        result = {
            __class__.name(): 1/result_df["min"][0] - 1
        }
        
        return result


class JensenShannonDistance(BaseEvaluator):
    """
    Computes the Jensen-Shannon distance between two probability distributions :math:`P` and :math:`Q`. The JSD is 
    defined as the square root of the Jensen-Shannon divergence and represents a symmetric measurement of the 
    similarity of two probability distributions :math:`P` and :math:`Q`. In this context, :math:`DKL` is the 
    KL-Divergence and :math:`M` is the average distribution of :math:`P` and :math:`Q` defined as 
    :math:`M = \\frac{1}{2} (P + Q)`. 
    The JSD is given by:
    
    .. math::
        JSD(P||Q) = \sqrt{\\frac{1}{2}DKL(P||M) + \\frac{1}{2}DKL(Q||M)}

    :param name: The name of the Evaluator, optional.
    :type name: str, optional
    """
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "JSD"
    
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
        **kwargs
    ):
        """
        Computes the Jensen-Shannon distance score

        :param X_real: The original (real) data
        :type X_real: np.array
        :param X_synth: The synthetic data
        :type X_synth: np.array
        :param kwargs: Other optional parameters
        :type kwargs: dict, optional
        """

        # If not already convert data to type pd.DataFrame
        X_real = to_dataframe(X_real)
        X_synth = to_dataframe(X_synth)

        # Compute metric
        result_df = Metrics.evaluate(
            X_gt=X_real,
            X_syn=X_synth,
            metrics={'stats': ["jensenshannon_dist"]}
        )

        result = {
            __class__.name(): result_df["min"][0]
        }
        
        return result


class WassersteinDistance(BaseEvaluator):
    """
    Computes the Wasserstein distance between two probability distributions :math:`r` and :math:`s`. 
    The Wasserstein distance quantifies the minimum probability mass that needs to be moved to reshape 
    one distribution into the other. It represents the Wasserstein distance between the probability 
    distributions of the real and the synthetic data. While extensively used as a loss function in 
    Wasserstein GAN (WGAN), it's also a valuable metric for assessing the resemblance of synthetic tabular data.

    .. math::
        W(r,s) = \int_{-\infty}^{\infty} |R - S|

    :param name: The name of the Evaluator, optional.
    :type name: str, optional
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "Wasserstein Distance"
    
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
        **kwargs
    ):
        """
        Computes the Wasserstein distance score

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
        result_df = Metrics.evaluate(
            X_gt=X_real,
            X_syn=X_synth,
            metrics={'stats': ["wasserstein_dist"]}
        )

        result = {
            __class__.name(): result_df["min"][0]
        }
        
        return result


class MaximumMeanDiscrepancy(BaseEvaluator):
    """
    Maximum Mean Discrepancy (MMD) measures the difference between two distributions with respect to 
    the unit ball of a reproducing kernel Hilbert space (RKHS) H. It requires choosing a kernel beforehand,
    and the selection can significantly impact the test results.

    .. math::
        MMD^2 = E[h_{P,Q}(X)] - E[h_{P,Q}(Y)]

    :param name: The name of the Evaluator, optional.
    :type name: str, optional
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "MMD"
    
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
        **kwargs
    ):
        """
        Computes the Maximum Mean Discrepancy score

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
        result_df = Metrics.evaluate(
            X_gt=X_real,
            X_syn=X_synth,
            metrics={'stats': ["max_mean_discrepancy"]}
        )

        result = {
            __class__.name(): result_df["min"][0]
        }
        
        return result

