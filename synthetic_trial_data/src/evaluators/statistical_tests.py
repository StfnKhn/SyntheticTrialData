import logging
import numpy as np
import pandas as pd

from typing import Union, List
from sklearn.gaussian_process.kernels import RBF

from synthetic_trial_data.src.evaluators.base_evaluator import BaseEvaluator
from synthetic_trial_data.src.metrics.mmd.mmd_boot import perform_mmd_test, ImplementedMMDSchemes
from synthetic_trial_data.src.metrics.mmd.kernel_wrapper import KernelWrapper
from synthetic_trial_data.src.metrics.mmd.mmd_agg import mmdagg
from synthetic_trial_data.src.metrics.mmd.mmd_3_sample import MMD_3_Sample_Test


logger = logging.getLogger(__name__)


class MMD_BOOT(BaseEvaluator):
    """
    Maximum Mean Discrepancy Bootstrap (MMD_BOOT) class for evaluating if two data
    sets come from the same distribution.

    This class is based on the Maximum Mean Discrepancy (MMD) bootstrap method, 
    which is a statistical test used to determine if two sets of data come from 
    the same distribution.

    :param name: The name of the Evaluator, optional.
    :type name: str, optional
    :param kernel: The kernel used to transform the data for computing MMD, optional.
    :type kernel: KernelWrapper, optional
    :param scheme: The MMD computation scheme to use, optional.
    :type scheme: ImplementedMMDSchemes, optional
    :param test_level: The significance level for the test, defaults to 0.5.
    :type test_level: float, optional

    .. note::
    This class uses the methodology described in :cite:`gretton2012`.

    .. [gretton2012] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. 
    A kernel two-sample test. Journal of Machine Learning Research, 13: 723–773.
    """
    def __init__(
        self,
        kernel: KernelWrapper = None,
        scheme: ImplementedMMDSchemes = ImplementedMMDSchemes.PERMUTATION,
        test_level = 0.5
    ):
        super().__init__()
        self.kernel = kernel
        self.scheme = scheme
        self.test_level = test_level

    @staticmethod
    def name() -> str:
        return "MMD_BOOT"
    
    @staticmethod
    def metrics() -> List[str]:
        return ["Reject H0 (H0 : P_X = P_Y)", "MMD", "Rejection threshold"]

    @staticmethod
    def polarity() -> int:
        return {"p-value": -1}
        
    def evaluate(self, X_real: np.array, X_synth: np.array, **kwargs):
        """
        Perform the MMD-BOOT test between two data sets.

        :param X_real: The original (real) data
        :type X_real: np.array
        :param X_synth: The synthetic data
        :type X_synth: np.array
        :param kwargs: Other optional parameters
        :type kwargs: dict, optional
        :return: A dictionary with results of the MMD-BOOT test containing keys:
                 "Reject H0 (H0 : P_X = P_Y)", "MMD", and "Rejection threshold"
        :rtype: dict
        """

        # Convert to numpy array if input is of type pd.DataFrame
        if isinstance(X_real, pd.DataFrame):
            X_real = X_real.to_numpy()
        if isinstance(X_synth, pd.DataFrame):
            X_synth = X_synth.to_numpy()

        # A priori selection of kernel that is used to transform the data
        # to a space where the MMD is being computed
        if self.kernel == None:
            # Here the initial condition is a Gaussian kernal
            self.kernel = KernelWrapper(
                RBF(length_scale=np.median(np.abs(X_real - X_synth)))
            )

        # Execute MMD_Boot test
        result = perform_mmd_test(
            data_x=X_real,
            data_y=X_synth,
            kernel=self.kernel,
            test_level=self.test_level,
            scheme=self.scheme
        )
        
        return result


class MMD_Agg(BaseEvaluator):
    """
    A class that wraps around an aggregated version of the Maximum Mean Discrepancy (MMD) test
    that is referred to as MMDAgg. For more details regarding the methodology see :cite:`gretton2021`.

    :param name: The name of the MMD_Agg instance, defaults to "MMD_Agg_test".
    :type name: str, optional
    :param kernel: The kernel to be used for MMD tests, defaults to "gaussian".
    :type kernel: str, optional
        
    .. [gretton2021] Schrab, A., Kim, I., Albert, M., Laurent, B., Guedj, B., & Gretton, A. (2021). 
        MMD aggregated two-sample test. arXiv preprint arXiv:2110.15073.
    """
    def __init__(self, kernel: str = "gaussian"):
        super().__init__()
        self.kernel = kernel

    @staticmethod
    def name() -> str:
        return "MMDAgg-test"
    
    @staticmethod
    def metrics() -> List[str]:
        return [
                "MMDAgg: Reject H0 (H0 : P_X_train = P_Y)",
                "MMDAgg: Reject H0 (H0 : P_X_test = P_Y)",
                "MMDAgg: p-value (X_train)",
                "MMDAgg: p-value (X_test)"
        ]

    @staticmethod
    def polarity() -> int:
        return {
                "MMDAgg: Reject H0 (H0 : P_X_train = P_Y)": -1,
                "MMDAgg: Reject H0 (H0 : P_X_test = P_Y)": -1,
                "MMDAgg: p-value (X_train)": 1,
                "MMDAgg: p-value (X_test)": 1
        }

    def _compute_mean_of_single_test_metric(self, result_dict: dict, metric_name: str):
        """
        Compute the mean of a specific metric from the results of MMD tests.

        :param result_dict: A dictionary containing the results of MMD tests.
        :type result_dict: dict
        :param metric_name: The name of the metric to compute the mean for.
        :type metric_name: str
        :return: The computed mean.
        :rtype: float
        """
        metric_values = []

        for key, sub_dict in result_dict.items():
            if isinstance(sub_dict, dict) and metric_name in sub_dict:
                metric_values.append(sub_dict[metric_name])
        
        return np.mean(np.array(metric_values))

    def _evaluate_single(self, X_real: np.array, X_synth: np.array):
        # Execute mmdagg test
        large_return_dict = mmdagg(X=X_real, Y=X_synth, return_dictionary=True, kernel=self.kernel)
        reject = large_return_dict[1]['MMDAgg test reject']

        # Aggregate p-values of the individual single tests
        p_value_aggregated = self._compute_mean_of_single_test_metric(large_return_dict[1], "p-value")

        return p_value_aggregated, reject

    def evaluate(self, X_real: np.array, X_synth: np.array, X_real_val: np.array, **kwargs):
        """
        Evaluate the similarity between the real and synthetic data using MMD tests.

        :param X_real: The real data.
        :type X_real: np.array
        :param X_synth: The synthetic data.
        :type X_synth: np.array
        :param kwargs: Additional keyword arguments.
        :type kwargs: dict, optional
        :return: A dictionary containing the results of the MMD tests.
        :rtype: dict
        """
        # Convert to numpy array if input is of type pd.DataFrame
        if isinstance(X_real, pd.DataFrame):
            X_real = X_real.to_numpy()
        if isinstance(X_real_val, pd.DataFrame):
            X_real_val = X_real_val.to_numpy()
        if isinstance(X_synth, pd.DataFrame):
            X_synth = X_synth.to_numpy()
        
        p_value_train, reject_train = self._evaluate_single(X_real=X_real, X_synth=X_synth)
        p_value_val, reject_val = self._evaluate_single(X_real=X_real_val, X_synth=X_synth)

        # Define result dict
        result = {}

        result["MMDAgg: Reject H0 (H0 : P_X_train = P_Y)"] = reject_train
        result["MMDAgg: Reject H0 (H0 : P_X_test = P_Y)"] = reject_val
        result["MMDAgg: p-value (X_train)"] = p_value_train
        result["MMDAgg: p-value (X_test)"] = p_value_val

        return result


class MMD_3_Sample(BaseEvaluator):
    """
    Performs the relative MMD test which returns a test statistic for whether Y is closer to X or than Z.
    As the method is described in:cite:`bounliphone2015` it operates with an rbf_kernel per default.

    :param name: The name of the MMD_3_Sample instance.
    :type name: str
    :param sigma: If None then the bandwith heuristic :cite:`gretton2012` is being used to select
        the sigma for the gaussian kernel
    :type sigma: float, optional
    :param computeMMDs: Flag indicating whether to compute MMDs or not, defaults to True.
    :type computeMMDs: bool, optional

    .. [bounliphone2015] Bounliphone, W., Belilovsky, E., Blaschko, M. B., Antonoglou, I., & Gretton, A. (2015). 
        A test of relative similarity for model selection in generative models. 
        arXiv preprint arXiv:1511.04581.

    .. [gretton2012] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. 
    A kernel two-sample test. Journal of Machine Learning Research, 13: 723–773.
    """

    def __init__(
        self,
        name: str,
        sigma: float = None,
        computeMMDs: bool = True
    ):
        super().__init__()
        self.sigma = sigma
        self.computeMMDs = computeMMDs

    def evaluate(
        self, 
        X_real: np.ndarray,
        X_synth_1: np.ndarray,
        X_synth_2: np.ndarray
    ) -> dict:
        """
        """
        p_value, tstat, sigma, mmd_XY, mmd_XZ = MMD_3_Sample_Test(
            X=X_real,
            Y=X_synth_1,
            Z=X_synth_2,
            sigma=self.sigma,
            computeMMDs=self.computeMMDs
        )

        result = {}
        result["p-value"] = p_value
        result["MMD(X_real, X_synth_1)"] = mmd_XY
        result["MMD(X_real, X_synth_2)"] = mmd_XZ
        
        return result
