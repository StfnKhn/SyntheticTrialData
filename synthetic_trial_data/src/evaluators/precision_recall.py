import logging
import numpy as np

from typing import Union, List

from synthetic_trial_data.src.evaluators.base_evaluator import BaseEvaluator
from synthetic_trial_data.src.metrics.precision_recall import knn_precision_recall_features


logger = logging.getLogger(__name__)


class PrecisionRecall(BaseEvaluator):
    """
    Preforms the computation of precision and recall of the real and the synthetic samples
    according to :cite:`[Kynkäänniemi2019]`. This improved version is based on computing non-parametic 
    representations of the manifolds of the real and the synthetic data. From these manifolds the 
    precision and recall can be estimated.

    :param name: Name of the precision_recall instance.
    :type name: str
    :param knn: K nearest neighbors, defaults to [3].
    :type knn: list of int, optional
    :param row_batch_size: Row batch size to compute pairwise distances
        (parameter to trade-off between memory usage and performance),
        defaults to 1000.
    :type row_batch_size: int, optional
    :param col_batch_size: Column batch size to compute pairwise distances
        (parameter to trade-off between memory usage and performance),
        defaults to 1000.
    :type col_batch_size: int, optional

    .. [Kynkäänniemi2019] Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019).
        Improved precision and recall metric for assessing generative models. 
        Advances in Neural Information Processing Systems, 32.
    """

    def __init__(
        self,
        knn: int = 3,
        row_batch_size: int = 1000, 
        col_batch_size: int = 1000
    ):
        super().__init__()
        self.knn = knn
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size

    @staticmethod
    def name() -> str:
        return "precision_recall"
    
    @staticmethod
    def metrics() -> List[str]:
        return ["precision", "recall"]
    
    @staticmethod
    def polarity() -> int:
        return {"precision": 1, "recall": 1}

    def evaluate(
        self, 
        X_real: np.ndarray,
        X_synth: np.ndarray
    ) -> dict:
        """
        Preforms the computation of precision and recall of the real and the synthetic samples

        :param X_real: The original (real) data
        :type X_real: np.array
        :param X_synth: The synthetic data
        :type X_synth: np.array
        :return: A dictionary containing the precision and recall metrics.
        :rtype: dict
        """
        result = knn_precision_recall_features(
            X_real,
            X_synth, 
            nhood_sizes=[self.knn],
            row_batch_size=self.row_batch_size,
            col_batch_size=self.col_batch_size,
            num_gpus=1
        )
        output_dict = {
            "precision": result["precision"][0],
            "recall": result["recall"][0]
        }

        return output_dict