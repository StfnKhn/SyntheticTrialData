"""
Copyright [2023] [Qian, ; Cebere, Bogdan-Constanti; van der Schaar, Mihaela]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications
-------------
* The code was modified in a way that the scipy kernel density estimation was replaced
with the method used by sklearn. Using sklearns KernelDensity is more stable.

* Utils are being imported locally
"""

# stdlib
import platform
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, Tuple, Union, Optional, Callable

# third party
import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from scipy import stats
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, KernelDensity

# synthcity absolute
import synthcity.logger as log
from synthcity.metrics import _utils
from synthcity.plugins.core.dataloader import DataLoader
from synthcity.utils.constants import DEVICE
from synthcity.utils.serialization import load_from_file, save_to_file



class DomiasMIA():
    """
    .. inheritance-diagram:: synthcity.metrics.eval_privacy.domias
        :parts: 1

    DOMIAS is a membership inference attacker model against synthetic data, that incorporates
    density estimation to detect generative model overfitting. That is it uses local overfitting to
    detect whether a data point was used to train the generative model or not.

    Returns:
    A dictionary with a key for each of the `synthetic_sizes` values.
    For each `synthetic_sizes` value, the dictionary contains the keys:
        * `MIA_performance` : accuracy and AUCROC for each attack
        * `MIA_scores`: output scores for each attack

    Reference: Boris van Breugel, Hao Sun, Zhaozhi Qian,  Mihaela van der Schaar, AISTATS 2023.
    DOMIAS: Membership Inference Attacks against Synthetic Data through Overfitting Detection.

    """

    def __init__(self, **kwargs: Any) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "DomiasMIA"

    @staticmethod
    def direction() -> str:
        return "minimize"

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_default(
        self,
        X_gt: DataLoader,
        X_syn: DataLoader,
        X_train: DataLoader,
        X_ref_syn: DataLoader,
        reference_size: int,
    ) -> float:
        return self.evaluate(
            X_gt,
            X_syn,
            X_train,
            X_ref_syn,
            reference_size=reference_size,
        )[self._default_metric]

    @abstractmethod
    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Any:
        ...

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
        X_gt: Union[
            DataLoader, Any
        ],  # TODO: X_gt needs to be big enough that it can be split into non_mem_set and also ref_set
        synth_set: Union[DataLoader, Any],
        X_train: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_size: int = 100,  # look at default sizes
        device: Any = DEVICE,
    ) -> Dict:
        """
        Evaluate various Membership Inference Attacks, using the `generator` and the `dataset`.
        The provided generator must not be fitted.

        Args:
            generator: GeneratorInterface
                Generator with the `fit` and `generate` methods. The generator MUST not be fitted.
            X_gt: Union[DataLoader, Any]
                The evaluation dataset, used to derive the training and test datasets.
            synth_set: Union[DataLoader, Any]
                The synthetic dataset.
            X_train: Union[DataLoader, Any]
                The dataset used to create the mem_set.
            synth_val_set: Union[DataLoader, Any]
                The dataset used to calculate the density of the synthetic data
            reference_size: int
                The size of the reference dataset
            device: PyTorch device
                CPU or CUDA

        Returns:
            A dictionary with the AUCROC and accuracy scores for the attack.
        """

        mem_set = X_train
        non_mem_set, reference_set = (
            X_gt.to_numpy()[:reference_size],
            X_gt.to_numpy()[-reference_size:],
        )

        all_real_data = np.concatenate((X_train.to_numpy(), X_gt.to_numpy()), axis=0)

        continuous = []
        for i in np.arange(all_real_data.shape[1]):
            if len(np.unique(all_real_data[:, i])) < 10:
                continuous.append(0)
            else:
                continuous.append(1)

        self.norm = _utils.normal_func_feat(all_real_data, continuous)

        """ 3. Synthesis with the GeneratorInferface"""

        # get real test sets of members and non members
        X_test = np.concatenate([mem_set, non_mem_set])
        Y_test = np.concatenate(
            [np.ones(mem_set.shape[0]), np.zeros(non_mem_set.shape[0])]
        ).astype(bool)

        """ 4. density estimation / evaluation of Eqn.(1) & Eqn.(2)"""
        # First, estimate density of synthetic data then
        # eqn2: \prop P_G(x_i)/P_X(x_i)
        # p_R estimation
        p_G_evaluated, p_R_evaluated = self.evaluate_p_R(
            synth_set, synth_val_set, reference_set, X_test, device
        )

        p_rel = p_G_evaluated / (p_R_evaluated + 1e-10)

        acc, auc = _utils.compute_metrics_baseline(p_rel, Y_test)
        return {
            "accuracy": acc,
            "aucroc": auc,
        }


class DomiasMIAPrior(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_prior"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        p_G_evaluated = density_gen(X_test.transpose(1, 0))
        p_R_evaluated = self.norm.pdf(X_test)
        return p_G_evaluated, p_R_evaluated


class DomiasMIAKDE(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_KDE"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if synth_set.shape[0] > X_test.shape[0]:
            log.debug(
                """
The data appears to lie in a lower-dimensional subspace of the space in which it is expressed.
This has resulted in a singular data covariance matrix, which cannot be treated using the algorithms
implemented in `gaussian_kde`. If you wish to use the density estimator `kde` or `prior`, consider performing principle component analysis / dimensionality reduction
and using `gaussian_kde` with the transformed data. Else consider using `bnaf` as the density estimator.
                """
            )

        # density_gen = stats.gaussian_kde(synth_set.values.transpose(1, 0))
        # density_data = stats.gaussian_kde(reference_set.transpose(1, 0))
        # p_G_evaluated = density_gen(X_test.transpose(1, 0))
        # p_R_evaluated = density_data(X_test.transpose(1, 0))
        density_gen = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(
            synth_set.values
        )
        density_data = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(
            reference_set
        )
        p_G_evaluated = density_gen.score_samples(X_test)
        p_R_evaluated = density_data.score_samples(X_test)
        return p_G_evaluated, p_R_evaluated


class DomiasMIABNAF(DomiasMIA):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def name() -> str:
        return "DomiasMIA_BNAF"

    def evaluate_p_R(
        self,
        synth_set: Union[DataLoader, Any],
        synth_val_set: Union[DataLoader, Any],
        reference_set: np.ndarray,
        X_test: np.ndarray,
        device: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _, p_G_model = _utils.density_estimator_trainer(
            synth_set.values,
            synth_val_set.values[: int(0.5 * synth_val_set.shape[0])],
            synth_val_set.values[int(0.5 * synth_val_set.shape[0]) :],
            epochs=20
        )
        _, p_R_model = _utils.density_estimator_trainer(reference_set, epochs=2)
        p_G_evaluated = np.exp(
            _utils.compute_log_p_x(
                p_G_model, torch.as_tensor(X_test).float().to(device),
            )
            .cpu()
            .detach()
            .numpy()
        )
        p_R_evaluated = np.exp(
            _utils.compute_log_p_x(
                p_R_model, torch.as_tensor(X_test).float().to(device)
            )
            .cpu()
            .detach()
            .numpy()
        )
        return p_G_evaluated, p_R_evaluated