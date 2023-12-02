import logging
import numpy as np
import pandas as pd
from typing import Union, List
from synthcity.metrics.eval import Metrics

from synthetic_trial_data.src.evaluators.base_evaluator import BaseEvaluator
from synthetic_trial_data.src.utils.dataframe_handling import to_dataframe
from synthetic_trial_data.src.metrics.identification_risk import identification_risk
from synthetic_trial_data.src.preprocessing.preprocessors import Encoder, Imputer
from synthetic_trial_data.src.metrics.domiasMIA import DomiasMIABNAF, DomiasMIAKDE


logger = logging.getLogger(__name__)


class Identifiability(BaseEvaluator):
    """
    Measures the epsilon-Identifiability according to :cite:`yoon2020`

    :param name: The name of the Evaluator, optional.
    :type name: str, optional

    .. [yoon2020] Yoon, J., Drumright, L. N., & Van Der Schaar, M. (2020).
        Anonymization through data synthesis using generative adversarial networks (ads-gan).
        IEEE journal of biomedical and health informatics, 24(8), 2378-2388.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "epsilon-identifiability"
    
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
        Computes the epsilon-Identifiability score

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
            metrics={'privacy': ["identifiability_score"]}
        )

        result = {
            __class__.name(): result_df["max"]["privacy.identifiability_score.score"]
        }
        
        return result


class AttributeDisclosureRisk(BaseEvaluator):
    """
    Compute the identification risk for given sensitive columns in the real dataset 
    against the synthetic dataset. The methodology is proposed by :cite:`Emam2020` and 
    enhanced by :cite:`Mendelevitch2021`. The enhancements made by :cite:`Mendelevitch2021`
    are all used in this implementation. It is important to note that the adjustment factor
    $\lambda_s^'$ is fixed to 1 in this implementation, which corresponds to the most
    conservative assumption for this factor.
    
    :param X_real: Real dataset with actual records.
    :type X_real: pd.DataFrame
    :param X_synth: Synthetic dataset.
    :type X_synth: pd.DataFrame
    :param QID: List of quasi-identifier columns.
    :type QID: List[str]
    :param S: List of sensitive columns for which the risk is to be calculated.
    :type S: List[str]
    :param risk_per_column: If True then the risk per column is being returned
        otherwise the sum of all sensitive columns in S are being returned.
        Default is False
    :type risk_per_column: bool
    
    :return: Dictionary with the calculated risk per sensitive column.
    :rtype: dict[str, float]

    Example:
    .. code-block:: python

        # Create some sample data
        data_real = {
            "age": [25, 30, 35, 40, 45],
            "gender": ["male", "female", "male", "female", "male"],
            "income": [50000, 55000, 60000, 65000, 70000],
            "married": [True, False, True, False, True]
        }

        data_synth = {
            "age": [26, 31, 36, 41, 46],
            "gender": ["male", "female", "male", "female", "male"],
            "income": [51000, 56000, 61000, 66000, 71000],
            "married": [True, False, True, False, True]
        }

        df_real = pd.DataFrame(data_real)
        df_synth = pd.DataFrame(data_synth)

        # Define QID and S
        QID = ["age", "gender"]
        S = ["income", "married"]

        # Compute the identification risk
        risk = identification_risk(df_real, df_synth, QID, S)
        print(risk)
    
    .. [Emam2020] El Emam, K., Mosquera, L., & Bass, J. (2020). 
        Evaluating identity disclosure risk in fully synthetic health data: 
        model development and validation. Journal of medical Internet 
        research, 22(11), e23139.

    .. [Mendelevitch2021] Mendelevitch, O., & Lesh, M. D. (2021). 
        Fidelity and privacy of synthetic medical data. 
        arXiv preprint arXiv:2101.08658.
    """
    def __init__(
        self,
        QID: List[str], 
        S: List[str],
        risk_per_column: bool = False
    ):
        super().__init__()
        self.QID = QID
        self.S = S
        self.risk_per_column = risk_per_column
        
    @staticmethod
    def name() -> str:
        return "Attribute Disclosure Risk"
    
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
        Compute the AttributeDisclosureRisk.

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

        # Imputation required for stable computation
        X_synth = Imputer().fit_transform(X_synth)
        X_real = Imputer().fit_transform(X_real)

        # Execute AttributeDisclosureRisk comnputation
        result = identification_risk(X_real, X_synth, self.QID, self.S)

        if self.risk_per_column:
            result = {"attr_disclosure_risk_per_col": result}
        else:
            total_atrr_discolsure_risk = sum(list(result.values()))
            result = {__class__.name(): total_atrr_discolsure_risk}
        
        return result


class DomiasKDE(BaseEvaluator):
    """
    DOMIAS is a membership inference attack framework for synthetic data, utilizing
    density estimation to identify overfitting in generative models :cite:`Breugel2023`. 
    Essentially, it leverages local overfitting to determine if a data point was part of
    the generative model's training set. The density estimation using DomiasKDE is
    implemented using Kernel Density Estimation. It is reccomended to chose DomiasKDE 
    for smaller datasets.

    Key Features:
        * Assumes knowledge of the true data distribution pR(X) in terms of a 
        reference Dataset (X_real_val) that was not used in the training process
        of the generative model.
        * Weights the generative model's output distribution by the real data
        distribution, making it robust against changes in data representation 
        * Overcomes the limitations of previous black-box MIA methods that do
        not account for the intrinsic distribution of data.
        * Does not give a false sense of privacy safety for underrepresented
        groups. It recognizes that low-density regions, which might correspond
        to minority groups, can still be attacked.

    Returns a dictionary with accuracy and AUCROC for each attack.

    .. [Breugel2023] Boris van Breugel, Hao Sun, Zhaozhi Qian,  Mihaela van der Schaar, 
        AISTATS 2023. DOMIAS: Membership Inference Attacks against Synthetic Data 
        through Overfitting Detection.

    """
    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "domias"

    @staticmethod
    def metrics() -> List[str]:
        return ["DOMIAS - accuracy", "DOMIAS - aucroc"]

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def polarity() -> int:
        return {"DOMIAS - accuracy": -1, "DOMIAS - aucroc": -1} 
        
    def evaluate(
        self,
        X_real_train: Union[np.ndarray, pd.DataFrame],
        X_real_val: Union[np.ndarray, pd.DataFrame],
        X_synth: Union[np.ndarray, pd.DataFrame],
        reference_size: int = 100,
        **kwargs
    ):
        """
        Performs DOMIAS MIA based on Kernel Density Estimation (KDE) using the provided 
        real and synthetic datasets as well as a real reference dataset

        Make sure you cleaned the dataset from all NaN values beforehand, since 
        sklearns KernelDensity does not accept missing values encoded as NaN 
        natively.

        :param X_real_train: Real training dataset.
        :type X_real_train: Union[np.ndarray, pd.DataFrame]
        :param X_real_val: Real validation dataset that was not used for the training
            of the generative model one wants to asses. It is 50% of datapoints are
            being used as a reference data set and other 50% are being used to 
            fill up the real training data that the attack is targeted to.
        :type X_real_val: Union[np.ndarray, pd.DataFrame]
        :param X_synth: Synthetic dataset.
        :type X_synth: Union[np.ndarray, pd.DataFrame]
        :param reference_size: Number of points from X_real_val that should 
            be considered as the reference set.
        :type reference_size: int
        
        :return: Results of the evaluation containing metrics.
        :rtype: Dict
        """

        # If not already convert data to type pd.DataFrame
        X_real_train = to_dataframe(X_real_train)
        X_real_val = to_dataframe(X_real_val)
        X_synth = to_dataframe(X_synth)

        # OHE categorical columns
        encoder = Encoder(cat_encoder="ohe")
        encoder.fit(X_real_train)
        X_real_train_encoded = encoder.transform(X_real_train)
        X_real_val_encoded = encoder.transform(X_real_val)
        X_synth_encoded = encoder.transform(X_synth)

        # Evaluate DomiasMIA based on Kernel density estimation
        # Note: synth_val_set is not being used in the computation
        #   it is only required to use the DomiasMIA factory of
        #   the synthcity package
        result = DomiasMIAKDE().evaluate(
            X_gt=X_real_val_encoded,
            synth_set=X_synth_encoded,
            X_train=X_real_train_encoded,
            synth_val_set=X_synth_encoded,
            reference_size=reference_size
        )

        output_dict = {
            "DOMIAS - accuracy": result["accuracy"],
            "DOMIAS - aucroc": result["aucroc"]
        }

        return output_dict


class DomiasBNAF(BaseEvaluator):
    """
    DOMIAS is a membership inference attack framework for synthetic data, utilizing
    density estimation to identify overfitting in generative models :cite:`Breugel2023`. 
    Essentially, it leverages local overfitting to determine if a data point was part of
    the generative model's training set. The density estimation using DomiasBNAF is
    implemented using a Block Neural Autoregressive Flow (BNAF) :cite:`Cao2020`.
    Use it for larger data sets where Kernel Density Estimation runs into memory and
    runtime issues.

    Key Features:
        * Assumes knowledge of the true data distribution pR(X) in terms of a 
        reference Dataset (X_real_val) that was not used in the training process
        of the generative model.
        * Weights the generative model's output distribution by the real data
        distribution, making it robust against changes in data representation 
        * Overcomes the limitations of previous black-box MIA methods that do
        not account for the intrinsic distribution of data.
        * Does not give a false sense of privacy safety for underrepresented
        groups. It recognizes that low-density regions, which might correspond
        to minority groups, can still be attacked.

    Returns a dictionary with accuracy and AUCROC for each attack.

    .. [Breugel2023] Boris van Breugel, Hao Sun, Zhaozhi Qian,  Mihaela van der Schaar, 
        AISTATS 2023. DOMIAS: Membership Inference Attacks against Synthetic Data 
        through Overfitting Detection.

    .. [Cao2020] De Cao, N., Aziz, W., & Titov, I. (2020, August). 
        Block neural autoregressive flow. In Uncertainty in artificial
        intelligence (pp. 1263-1273). PMLR.

    """
    def __init__(
        self,
    ):
        super().__init__()

    @staticmethod
    def name() -> str:
        return "domiasBNAF"

    @staticmethod
    def metrics() -> List[str]:
        return ["DOMIAS-BNAF - accuracy", "DOMIAS-BNAF - aucroc"]

    @staticmethod
    def direction() -> str:
        return "minimize"

    @staticmethod
    def polarity() -> int:
        return {"DOMIAS-BNAF - accuracy": -1, "DOMIAS-BNAF - aucroc": -1}
        
    def evaluate(
        self,
        X_real_train: Union[np.ndarray, pd.DataFrame],
        X_real_val: Union[np.ndarray, pd.DataFrame],
        X_synth: Union[np.ndarray, pd.DataFrame],
        X_synth_val: Union[np.ndarray, pd.DataFrame],
        reference_size: int = 100,
        **kwargs
    ):
        """
        Performs DOMIAS MIA based on density estimation through Block Neural 
        Autoregressive Flow (BNAF) using the provided real and synthetic datasets
        as well as a real reference dataset. Here also a synthetic validation set
        is required for the training of the BNAF model.

        Make sure you cleaned the dataset from all NaN values beforehand.

        :param X_real_train: Real training dataset.
        :type X_real_train: Union[np.ndarray, pd.DataFrame]
        :param X_real_val: Real validation dataset that was not used for the training
            of the generative model one wants to asses. It is 50% of datapoints are
            being used as a reference data set and other 50% are being used to 
            fill up the real training data that the attack is targeted to.
        :type X_real_val: Union[np.ndarray, pd.DataFrame]
        :param X_synth: Synthetic dataset.
        :type X_synth: Union[np.ndarray, pd.DataFrame]
        :param X_synth_val: Synthetic validation dataset required for the
            training of the BNAF.
        :type X_synth_val: Union[np.ndarray, pd.DataFrame]
        :param reference_size: Number of points from X_real_val that should 
            be considered as the reference set.
        :type reference_size: int
        
        :return: Results of the evaluation containing metrics.
        :rtype: Dict
        """

        # If not already convert data to type pd.DataFrame
        X_real_train = to_dataframe(X_real_train)
        X_real_val = to_dataframe(X_real_val)
        X_synth = to_dataframe(X_synth)
        X_synth_val = to_dataframe(X_synth_val)

        # OHE categorical columns
        encoder = Encoder(cat_encoder="ohe")
        encoder.fit(X_real_train)
        X_real_train_encoded = encoder.transform(X_real_train)
        X_real_val_encoded = encoder.transform(X_real_val)
        X_synth_encoded = encoder.transform(X_synth)
        X_synth_val_encoded = encoder.transform(X_synth_val)


        # Evaluate DomiasMIA based on Kernel density estimation
        # Note: synth_val_set is not being used in the computation
        #   it is only required to use the DomiasMIA factory of
        #   the synthcity package
        result = DomiasMIABNAF().evaluate(
            X_gt=X_real_val_encoded,
            synth_set=X_synth_encoded,
            X_train=X_real_train_encoded,
            synth_val_set=X_synth_val_encoded,
            reference_size=reference_size
        )

        output_dict = {
            "DOMIAS-BNAF - accuracy": result["accuracy"],
            "DOMIAS-BNAF - aucroc": result["aucroc"]
        }

        return output_dict