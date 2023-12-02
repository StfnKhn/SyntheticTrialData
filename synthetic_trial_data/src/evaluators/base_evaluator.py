from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """
    Base class for all Evaluator classes.

    :param name: The name of the Evaluator
    :type name: str
    """

    def __init__(self):
        pass
    
    @abstractmethod
    def evaluate(self, X_real, X_synth, **kwargs) -> dict:
        """
        Evaluate the synthetic data generated against the real data.

        This is an abstract method and must be implemented in any class that 
        inherits from this class.

        :param X_real: The real data to be compared against
        :type X_real: type is dependent on implementation
        :param X_synth: The synthetic data to be evaluated
        :type X_synth: type is dependent on implementation
        :param kwargs: Other optional parameters for evaluation
        :type kwargs: various
        :return: A dictionary with evaluation metrics and their respective scores
        :rtype: dict
        :raises NotImplementedError: This is an abstract method and should be 
                                      implemented in any subclass.
        """
        raise NotImplementedError