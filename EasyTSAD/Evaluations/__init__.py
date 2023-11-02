from abc import ABCMeta,abstractmethod
from typing import Dict, Type

class MetricInterface(object):
    '''
    The MetricInterface class is an abstract base class that defines the interface for metrics. It serves as a blueprint for creating subclasses that represent specific metrics. 
    
    The class includes three abstract methods: add(), avg(), and to_dict(). Subclasses inheriting from this class must implement these methods according to their specific metric calculations and requirements.
    
    You should implement the following methods:
    
        - add(self, other_metric): This abstract method represents the operation of combining two metrics. It takes another metric object (other_metric) as a parameter and is responsible for adding its values to the current metric.

        - avg(self): This abstract method calculates the average value of the metric. It should be implemented by subclasses to compute the average based on the accumulated values.

        - to_dict(self): This abstract method converts the metric object into a dictionary representation. It should return a dictionary containing the metric's values and any additional information needed for representation or storage.
        
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def add(self, other_metric):
        """
        This abstract method represents the operation of combining two metrics. It takes another metric object (other_metric) as a parameter and is responsible for adding its values to the current metric.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def avg(self):
        """
        This abstract method calculates the average value of the metric. It should be implemented by subclasses to compute the average based on the accumulated values.
        """
        raise NotImplementedError()
    
    @abstractmethod
    def to_dict(self):
        """
        This abstract method converts the metric object into a dictionary representation. It should return a dictionary containing the metric's values and any additional information needed for representation or storage.
        """
        raise NotImplementedError()
 
    
class EvalInterface(object):
    '''
    The EvalInterface is an abstract base class that defines the interface for evaluation metrics in a generic evaluation system. It serves as a blueprint for concrete evaluation metric classes that implement specific evaluation logic.

    Methods:
        - calc(self, scores, labels, margins) -> Type[MetricInterface]: Abstract method that calculates the evaluation metric based on the provided scores, labels, and margins parameters. It returns an instance of a class that inherits from MetricInterface.
    
        - get_name(self): Abstract method that returns the name of the evaluation metric. Concrete classes implementing this interface should provide their own implementation of this method.
    '''
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def calc(self, scores, labels, margins) -> Type[MetricInterface]:
        """
        Calculates the evaluation metric based on the provided scores, labels, and margins.

        Args:
            scores (numpy.ndarray): An array of anomaly scores.
            labels (numpy.ndarray): An array of ground truth labels.
            margins ([int(margin-before-anomaly), int(margin-after-anomaly)]): You can use margin to prune your evaluations if needed. Defined in https://github.com/dawnvince/EasyTSAD/blob/main/EasyTSAD/Controller/GlobalCfg.toml. 

        Returns:
            MetricInterface: An instance of the event detection metric.

        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_name(self):
        return self.name
    
from .Performance import Performance
