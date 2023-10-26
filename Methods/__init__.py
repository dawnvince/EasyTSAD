import numpy as np
from ..DataFactory import TSData

class BaseMethodMeta(type):
    """
    Metaclass register implemented methods automaticly. This allows the usage of runtime arguments to specify the method to run experiments.

    Attributes:
        registry (dict): Registry to store the registered methods.

    """
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if name != 'BaseMethod':
            BaseMethodMeta.registry[name] = cls
     
            
class BaseMethod(metaclass=BaseMethodMeta):
    """
    BaseMethod class. Assuming that the name of your method is A, following the steps to run your method:
    1. Create "A" folder under "Method" directory; (Recommanded & Optional)

    2. Create "config.toml" to set the parameters needed for DataPreprocess and Analysis.
    
    3. Create "A.py" and create class A which should inherite abstract class BaseMethod. 
     - Override the function "train_valid_phase" to train model A; 
     - Override the function "test_phase" to test A and generate anomaly scores; 
     - Override the function "anomaly_score" which returns the anomaly score in np.ndarray format for further evaluation;
     
     Optional: 
      if your method support training in all_in_one mode or transfer mode, OVERRIDE the function "train_valid_phase_all_in_one" to train model A; 
     Optional:
      if you want to active "plot_y_hat" option in config.toml, OVERRIDE the function "get_y_hat" to save y_hat values; 
     NOTE: 
      the function "train_valid_phase_all_in_one" receive a Dict of TSDatas INSTEAD OF one TSData instanece in function "train_valid_phase" as its args.
    
    4. Fullfill "config.toml" to set the parameters needed in your class. 

    Methods:
        anomaly_score(self) -> np.ndarray:
            Calculates the anomaly scores for the time series data. Must be implemented by subclasses.

        get_y_hat(self) -> np.ndarray:
            Returns the  values for the time series data if . Can be overridden by subclasses.

        train_valid_phase(self, tsTrain: TSData):
            Performs the training and validation phase for the given training time series data. Must be implemented by subclasses.

        train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
            Performs the training and validation phase, considering multiple training time series datasets. Can be overridden by subclasses.

        test_phase(self, tsData: TSData):
            Performs the testing phase on the given time series data. Must be implemented by subclasses.
            
        param_statistic(self):
            Counting the params/Flops before training. Torchinfo is recommended if the method is implemented in Pytorch.
    """
    def anomaly_score(self) -> np.ndarray:
        """
        Calculates the anomaly scores for the time series data.

        Returns:
            np.ndarray: Array of anomaly scores.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError()
    
    def get_y_hat(self) -> np.ndarray:
        """
        Returns the predicted values for the time series data. Recommended to override when you employ MAE/MSE loss.

        Returns:
            np.ndarray: Array of predicted values.

        """
        pass
    
    def train_valid_phase(self, tsTrain: TSData):
        """
        Performs the training and validation phase for the given training time series data.

        Args:
            tsTrain (TSData): Training time series data.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError()
    
    def train_valid_phase_all_in_one(self, tsTrains: dict[str, TSData]):
        """
        Performs the training and validation phase, considering multiple training time series datasets.

        Args:
            tsTrains (Dict[str, TSData]): Dictionary of training time series datasets.

        """
        pass
    
    def test_phase(self, tsData: TSData):
        """
        Performs the testing phase on the given time series data.

        Args:
            tsData (TSData): Time series data for testing.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.

        """
        raise NotImplementedError()
    
    def param_statistic(self, save_file):
        pass

# __all__ = ['AE', 'AnomalyTransformer', 'AR', 'Donut', 'EncDecAD', 'FCVAE', 'LSTMADalpha', 'LSTMADbeta', 'SRCNN', 'TFAD', 'TimesNet']

# from .AE.AE import AE
# from .AnomalyTransformer.AnomalyTransformer import AnomalyTransformer
# from .AR.AR import AR
# from .Donut.Donut import Donut
# from .EncDecAD.EncDecAD import EncDecAD
# from .FCVAE.FCVAE import FCVAE
# from .LSTMADalpha.LSTMADalpha import LSTMADalpha
# from .LSTMADbeta.LSTMADbeta import LSTMADbeta
# from .SRCNN.SRCNN import SRCNN
# from .TFAD.TFAD import TFAD
# from .TimesNet.TimesNet import TimesNet