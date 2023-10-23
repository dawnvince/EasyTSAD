from abc import ABCMeta,abstractmethod
import numpy as np
from DataFactory import TSData
from typing import Dict

class BaseMethod(object):
    '''
    BaseMethod class. Assuming that the name of your method is A, following the steps to run your method:
    1. Create "A" folder under "Method" directory;

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
      the function "train_valid_phase_all_in_one" receive a Dict of TSData INSTEAD OF one TSData instanece in function "train_valid_phase" as its args.
    
    4. Fullfill "config.toml" to set the parameters needed in your class. 
    '''
    __metaclass__ = ABCMeta
    
    @property
    @abstractmethod
    def anomaly_score(self) -> np.ndarray:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def get_y_hat(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def train_valid_phase(self, tsTrain: TSData):
        raise NotImplementedError()
    
    @abstractmethod
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
    
    @abstractmethod
    def test_phase(self, tsData: TSData):
        raise NotImplementedError()
    