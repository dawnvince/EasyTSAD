from typing import Dict
import tqdm
from ...DataFactory import TSData
from .. import BaseMethod
import numpy as np


class Diff(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
    
    def train_valid_phase(self, tsTrain: TSData):
        pass
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
        
    def test_phase(self, tsData: TSData):
        self.__anomaly_score = np.abs(tsData.test)
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score