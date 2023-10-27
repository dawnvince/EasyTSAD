from typing import Dict
import numpy as np
from ...DataFactory import TSData
from .. import BaseMethod
from .spot import SPOT

class Spot(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.s = SPOT(params["q"])
        
    def train_valid_phase(self, tsTrain: TSData):
        pass
    
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
    
    def test_phase(self, tsData: TSData):
        self.s.fit(tsData.train, tsData.test)
        self.s.initialize() 		# initialization step
        results = self.s.run()
        
        score = np.zeros_like(tsData.test)
        print(results['alarms'])
        score[results['alarms']]=1
        
        self.__anomaly_score = score
        
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score