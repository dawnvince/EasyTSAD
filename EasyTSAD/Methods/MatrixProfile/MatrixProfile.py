from typing import Dict
import matrixprofile as mp
import numpy as np
from ...DataFactory import TSData
from .. import BaseMethod

class MatrixProfile(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.window = params["window"]
        self.name = 'MatrixProfile'
        self.n_jobs = params["n_jobs"]
    
    def train_valid_phase(self, tsTrain: TSData):
        pass
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
        
        
    def test_phase(self, tsData: TSData):
        test_len = tsData.test.shape[0]
        cat_data = np.concatenate([tsData.train, tsData.valid, tsData.test])
        profile = mp.compute(cat_data, windows=self.window, n_jobs=self.n_jobs)
        self.__anomaly_score = profile["mp"][-test_len:]
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score