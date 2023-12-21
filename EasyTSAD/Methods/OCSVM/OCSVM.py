from typing import Dict
from sklearn.svm import OneClassSVM
import numpy as np
from ...DataFactory import TSData
from .. import BaseMethod

class OCSVM(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.ocsvm = OneClassSVM(
            kernel=params["kernel"],
            degree=params["degree"],
            gamma=params["gamma"],
            coef0=params["coef0"],
            tol=params["tol"],
            nu=params["nu"],
            shrinking=params["shrinking"],
            cache_size=params["cache_size"],
            verbose=params["verbose"],
            max_iter=params["max_iter"]
        )
    
    def train_valid_phase(self, tsTrain: TSData):
        cat_data = np.concatenate([tsTrain.train, tsTrain.valid])
        cat_data = np.reshape(cat_data, (-1, 1))
        self.ocsvm.fit(cat_data)
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
        
    def test_phase(self, tsData: TSData):
        self.__anomaly_score = -self.ocsvm.decision_function(tsData.test.reshape((-1, 1)))
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score