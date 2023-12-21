from typing import Dict
from sklearn.svm import OneClassSVM
import numpy as np
from ...DataFactory import TSData
from .. import BaseMethod

class SubOCSVM(BaseMethod):
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
        
        self.seq_len = params["seq_len"]
    
    def gen_subseq(self, data):
        # N x 1 --> N x seq_len
        new_data = []
        for i in range(data.shape[0] - self.seq_len + 1):
            new_data.append(data[i:i+self.seq_len])
        new_data = np.array(new_data)
        
        return new_data
    
    def train_valid_phase(self, tsTrain: TSData):
        cat_data = np.concatenate([tsTrain.train, tsTrain.valid])
        cat_data = self.gen_subseq(cat_data)
        self.ocsvm.fit(cat_data)
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
        
    def test_phase(self, tsData: TSData):
        test_data = self.gen_subseq(tsData.test)
        self.__anomaly_score = -self.ocsvm.decision_function(test_data)
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score