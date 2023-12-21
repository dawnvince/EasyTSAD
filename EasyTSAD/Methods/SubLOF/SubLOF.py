from typing import Dict
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from ...DataFactory import TSData
from .. import BaseMethod

class SubLOF(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.lof = LocalOutlierFactor(
            n_neighbors=params["n_neighbors"],
            algorithm=params["algorithm"],
            metric=params["metric"],
            n_jobs=params["n_jobs"],
            novelty=True
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
        self.lof.fit(cat_data)
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
        
    def test_phase(self, tsData: TSData):
        test_data = self.gen_subseq(tsData.test)
        self.__anomaly_score = -self.lof.decision_function(test_data)
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score