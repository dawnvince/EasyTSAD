from typing import Dict
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from ...DataFactory import TSData
from .. import BaseMethod

class LOF(BaseMethod):
    def __init__(self, params:dict) -> None:
        super().__init__()
        self.lof = LocalOutlierFactor(
            n_neighbors=params["n_neighbors"],
            algorithm=params["algorithm"],
            metric=params["metric"],
            n_jobs=params["n_jobs"],
            novelty=True
        )
    
    def train_valid_phase(self, tsTrain: TSData):
        cat_data = np.concatenate([tsTrain.train, tsTrain.valid])
        cat_data = np.reshape(cat_data, (-1, 1))
        self.lof.fit(cat_data)
            
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        pass
        
    def test_phase(self, tsData: TSData):
        self.__anomaly_score = -self.lof.decision_function(tsData.test.reshape((-1, 1)))
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score