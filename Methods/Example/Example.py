'''
Example method class. Assuming that the name of your method is A, following the steps to run your method:
1. Create "A" folder under "Method" directory;

2. Create "A.py" and create class A which should inherite abstract class BaseMethod. 
 - Override the function "train_valid_phase" to train model A; 
 - Override the function "test_phase" to test A and generate anomaly scores; 
 - Override the function "anomaly_score" which returns the anomaly score in np.ndarray format for further evaluation;
 
 NOTE: if your method support training in all_in_one mode or transfer mode, OVERRIDE the function "train_valid_phase_all_in_one" to train model A; 
     
 ATTENTION: the function "train_valid_phase_all_in_one" receive a Dict of TSData INSTEAD OF TSData in function "train_valid_phase" as its args.
 
3. Create "config.toml" to set the parameters needed in your class. "config.toml" also includes some running settings.
'''
import numpy as np
from .. import BaseMethod
from ...DataFactory import TSData
from typing import Dict

class Example(BaseMethod):
    def __init__(self, params, cuda) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.param_1 = params["param_1"]
        self.model = ... # your model
        
    def train_valid_phase(self, tsTrain: TSData):
        
        train_data = tsTrain.train
        valid_data = tsTrain.valid
        
        train_label = tsTrain.train_label
        valid_label = tsTrain.valid_label
        
        self.model = Model(self.param_1)
        self.model.train(train_data)
        self.model.valid(valid_data)
        
    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_data = None
        
        '''
        Concat multiple curves in a dataset into one.
        
        For example, Donut append all time windows into one torch.Dataset for torch-based & window-based method. More details please refer to TSDataset in Donut Method.
        '''
        for k, v in tsTrains.items():
            if not train_data:
                train_data = v.train
                valid_data = v.valid
            else:
                train_data = train_data.concat(v.train)
                valid_data = valid_data.concat(v.test)
                
        self.model = Model(self.param_1)
        self.model.train(train_data)
        self.model.valid(valid_data)
    
    def test_phase(self):
        test_data = self.tsData.test

        scores = self.model.test(test_data)
        
        self.__anomaly_score = scores
        
    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score
