from typing import Dict
import numpy as np
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["SMD", "SWaT", "WADI", "SMAP", "MSL", "PSM"]
    datasets = ["WADI"]
    dataset_types = "MTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="/path/to/datasets",
        datasets=datasets,
    )

    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import MTSData

    class MTSExample(BaseMethod):
        def __init__(self, params:dict) -> None:
            super().__init__()
            self.__anomaly_score = None
            
        def train_valid_phase(self, tsData):

            print(tsData.train.shape)
            pass
            
        def test_phase(self, tsData: MTSData):
            test_data = tsData.test
            
            scores = np.sum(np.square(test_data), axis=1)
            
            if len(scores) > 0:
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                
            self.__anomaly_score = scores
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            param_info = "Your Algo. info"
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    """============= Run your algo. ============="""
    # Specifying methods and training schemas

    training_schema = "mts"
    method = "MTSExample"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        # hparams={
        #     "param_1": 2,
        # },
        # use which method to preprocess original data. 
        # Default: raw
        # Option: 
        #   - z-score(Standardlization), 
        #   - min-max(Normalization), 
        #   - raw (original curves)
        preprocess="z-score", 
    )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
