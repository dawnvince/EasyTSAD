from typing import Dict
import numpy as np
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
    datasets = ["TODS"]
    dataset_types = "UTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="/path/to/datasets",
        datasets=datasets,
    )
    
    # Or specify certain curves in one dataset, 
    # e.g. AIOPS 0efb375b-b902-3661-ab23-9a0bb799f4e3 and ab216663-dcc2-3a24-b1ee-2c3e550e06c9
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="/path/to/datasets",
        datasets="AIOPS",
        specify_curves=True,
        curve_names=[
            "0efb375b-b902-3661-ab23-9a0bb799f4e3",
            "ab216663-dcc2-3a24-b1ee-2c3e550e06c9"
        ]
    )
    
    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import BaseMethod
    from EasyTSAD.DataFactory import TSData

    class YourAlgo(BaseMethod):
        def __init__(self, hparams) -> None:
            super().__init__()
            self.__anomaly_score = None
            self.param_1 = hparams["param_1"]
        
        def train_valid_phase(self, tsTrain: TSData):
            '''
            Define train and valid phase for naive mode. All time series needed are saved in tsTrain. 
            
            tsTrain's property :
                train (np.ndarray):
                    The training set in numpy format;
                valid (np.ndarray):
                    The validation set in numpy format;
                test (np.ndarray):
                    The test set in numpy format;
                train_label (np.ndarray):
                    The labels of training set in numpy format;
                test_label (np.ndarray):
                    The labels of test set in numpy format;
                valid_label (np.ndarray):
                    The labels of validation set in numpy format;
                info (dict):
                    Some informations about the dataset, which might be useful.
                    
            NOTE : test and test_label are not accessible in training phase
            '''
            ...
            
            return
                
        def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
            '''
            Define train and valid phase for all-in-one mode. All time series needed are saved in tsTrains. 
            
            tsTrain's structure:
                {
                    "name of time series 1": tsData1,
                    "name of time series 2": tsData2,
                    ...
                }
                
            '''
            ...
            
            return
            
        def test_phase(self, tsData: TSData):
            '''
            Define test phase for each time series. 
            '''
            
            # For example
            anomaly_score = np.abs(tsData.test)
            self.__anomaly_score = anomaly_score
            
        def anomaly_score(self) -> np.ndarray:
            return self.__anomaly_score
        
        def param_statistic(self, save_file):
            param_info = "Your Algo. info"
            with open(save_file, 'w') as f:
                f.write(param_info)
    
    """============= Run your algo. ============="""
    # Specifying methods and training schemas
    
    training_schema = "naive"
    method = "YourAlgo"  # string of your algo class
    
    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        cfg_path="YourAlgo.toml" # path/to/your
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
