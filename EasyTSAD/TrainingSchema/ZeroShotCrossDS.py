import json
from typing import Dict
import numpy as np
import sys
import math

from .BaseSchema import BaseSchema
from ..Methods import BaseMethodMeta
from ..Controller.PathManager import PathManager

class ZeroShotCrossDS(BaseSchema):
    def __init__(self, dc, method, cfg_path:str=None, diff_order:int=None, preprocess:str=None) -> None:
        """
        Initializes an instance of the ZeroShotCrossDS class.

        Args:
            - `dc` (dict): Data configuration parameters.
            - `ec` (dict): Evaluation configuration parameters.
            - `method` (str): The method being used.
            - `cfg_path` (str, optional): Path to a custom configuration file. Defaults to None.
            - `diff_order` (int, optional): The differential order. Defaults to None.
            - `preprocess` (str, optional): The preprocessing method. Options: "raw", "min-max", "z-score". Defaults to None (equals to "raw"). 
        """
        super().__init__(dc, method, "zero_shot_cross_ds", cfg_path, diff_order, preprocess)
        self.pm = PathManager.get_instance()
        
    def do_exp(self, tsDatas, hparams=None):
        if "Model_Params" in self.cfg:
            model_params = self.cfg["Model_Params"]["Default"]
        if hparams is not None:
            model_params = hparams
        
        if self.method in BaseMethodMeta.registry:
            method = BaseMethodMeta.registry[self.method](model_params)
        else:
            raise ValueError("Unknown method class \"{}\". Ensure that the method name matches the class name exactly (including case sensitivity).".format(self.method))

        new_tstrain = {}
        for dataset_name, value in tsDatas.items():
            if dataset_name in self.dc["src_datasets"]:
                for curve_name, tsData in value.items():
                    new_tstrain[dataset_name+curve_name] = tsData
        
        self.logger.info(
            "    [{}] is training using {}. The number of curves is {}.".format(self.method, new_tstrain.keys(), len(new_tstrain))
        )           
            
        ## training and validation phase
        ## self.train_valid_timer.tic()
        method.train_valid_phase_all_in_one(tsTrains=new_tstrain)
        ## self.train_valid_timer.toc()
        
        for dataset_name, value in tsDatas.items():
            if dataset_name in self.dc["dst_datasets"]:
                for curve_name, curve in value.items():
                    score_path = self.pm.get_score_path(self.method, self.schema, dataset_name, curve_name)
                    
                    ## test phase
                    self.test_timer.tic()
                    method.test_phase(curve)
                    self.test_timer.toc()
                    
                    score = method.anomaly_score()
                    
                    ## save scores and evaluation
                    np.save(score_path, score)
                    
            # # save running time info
            # time_path = self.pm.get_rt_time_path(self.method, self.schema, dataset_name)
            # time_dict = {
            #     "train_and_valid": self.train_valid_timer.get_total_time(),
            #     "test": self.test_timer.get_total_time()
            # }
            # with open(time_path, 'w') as f:
            #     json.dump(time_dict, f, indent=4)
                
        