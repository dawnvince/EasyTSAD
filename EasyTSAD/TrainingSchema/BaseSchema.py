import os
import toml
import numpy as np
import logging
import json

from ..Controller import PathManager
from ..DataFactory import TSData, LoadData
from ..Evaluations import Performance
from ..Runtime.Timer import PerformanceTimer

from ..utils import update_nested_dict

class BaseSchema(object):
    '''
    The `BaseSchema` class is a base class that provides common functionality for working with schemas in a specific method. It is used as a blueprint for creating subclasses that implement specific methods and schemas. 

    - `__init__(self, dc, ec, method, schema:str, cfg_path:str=None, diff_order:int=None, preprocess:str=None)`: This is the constructor method that initializes an instance of the class.

    - `load_data(self, use_diff=True)`: This method loads the data based on the specified configuration. It takes an optional parameter `use_diff` that determines whether to use the differential order. It returns a dictionary containing the loaded time series data.

    - `do_exp(self)`: This method performs the experiment. It is meant to be overridden by subclasses.

    - `do_eval(self, tsDatas, evals)`: This method performs evaluation on the time series data. 

    The class also has instance variables `dc`, `ec`, `method`, `schema`, `pm`, and `cfg` that store the respective values passed to the constructor.
    '''
    def __init__(self, dc, method, schema:str, cfg_path:str=None, diff_order:int=None, preprocess:str=None) -> None:
        """
        Initializes an instance of the BaseSchema class.

        Args:
            - `dc` (dict): Data configuration parameters.
            - `method` (str): The method being used.
            - `schema` (str): The schema being used.
            - `cfg_path` (str, optional): Path to a custom configuration file. Defaults to None.
            - `diff_order` (int, optional): The differential order. Defaults to None.
            - `preprocess` (str, optional): The preprocessing method. Options: "raw", "min-max", "z-score". Defaults to None (equals to "raw"). 
        """
        self.logger = logging.getLogger("logger")
        self.dc = dc
        self.schema = schema
        self.method = method
        self.pm = PathManager.get_instance()
        self.cfg = {
            "Data_Params" : {
                "preprocess" : "raw",
                "diff_order" : 0
                }
            }
        
        use_cfg = False
        
        if cfg_path is None:
            cfg_path = self.pm.get_default_config_path(method)
            if os.path.exists(cfg_path):
                self.logger.info("    Use Default Method Config. Path: %s"%cfg_path)
                new_cfg = toml.load(cfg_path)
                self.cfg = update_nested_dict(self.cfg, new_cfg)
                use_cfg = True
            else:
                self.logger.info("Use Function Parameters.")
                
        else:
            use_cfg = True
            cfg_path = os.path.abspath(cfg_path)
            self.pm.check_valid(cfg_path, "Config File doesn't exist. Please specify the config file.")
            self.logger.info("Use Customized Method Config. Path: %s"%cfg_path)
            self.cfg = toml.load(cfg_path)
            
        ## function params overwrite original ones
        if diff_order:
            if use_cfg:
                self.logger.warning("The params \"diff_order\"(%d) will overwrite the original ones(%d)."%(diff_order, self.cfg["Data_Params"]["diff_order"]))
            self.cfg["Data_Params"]["diff_order"] = diff_order
            
        if preprocess:
            if use_cfg:
                self.logger.warning("The params \"preprocess\"(%s) will overwrite the original ones(%s)."%(preprocess, self.cfg["Data_Params"]["preprocess"]))
            self.cfg["Data_Params"]["preprocess"] = preprocess    
            
            
        ## initialize training timer and test timer.
        self.train_valid_timer = PerformanceTimer()
        self.test_timer = PerformanceTimer()
 
        
    def load_data(self):
        """
        Loads the data based on the specified configuration.

        Returns:
            dict: A dictionary containing the loaded time series data.
        """
        
        return LoadData.load_data(
            self.dc, 
            self.cfg["Data_Params"]["preprocess"], 
            self.cfg["Data_Params"]["diff_order"]
        )
        
    def do_exp(self):
        """
        Performs the experiment.
        """
        pass
    
        
                
                
                