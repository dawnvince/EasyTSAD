import os
import toml
import numpy as np
import logging
import json

from Controller import PathManager
from DataFactory import TSData, LoadData


logger = logging.getLogger("logger")

class BaseSchema(object):
    '''
    The `BaseSchema` class is a base class that provides common functionality for working with schemas in a specific method. It is used as a blueprint for creating subclasses that implement specific methods and schemas. 

    - `__init__(self, dc, ec, method, schema:str, cfg_path:str=None, diff_order:int=None, preprocess:str=None)`: This is the constructor method that initializes an instance of the class.

    - `load_data(self, use_diff=True)`: This method loads the data based on the specified configuration. It takes an optional parameter `use_diff` that determines whether to use the differential order. It returns a dictionary containing the loaded time series data.

    - `do_exp(self)`: This method performs the experiment. It is meant to be overridden by subclasses.

    - `do_eval(self, tsDatas, evals)`: This method performs evaluation on the time series data. 

    The class also has instance variables `dc`, `ec`, `method`, `schema`, `pm`, and `cfg` that store the respective values passed to the constructor.
    '''
    def __init__(self, dc, ec, method, schema:str, cfg_path:str=None, diff_order:int=None, preprocess:str=None) -> None:
        """
        Initializes an instance of the BaseSchema class.

        Args:
            dc (dict): Data configuration parameters.
            ec (dict): Evaluation configuration parameters.
            method (str): The method being used.
            schema (str): The schema being used.
            cfg_path (str, optional): Path to a custom configuration file. Defaults to None.
            diff_order (int, optional): The differential order. Defaults to None.
            preprocess (str, optional): The preprocessing method. Options: "raw", "min-max", "z-score". Defaults to None (equals to "raw"). 
        """
        self.dc = dc
        self.ec = ec
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
                logger.info("Use Default Method Config. Path: %s"%cfg_path)
                self.cfg = toml.load(cfg_path)
                use_cfg = True
            else:
                logger.info("Use Function Parameters.")
                
        else:
            use_cfg = True
            cfg_path = os.path.abspath(cfg_path)
            self.pm.check_valid(cfg_path, "Config File doesn't exist. Please specify the config file.")
            logger.info("Use Customized Method Config. Path: %s"%cfg_path)
            self.cfg = toml.load(cfg_path)
            
        ## function params overwrite original ones
        if diff_order:
            if use_cfg:
                logger.warning("The params \"diff_order\"(%d) will overwrite the original ones(%d)."%(diff_order, self.cfg["Data_Params"]["diff_order"]))
            self.cfg["Data_Params"]["diff_order"] = diff_order
            
        if preprocess:
            if use_cfg:
                logger.warning("The params \"preprocess\"(%s) will overwrite the original ones(%s)."%(preprocess, self.cfg["Data_Params"]["preprocess"]))
            self.cfg["Data_Params"]["preprocess"] = preprocess    
 
        
    def load_data(self, use_diff=True):
        """
        Loads the data based on the specified configuration.

        Args:
            use_diff (bool, optional): Whether to use differential order. Defaults to True.

        Returns:
            dict: A dictionary containing the loaded time series data.
        """
        
        if self.dc["specify_curves"]:
            diff_p = self.cfg["Data_Params"]["diff_order"]
            if not use_diff:
                diff_p = 0
            tsDatas = LoadData.load_specific_curves(
                types=self.dc["dataset_type"],
                dataset=self.dc["datasets"][0],
                curve_names=self.dc["specify_curves"],
                train_proportion=self.dc["train_proportion"], 
                valid_proportion=self.dc["valid_proportion"],
                preprocess=self.cfg["Data_Params"]["preprocess"],
                diff_p=diff_p
            )
            return tsDatas

        else:
            diff_p = self.cfg["Data_Params"]["diff_order"]
            if not use_diff:
                diff_p = 0
            tsDatas = LoadData.load_all_datasets(
                types=self.dc["dataset_type"],
                dataset=self.dc["datasets"],
                train_proportion=self.dc["train_proportion"], 
                valid_proportion=self.dc["valid_proportion"],
                preprocess=self.cfg["Data_Params"]["preprocess"],
                diff_p=diff_p
            )
            return tsDatas
        
    def do_exp(self):
        """
        Performs the experiment.
        """
        pass
    
    def do_eval(self, tsDatas, evals):
        """
        Performs evaluation on the time series data.

        Args:
            tsDatas (dict): A dictionary containing the time series data.
            evals: The evaluation methods to use.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        
        for dataset_name, value in tsDatas.items():
            logger.info(">>> [{}] Eval dataset {} <<<".format(self.method, dataset_name))
            eval_dict = {}
            avg_res = None
            avg_len = -1
            
            margins = (0, 0)
            if self.ec["use_margin"] == True:
                if dataset_name in self.ec:
                    margins = (self.ec[dataset_name][0], self.ec[dataset_name][1])
                    logger.info("    [{}] Using margins {}".format(dataset_name, margins))
                else:
                    margins = (self.ec["default"][0], self.ec["default"][1])
                    logger.info("    [{}] Using default margins {}".format(dataset_name, margins))
            
            types = self.dc["dataset_type"] 
               
            score_files = self.pm.get_score_curves(self.method, self.schema, dataset_name)
            eval_dict_path = self.pm.get_eval_json_all(self.method, self.schema, dataset_name)
            avg_json_path = self.pm.get_eval_json_avg(self.method, self.schema, dataset_name)
            
            for score_file in score_files:
                curve_name = score_file[:-4] # delete the ".npy" suffix
                
                score_path = self.pm.get_score_path(self.method, self.schema, dataset_name, curve_name)
                score = np.load(score_path)
                
                # calculate performance using multiple evaluation methods
                eva = Evaluations(scores=score, labels=value[curve_name].test_label, margins=margins)
                if eva.all_label_normal:
                    continue
                res, res_dict = eva.do_eval(evals)
                
                eval_dict[curve_name] = res_dict

                with open(eval_dict_path, 'w') as f:
                    json.dump(eval_dict, f, indent=4)
                
                if avg_res is None:
                    avg_res = res
                    avg_len = len(res)
                else:
                    for i in range(avg_len):
                        avg_res[i].add(res[i])
                        
            with open(eval_dict_path, 'w') as f:
                json.dump(eval_dict, f, indent=4)
            with open(avg_json_path, 'w') as f:
                res_dict = {}
                for i in avg_res:
                    i.avg()
                    res_dict.update(i.to_dict())
                json.dump(res_dict, f, indent=4)