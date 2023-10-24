import numpy as np
import os
import toml
import json
from typing import Union

from .logger import setup_logger
from .PathManager import PathManager

def __update_nested_dict(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            d1[k] = __update_nested_dict(d1.get(k, {}), v)
        else:
            d1[k] = v
    return d1

class GlobalController:
    def __init__(self, cfg_path=None, log_path=None, log_level="info") -> None:
        """
        GlobalController class represents a controller that manages global configuration and logging.

        Args:
            cfg_path (str, optional): Path to the configuration file. If provided, the configuration will be applied from this file. Defaults to None (Not Recommanded).
            log_path (str, optional): Path to the log file. If not provided, a default log file named "TSADEval.log" will be built in current workspace. Defaults to None.
            log_level (str, optional): Log level to set for the logger. Options: "debug", "info", "warning", "error". Defaults to "info".
        """
        self.logger = setup_logger(log_path, level=log_level)
        
        origin_file_path = os.path.abspath(__file__)
        origin_directory = os.path.dirname(origin_file_path)
        origin_cfg_path = os.path.join(origin_directory, "GlobalCfg.toml")
        self.cfg = toml.load(origin_cfg_path)
        if cfg_path is not None:
            self.apply_cfg(cfg_path)
            
        PathManager.del_instance()
        self.pm = PathManager(self.cfg)
        
        # dataset controller
        self.dc = {
            "train_proportion": self.cfg["DatasetSetting"]["train_proportion"],
            "valid_proportion": self.cfg["DatasetSetting"]["valid_proportion"]
        }
        
        # evaluation controller
        self.ec = self.cfg["EvalSetting"]
    
    
    def set_dataset(self, datasets: Union[str, list[str]], dirname=None, dataset_type="UTS", specify_curves=False, curve_names: Union[None, str, list[str]]=None, train_proportion=None, valid_proportion=None):
        """
        Registers the dataset settings and related parameters for the GlobalController instance. This will check if the paths and the parameters are valid.
        
        NOTE: 
         If you want to run all curves in the dataset, set \"specify_curves\" to False. In this mode, \"curve_names\" should be set to None.\n
         Otherwise, if you want to specify some time series in a dataset, please specify ONLY ONE dataset and the curves in this dataset. E.g. set_dataset(datasets=\"WSD\", \"curve_names\"=[\"1\", \"2\"])

        Args:
            datasets (Union[str, list[str]]): Name(s) of the dataset(s) to be set. Can be a single string or a list of strings.
            dirname (str, optional): Path to the dataset directory. If not provided, it will be fetched from the configuration file. Defaults to None.
            curve_type (str, optional): Type of the datasets. Defaults to "UTS".
            specify_curves (bool, optional): Flag indicating whether to specify individual curves within the dataset(s). Defaults to False.
            curve_names (Union[None, str, list[str]], optional): Name(s) of the curve(s) to be used. Can be None, a single string, or a list of strings. Defaults to None.

        Raises:
            ValueError: If the dataset directory path is not specified.
            FileNotFoundError: If the dataset directory or any of the specified datasets or curves do not exist.

        Returns:
            None
        """
        
        # define dataset directory
        if dirname is None:
            if self.cfg["Path"]["dataset"] == "None":
                raise ValueError("Missing Dataset Directory Path. \nPlease specify the dataset directory path either using param: dirname or specifying the config path when building GlobalController instance.")
            dirname = self.cfg["Path"]["dataset"]
            
        dirname = os.path.abspath(dirname)
        if not os.path.exists(dirname):
            raise FileNotFoundError("Dataset Directory %s does not exist."%dirname)
        
        self.pm.load_dataset_path(dirname)
        self.logger.info("Dataset Directory has been loaded.")
        
        if specify_curves and isinstance(datasets, list) and len(dataset) > 1:
            raise TypeError("param datasets must be a string specifying ONLY ONE dataset when specify_curves is True.")
        if isinstance(datasets, list) and curve_names is not None:
            self.logger.warning("Finding Multiple datasets. All curves in these datasets will be employed, and the param \"curve_names\" will be ignored.")
            curve_names = None
            
        datasets = [datasets]
        # check if datasets exists
        for dataset in datasets:
            dataset_path = self.pm.get_dataset_path(dataset_type, dataset)
            if not os.path.exists(dataset_path):
                raise FileNotFoundError("%s does not exist. Please Check the directory path."%dataset_path)
            
        # check if curves exists
        if specify_curves:
            for curve in curve_names:
                curve_path = self.pm.get_dataset_one_curve(dataset_type, datasets[0], curve)
                if not os.path.exists(curve_path):
                    raise FileNotFoundError("%s does not exist. Please Check the directory path."%curve_path)
            
        self.dc["dataset_type"] = dataset_type
        self.dc["datasets"] = datasets
        self.dc["curves"] = curve_names
        self.dc["specify_curves"] = specify_curves
        
        if train_proportion is not None:
            self.dc["train_proportion"] = train_proportion
        if valid_proportion is not None:
            self.dc["valid_proportion"] = valid_proportion
        
    def set_evals(self, evals):
        '''
        Registers the evaluation protocols used for performance evaluations.
        '''
        self.evals = evals
        
    def run_exps(self, method, training_schema, cfg_path=None):
        if training_schema == "one_by_one":
            run_instance = Run(method, )
        elif training_schema
    
    def apply_cfg(self, path=None):
        """
        Applies configuration from a file.

        This method reads a configuration file from the specified path and overrides the corresponding default values.
        NOTE: If no path is provided, it uses a default configuration.

        Args:
            path (str, optional): 
                The path to the configuration file. If None, default configuration is used. Defaults to None.

        Returns:
            None
        """
        if path is None:
            self.logger.warning("Using Default Config %(origin_cfg_path)s.\n\
                ")
            return
        
        path = os.path.abspath(path)
        new_cfg = toml.load(path)
        self.cfg = __update_nested_dict(self.cfg, new_cfg)
        
        self.logger.info("Reload Config Successfully.")
        self.logger.debug(json.dumps(self.cfg, indent=4))