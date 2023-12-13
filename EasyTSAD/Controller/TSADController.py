import csv
import numpy as np
import os
import toml
import json
from typing import Union

from ..Evaluations import Performance
from ..utils import update_nested_dict
from ..Plots.plot import plot_uts_score_only

from .logger import setup_logger
from .PathManager import PathManager
from ..DataFactory.LoadData import load_data
from ..TrainingSchema import AllInOne, ZeroShot, OneByOne, ZeroShotCrossDS
from ..Summary import Summary

class TSADController:
    '''
    TSADController class represents a controller that manages global configuration and logging.
    
    Attributes:
        summary (EasyTSAD.Summary): summary the trained results, including generating CSV and aggregating all methods' anomaly scores on specific curve in one plot.
    '''
    
    def __init__(self, cfg_path=None, log_path=None, log_level="info") -> None:
        """
        TSADController class represents a controller that manages global configuration and logging.

        Args:
            cfg_path (str, optional): Path to the configuration file. If provided, the configuration will be applied from this file. Defaults to None (Not Recommanded).
            log_path (str, optional): Path to the log file. If not provided, a default log file named "TSADEval.log" will be built in current workspace. Defaults to None.
            log_level (str, optional): Log level to set for the logger. Options: "debug", "info", "warning", "error". Defaults to "info".
        """
        
        self.logger = setup_logger(log_path, level=log_level)
        self.logger.info("""
                         
███████╗ █████╗ ███████╗██╗   ██╗    ████████╗███████╗ █████╗ ██████╗ 
██╔════╝██╔══██╗██╔════╝╚██╗ ██╔╝    ╚══██╔══╝██╔════╝██╔══██╗██╔══██╗
█████╗  ███████║███████╗ ╚████╔╝        ██║   ███████╗███████║██║  ██║
██╔══╝  ██╔══██║╚════██║  ╚██╔╝         ██║   ╚════██║██╔══██║██║  ██║
███████╗██║  ██║███████║   ██║          ██║   ███████║██║  ██║██████╔╝
╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝          ╚═╝   ╚══════╝╚═╝  ╚═╝╚═════╝ 
                                                                      
                         """)
        
        origin_file_path = os.path.abspath(__file__)
        origin_directory = os.path.dirname(origin_file_path)
        origin_cfg_path = os.path.join(origin_directory, "GlobalCfg.toml")
        self.cfg = toml.load(origin_cfg_path)
        if cfg_path is not None:
            self.apply_cfg(cfg_path)
            
        PathManager.del_instance()
        self.pm = PathManager(self.cfg)
        self.summary = Summary()
        
        # dataset controller
        self.dc = {
            "train_proportion": self.cfg["DatasetSetting"]["train_proportion"],
            "valid_proportion": self.cfg["DatasetSetting"]["valid_proportion"]
        }
        
        self.dc["Transfer"] = self.cfg["Transfer"]
    
    
    def set_dataset(self, datasets: Union[str, list[str]], dirname=None, dataset_type="UTS", specify_curves=False, curve_names: Union[None, str, list[str]]=None, train_proportion=None, valid_proportion=None):
        """
        Registers the dataset settings and related parameters for the TSADController instance. This will check if the paths and the parameters are valid.
        
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
                raise ValueError("Missing Dataset Directory Path. \nPlease specify the dataset directory path either using param: dirname or specifying the config path when building TSADController instance.")
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
            
        if isinstance(datasets, str):
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
            
    def spilt_dataset_for_zero_shot_cross(self, src, dst):
        if not all(item in self.dc["datasets"] for item in src):
            raise ValueError("The param \"src\" must be the subset of the assigned \"datasets\"")
        if not all(item in self.dc["datasets"] for item in dst):
            raise ValueError("The param \"dst\" must be the subset of the assigned \"datasets\"")
            
        if type(src) is not list or None:
            raise TypeError("The param \"src_datasets\" must be None or the list of str.")
        if type(dst) is not list or None:
            raise TypeError("The param \"dst_datasets\" must be None or the list of str.")
            
        self.dc["src_datasets"] = src
        self.dc["dst_datasets"] = dst
    
        
    def run_exps(self, method, training_schema, cfg_path=None, diff_order=None, preprocess=None, hparams=None):
        """
        Run experiments using the specified method and training schema.

        Args:
            method (str): The method being used.
            training_schema (str): The training schema being used. One of one_by_one, all_in_one, zero_shot, zero_shot_cross_ds.
            cfg_path (str, optional): Path to a custom configuration file. Defaults to None.
            diff_order (int, optional): The differential order. Defaults to None.
            preprocess (str, optional): The preprocessing method. Options: "raw", "min-max", "z-score". Defaults to None (equals to "raw"). 
            hparams (dict, optional): Hyperparameters for the model. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If the specified training schema is not one of "one_by_one", "all_in_one", or "zero_shot".
        """
        
        self.logger.info("Run Experiments. Method[{}], Schema[{}].".format(method, training_schema))
        
        if training_schema == "one_by_one":
            run_instance = OneByOne(self.dc, method, cfg_path, diff_order, preprocess)
        elif training_schema == "all_in_one":
            run_instance = AllInOne(self.dc, method, cfg_path, diff_order, preprocess)
        elif training_schema == "zero_shot":
            run_instance = ZeroShot(self.dc, method, cfg_path, diff_order, preprocess)
        elif training_schema == "zero_shot_cross_ds":
            run_instance = ZeroShotCrossDS(self.dc, method, cfg_path, diff_order, preprocess)
        else:
            raise ValueError("Unknown \"training_schema\", must be one of one_by_one, all_in_one, zero_shot\n")
        
        tsDatas = run_instance.load_data()
        run_instance.do_exp(tsDatas=tsDatas, hparams=hparams)
        
    def set_evals(self, evals):
        '''
        Registers the evaluation protocols used for performance evaluations.
        
        Args:
            evals (list[EvalInterface]): The evaluation instances inherited from EvalInterface.
        '''
        self.logger.info("Register evaluations")
        self.evals = evals
        
        # evaluation controller
        self.ec = self.cfg["EvalSetting"]
        
    def do_evals(self, method, training_schema):
        """
        Performing evaluations based on saved anomaly scores. The result will be saved in Results/Evals, including the detailed evaluation results and the average evaluation results.
        
        Args:
            method (str): The method being used.
            training_schema (str): The training schema being used.
        """
        self.logger.info("Perform evaluations. Method[{}], Schema[{}].".format(method, training_schema))
        tsDatas = load_data(
            self.dc,
            preprocess="raw",
            diff_p=0
        )
        
        if self.dc["dst_datasets"] is not None:
            new_tsDatas = {}
            for dataset_name, value in tsDatas.items():
                if dataset_name in self.dc["dst_datasets"]:
                    new_tsDatas[dataset_name] = value
            
            tsDatas = new_tsDatas
        
        for dataset_name, value in tsDatas.items():
            self.logger.info("    [{}] Eval dataset {} <<<".format(method, dataset_name))
            eval_dict = {}
            avg_res = None
            avg_len = -1
            
            margins = (0, 0)
            if self.ec["use_margin"] == True:
                if dataset_name in self.ec["margin"]:
                    margins = (self.ec["margin"][dataset_name][0], self.ec["margin"][dataset_name][1])
                    self.logger.info("        [{}] Using margins {}".format(dataset_name, margins))
                else:
                    margins = (self.ec["margin"]["default"][0], self.ec["margin"]["default"][1])
                    self.logger.info("        [{}] Using default margins {}".format(dataset_name, margins))
               
            score_files = self.pm.get_score_curves(method, training_schema, dataset_name)
            eval_dict_path = self.pm.get_eval_json_all(method, training_schema, dataset_name)
            avg_json_path = self.pm.get_eval_json_avg(method, training_schema, dataset_name)
            
            for score_file in score_files:
                curve_name = score_file[:-4] # delete the ".npy" suffix
                
                score_path = self.pm.get_score_path(method, training_schema, dataset_name, curve_name)
                score = np.load(score_path)
                
                # calculate performance using multiple evaluation methods
                eva = Performance(method, dataset_name, curve_name, scores=score, labels=value[curve_name].test_label, margins=margins)
                
                # SKIP the curves without anomalies
                if eva.all_label_normal:
                    continue
                
                res, res_dict = eva.perform_eval(self.evals)
                
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
    
    def plots(self, method, training_schema):
        """
        Generate plots for the specified method and training schema. The plots are located in Results/Plots.

        Args:
            method (str): Method name.
            training_schema (str): Training schema name.

        Returns:
            None

        """
        self.logger.info("Plotting. Method[{}], Schema[{}].".format(method, training_schema))
        tsDatas = load_data(
            self.dc,
            preprocess="raw",
            diff_p=0
        )
        
        if self.dc["dst_datasets"] is not None:
            new_tsDatas = {}
            for dataset_name, value in tsDatas.items():
                if dataset_name in self.dc["dst_datasets"]:
                    new_tsDatas[dataset_name] = value
            
            tsDatas = new_tsDatas
            
        for dataset_name, value in tsDatas.items():
            self.logger.info("    [{}] Plot dataset {} score only ".format(method, dataset_name))
            
            score_files = self.pm.get_score_curves(method, training_schema, dataset_name)
            for score_file in score_files:
                curve_name = score_file[:-4] # delete the ".npy" suffix
                
                score_path = self.pm.get_score_path(method, training_schema, dataset_name, curve_name)
                score = np.load(score_path)
                save_path = self.pm.get_plot_path_score_only(method, training_schema, dataset_name, curve_name)
                plot_uts_score_only(
                    curve=value[curve_name].test, 
                    score=score, 
                    label=value[curve_name].test_label, 
                    save_path=save_path
                )
        
        # To be implemented 
        plot_yhat = None
        magic_number = 21647942    
        if plot_yhat == magic_number:
            for dataset_name, value in tsDatas.items():
                self.logger.info("    [{}] Plot dataset {} with yhat ".format(method, dataset_name))
                
                score_files = self.pm.get_score_curves(method, training_schema, dataset_name)
                for score_file in score_files:
                    curve_name = score_file[:-4] # delete the ".npy" suffix
                    
                    score_path = self.pm.get_score_path(method, training_schema, dataset_name, curve_name)
                    score = np.load(score_path)
                    save_path = self.pm.get_plot_path_score_only(method, training_schema, dataset_name, curve_name)
                    plot_uts_score_only(
                        curve=value[curve_name].test, 
                        score=score, 
                        label=value[curve_name].test_label, 
                        save_path=save_path
                    )
        
        
    
    def apply_cfg(self, path=None):
        """
        Applies configuration from a file.

        This method reads a configuration file from the specified path and overrides the corresponding default values.
        
        NOTE: 
            If no path is provided, it uses a default configuration.

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
        self.cfg = update_nested_dict(self.cfg, new_cfg)
        
        self.logger.info("Reload Config Successfully.")
        self.logger.debug(json.dumps(self.cfg, indent=4))
    