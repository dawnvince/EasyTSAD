import os
import toml
import numpy as np
import logging
import json

from Controller import PathManager
from DataFactory import TSData, LoadData

logger = logging.getLogger("logger")

class BaseSchema(object):
    def __init__(self, dc, ec, method, schema:str, cfg_path:str=None, diff_order:int=None, preprocess:str=None) -> None:
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
        pass
    
    def do_eval(self, tsDatas, evals):
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
                eva = Performance(scores=score, labels=value[curve_name].test_label, margins=margins)
                if eva.all_label_normal:
                    continue
                res, res_dict = eva.do_eval(self.evaluations)
                
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