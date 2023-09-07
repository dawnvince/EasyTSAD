import numpy as np
import toml
import os
import json
from Analysis.Plot import plot_uts_score_only, plot_uts_score_and_yhat

from DataFactory import LoadData
from utils.util import build_dir, get_method_class, dict_split
from Analysis import Performance

class RunBase(object):
    def __init__(self, method, method_path, gconfig_path, evaluations, task_mode) -> None:
        self.method = method
        self.evaluations = evaluations
        self.task_mode = task_mode
        self.global_cfg = toml.load(gconfig_path)
    
        method_cfg_path = os.path.join(method_path, method, "config.toml")
        self.method_cfg = toml.load(method_cfg_path)
        
        # build related directory for method
        plot_dir = build_dir(
            self.global_cfg["Analysis"]["base_dir"], 
            self.global_cfg["Analysis"]["plot_dir"]
        )
        
        self.use_plot = self.method_cfg["Analysis"]["plot_score"] and self.method_cfg["Analysis"]["plot_y_hat"]
        if self.use_plot == True:
            plot_path = build_dir(plot_dir, method)
            self.plot_path = build_dir(plot_path, task_mode)
        
        eval_dir = build_dir(
            self.global_cfg["Analysis"]["base_dir"], 
            self.global_cfg["Analysis"]["evaluation_dir"]
        )
        self.eval_path = build_dir(eval_dir, method)
        
        score_dir = build_dir(
            self.global_cfg["Analysis"]["base_dir"], 
            self.global_cfg["Analysis"]["score_dir"]
        )
        score_path = build_dir(score_dir, method)
        self.score_path = build_dir(score_path, task_mode)
        
        if self.method_cfg["Analysis"]["plot_y_hat"]:
            y_hat_dir = build_dir(
                self.global_cfg["Analysis"]["base_dir"], 
                self.global_cfg["Analysis"]["y_hat_dir"]
            )
            y_hat_path = build_dir(y_hat_dir, method)
            self.y_hat_path = build_dir(y_hat_path, task_mode)
        
        
    def load_data(self):
        data_params = self.method_cfg["Data_Params"]
        tsDatas = LoadData.all_dataset(
            self.global_cfg["dataset_dir"]["path"],
            data_params["types"], 
            data_params["datasets"], 
            data_params["train_proportion"], 
            data_params["valid_proportion"], 
            data_params["preprocess"]
        )
        return tsDatas
    
    def do_exp(self):
        pass
    
    def do_analysis(self, tsDatas):
        
        for dataset_name, value in tsDatas.items():
            print(">>> [{}] Analyzing dataset {} <<<".format(self.method, dataset_name))
            evaldata_path = build_dir(self.eval_path, dataset_name)            
            eval_dict_path = os.path.join(evaldata_path, "{}.json".format(self.task_mode))
            eval_dict = {}
            avg_json_path = os.path.join(evaldata_path, "{}_avg.json".format(self.task_mode))
            avg_res = None
            avg_len = -1
            
            if self.use_plot:
                if self.method_cfg["Analysis"]["plot_score"]:
                    plot_score_data_path = build_dir(self.plot_path, "score_only")
                    plot_score_data_path = build_dir(plot_score_data_path, dataset_name)
                if self.method_cfg["Analysis"]["plot_y_hat"]:
                    plot_y_hat_data_path = build_dir(self.plot_path, "score_and_y_hat")
                    plot_y_hat_data_path = build_dir(plot_y_hat_data_path, dataset_name)
                    y_hat_dir = os.path.join(self.y_hat_path, dataset_name)
            
            score_dir = os.path.join(self.score_path, dataset_name)
            
            for score_file in os.listdir(score_dir):
                curve_name = score_file[:-4] # delete the ".npy" suffix
                
                score_path = os.path.join(score_dir, score_file)
                score = np.load(score_path)
                
                # Plot curves
                
                if self.method_cfg["Analysis"]["plot_score"]:
                    plot_uts_score_only(
                        value[curve_name].test, 
                        score, 
                        value[curve_name].test_label, 
                        os.path.join(plot_score_data_path, curve_name)
                    )
                    
                if self.method_cfg["Analysis"]["plot_y_hat"]:
                    y_hat_path = os.path.join(y_hat_dir, score_file)
                    y_hat = np.load(y_hat_path)
                    plot_uts_score_and_yhat(
                        value[curve_name].test,
                        y_hat, 
                        score, 
                        value[curve_name].test_label, 
                        os.path.join(plot_y_hat_data_path, curve_name)
                    )
                
                # calculate performance using multiple evaluation methods
                eva = Performance(scores=score, labels=value[curve_name].test_label)
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
                        
                        
    def easy_run(self):
        pass
    
    def offline_analysis(self):
        tsDatas = self.load_data()
        self.do_analysis(tsDatas)
        