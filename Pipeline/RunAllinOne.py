import json
import os

import numpy as np

from Pipeline import RunBase
from utils.util import build_dir, get_method_class

class RunAllinOne(RunBase):
    def __init__(self, method, method_path, gconfig_path, evaluations, task_mode="all_in_one") -> None:
        super().__init__(method, method_path, gconfig_path, evaluations, task_mode)
        
    def do_exp(self, tsDatas):
        for dataset_name, value in tsDatas.items():
            model_params = self.method_cfg["Model_Params"]["Default"]
            if dataset_name in self.method_cfg["Model_Params"]:
                specific_params = self.method_cfg["Model_Params"][dataset_name]
                for k, v in specific_params.items():
                    model_params[k] = v
            
            timerdata_path = build_dir(self.time_path, dataset_name)
            self.train_valid_timer.reset_total()
            self.test_timer.reset_total()
            
            scoredata_path = build_dir(self.score_path, dataset_name)
            if self.method_cfg["Analysis"]["plot_y_hat"]:
                y_hatdata_path = build_dir(self.y_hat_path, dataset_name)
            
            print("\n>>>[{}] training dataset {}<<<".format(self.method, dataset_name))
            methodclass = get_method_class(
                "Method.{}.{}".format(self.method, self.method), self.method
            )
            method = methodclass(model_params)
            
            ## training and validation phase
            self.train_valid_timer.tic()
            method.train_valid_phase_all_in_one(tsTrains=value)
            self.train_valid_timer.toc()
            
            for curve_name, curve in value.items():
                score_path = os.path.join(scoredata_path, curve_name) + ".npy"
                
                ## test phase
                self.test_timer.tic()
                method.test_phase(curve)
                self.test_timer.toc()
                
                score = method.anomaly_score()
                
                ## save scores and evaluation
                np.save(score_path, score)
                
                if self.method_cfg["Analysis"]["plot_y_hat"]:
                    y_hat = method.get_y_hat()
                    y_hat_path = os.path.join(y_hatdata_path, curve_name) + ".npy"
                    np.save(y_hat_path, y_hat)
                    
            # save running time info
            time_path = os.path.join(timerdata_path, self.task_mode) + ".json"
            time_dict = {
                "train_and_valid": self.train_valid_timer.get_total_time(),
                "test": self.test_timer.get_total_time()
            }
            with open(time_path, 'w') as f:
                json.dump(time_dict, f, indent=4)
                
                
    
    def easy_run(self):
        tsDatas = self.load_data()
        
        self.do_exp(tsDatas)
        self.do_analysis(tsDatas)
        