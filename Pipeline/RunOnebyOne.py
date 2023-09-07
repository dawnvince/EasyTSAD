import os

import numpy as np

from Pipeline import RunBase
from utils.util import build_dir, get_method_class

class RunOnebyOne(RunBase):
    def __init__(self, method, method_path, gconfig_path, evaluations, task_mode="one_by_one") -> None:
        super().__init__(method, method_path, gconfig_path, evaluations, task_mode)
        
    def do_exp(self, tsDatas):
        for dataset_name, value in tsDatas.items():
            model_params = self.method_cfg["Model_Params"]["Default"]
            if dataset_name in self.method_cfg["Model_Params"]:
                specific_params = self.method_cfg["Model_Params"][dataset_name]
                for k, v in specific_params.items():
                    model_params[k] = v
            
            scoredata_path = build_dir(self.score_path, dataset_name)
            if self.method_cfg["Analysis"]["plot_y_hat"]:
                y_hatdata_path = build_dir(self.y_hat_path, dataset_name)
            
            for curve_name, curve in value.items():
                score_path = os.path.join(scoredata_path, curve_name) + ".npy"
                print(">>> [{}] handling dataset {} | curve {} <<<".format(self.method, dataset_name, curve_name))
                
                ## training & test step
                
                methodclass = get_method_class("Method.{}.{}".format(self.method, self.method), self.method)
                method = methodclass(model_params)
                
                method.train_valid_phase(curve)
                method.test_phase(curve)
                
                score = method.anomaly_score()
                
                ## save scores and y_hat (if exists)
                np.save(score_path, score)
                
                if self.method_cfg["Analysis"]["plot_y_hat"]:
                    y_hat = method.get_y_hat()
                    y_hat_path = os.path.join(y_hatdata_path, curve_name) + ".npy"
                    np.save(y_hat_path, y_hat)
                
                
    
    def easy_run(self):
        tsDatas = self.load_data()
        
        self.do_exp(tsDatas)
        self.do_analysis(tsDatas)
        
    def offline_analysis(self):
        tsDatas = self.load_data()
        self.do_analysis(tsDatas)