import numpy as np
import os
import argparse

import warnings
import Pipeline
from Analysis import Evaluations
warnings.filterwarnings('ignore')

evaluations = [
    Evaluations.BestF1underPA(),
    Evaluations.KthBestF1underPA(10),
    Evaluations.KthBestF1underPA(20),
    Evaluations.KthBestF1underPA(30),
    Evaluations.AurocPA(),
    Evaluations.AuprcPA()
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for running')
    parser.add_argument('--method', type=str, help='The name of the method')
    parser.add_argument('--method_path', default="Method", type=str, help='The absolute path of file Method')
    parser.add_argument('--task_mode', default="one_by_one", type=str, help='Task mode', choices=["one_by_one", "all_in_one", "transfer_within_dataset"])
    parser.add_argument('--behavior', default="run", type=str, help='do exp or not', choices=["run", "analysis_only"])
    
    
    args = parser.parse_args()
    
    glo_cfg_path = "./GlobalConfig.toml"
    method_name = args.method
    method_path = args.method_path
    
    if args.task_mode == "one_by_one":
        easy_run = Pipeline.RunOnebyOne(method_name, method_path, glo_cfg_path, evaluations)
    elif args.task_mode == "all_in_one":
        easy_run = Pipeline.RunAllinOne(method_name, method_path, glo_cfg_path, evaluations)
    elif args.task_mode == "transfer_within_dataset":
        easy_run = Pipeline.RunTransInDataset(method_name, method_path, glo_cfg_path, evaluations)
        
    if args.behavior == "run":
        easy_run.easy_run()
    elif args.behavior == "analysis_only":
        easy_run.offline_analysis()
    
    