import numpy as np
import os
import argparse

import warnings

import toml
import Pipeline
from Analysis import Evaluations
from Statistics import geneCSV, genePlots, geneCDFs, geneDistributionsEachCurve, geneDistributionSummary

warnings.filterwarnings('ignore')

evaluations = [
    Evaluations.BestF1underPA(),
    Evaluations.KthBestF1underPA(10),
    Evaluations.KthBestF1underPA(20),
    Evaluations.KthBestF1underPA(50),
    Evaluations.KthBestF1underPA(150),
    Evaluations.EventF1PA(mode="log", base=3),
    Evaluations.EventKthF1PA(10, mode="log", base=3),
    Evaluations.EventKthF1PA(20, mode="log", base=3),
    Evaluations.EventKthF1PA(50, mode="log", base=3),
    Evaluations.EventKthF1PA(150, mode="log", base=3),
    Evaluations.EventRocPA(mode="log", base=3),
    Evaluations.EventPrcPA(mode="log", base=3),
    Evaluations.EventF1PA(mode="squeeze"),
    Evaluations.EventKthF1PA(10, mode="squeeze"),
    Evaluations.EventKthF1PA(20, mode="squeeze"),
    Evaluations.EventKthF1PA(50, mode="squeeze"),
    Evaluations.EventKthF1PA(150, mode="squeeze"),
    Evaluations.AurocPA(),
    Evaluations.AuprcPA(),
    Evaluations.EventRocPA(mode="log", base=3),
    Evaluations.EventPrcPA(mode="log", base=3),
    Evaluations.EventRocPA(mode="squeeze"),
    Evaluations.EventPrcPA(mode="squeeze"),
    Evaluations.EventDetect(),
]

# evaluations = [
#     Evaluations.EventKthF1PA(1, mode="squeeze"),
#     Evaluations.EventKthF1PA(50, mode="squeeze"),
#     Evaluations.EventKthF1PA(150, mode="squeeze"),
#     Evaluations.EventRocPA(mode="squeeze"),
#     Evaluations.EventPrcPA(mode="squeeze"),
# ]

# eval_items = [i.get_name() for i in evaluations]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for running')
    parser.add_argument('--method', type=str, help='The name of the method', default=None)
    parser.add_argument('--method_path', default="Method", type=str, help='The absolute path of file Method')
    parser.add_argument('--task_mode', default="one_by_one", type=str, help='Task mode', choices=["one_by_one", "all_in_one", "transfer_within_dataset"])
    parser.add_argument('--behavior', default="run", type=str, help='do exp or not', choices=["run", "analysis_only", "csv_summary", "plot_summary", "cdf_summary", "dis_summary", "dis_each_curve"])
    
    
    args = parser.parse_args()
    
    glo_cfg_path = "./GlobalConfig.toml"
    glo_cfg = toml.load(glo_cfg_path)
    
    if args.behavior == "csv_summary":
        geneCSV(glo_cfg, args.task_mode)
        
    elif args.behavior == "plot_summary":
        genePlots(glo_cfg, args.task_mode)
        
    elif args.behavior == "cdf_summary":
        geneCDFs(glo_cfg, args.task_mode)
        
    elif args.behavior == "dis_each_curve":
        geneDistributionsEachCurve(glo_cfg, args.task_mode)
    
    elif args.behavior == "dis_summary":
        geneDistributionSummary(glo_cfg, args.task_mode)
    
    elif args.behavior == "run" or args.behavior == "analysis_only":
        method_name = args.method
        method_path = args.method_path
        
        if args.task_mode == "one_by_one":
            easy_run = Pipeline.RunOnebyOne(method_name, method_path, glo_cfg, evaluations)
        elif args.task_mode == "all_in_one":
            easy_run = Pipeline.RunAllinOne(method_name, method_path, glo_cfg, evaluations)
        elif args.task_mode == "transfer_within_dataset":
            easy_run = Pipeline.RunTransInDataset(method_name, method_path, glo_cfg, evaluations)
            
        if args.behavior == "run":
            easy_run.easy_run()
        elif args.behavior == "analysis_only":
            easy_run.offline_analysis()
    
    