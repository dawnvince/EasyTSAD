import csv
import toml
import json
import os
from utils.util import build_dir

def get_info(base_path, method, dataset, mode, index_cfg):
    info_path = os.path.join(base_path, method, dataset, mode) + "_avg.json"
    
    metrics = index_cfg[dataset] if dataset in index_cfg else index_cfg["Default"]
    
    if not os.path.exists(info_path):
        return [method] + ["NaN"]*len(metrics)
    with open(info_path, 'r') as f:
        info = json.load(f)
        
    info_to_csv = [method]
    for metric in metrics:
        value = info
        for key in metric:
            value = value[key]
            
        if isinstance(value, float):
            value = round(value, 4)
        info_to_csv.append(value)

    return info_to_csv

def geneTableHead(datasets, index_cfg):
    heads = []
    eval_heads = []
    default_len = len(index_cfg["Default"]) + 1
    for dataset in datasets:
        # write heads
        if dataset in index_cfg:
            heads += [dataset] * (len(index_cfg[dataset]) + 1)
        else:
            heads += [dataset] * default_len
            
        # write eval_heads
        eval_heads += ["Method"]
        metrics = index_cfg[dataset] if dataset in index_cfg else index_cfg["Default"]
        
        for metric in metrics:
            eval_heads.append(metric[0])     
        
    return heads, eval_heads
    

def geneCSV(glo_cfg, task_mode):
    base_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["evaluation_dir"])
    
    csv_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["statistic_dir"])
    csv_path = build_dir(csv_path, "CsvSummary")
    
    sta_cfg_path = "Statistics/config.toml"
    sta_cfg = toml.load(sta_cfg_path)
    
    methods = sta_cfg["Methods"]
    datasets = sta_cfg["Datasets"]
    
    file_name = os.path.join(csv_path, "{}.csv".format(task_mode))
    
    with open(file_name, 'w') as f:
        w = csv.writer(f)
        heads, eval_heads = geneTableHead(datasets, sta_cfg["Statistic_index"])
        w.writerow(heads)
        w.writerow(eval_heads)
        
        for method in methods:
            res = []
            for dataset in datasets:
                res += get_info(base_path, method, dataset, task_mode, sta_cfg["Statistic_index"])
        
            w.writerow(res)
        