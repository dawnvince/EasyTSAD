import json
import os
import numpy as np
from matplotlib import pyplot as plt
import toml

from utils.util import build_dir
from Analysis.Plot import plot_uts_summary, plot_cdf_summary, plot_distribution_each_curve, plot_distribution_datasets

def aggre_scores(score_path, dataset, curve_name, mode, methods):
    scores = []
    for i in range(len(methods)):
        score_path_t = os.path.join(score_path, methods[i], mode, dataset, curve_name) + ".npy"
        if os.path.exists(score_path_t):
            scores.append(np.load(score_path_t))
        else:
            scores.append(None)
    return scores

def aggre_thresholds(eval_path, dataset, mode, methods):
    json_dict = {}
    for method in methods:
        json_path_t = os.path.join(eval_path, method, dataset, mode) + ".json"
        if os.path.exists(json_path_t):
            with open(json_path_t, 'r') as f:
                json_dict[method] = json.load(f)
        else:
            json_dict[method] = None
    return json_dict
        
def get_value_in_dict(dic, idx_list):
    if dic is None:
        return 0
    value = dic
    for key in idx_list:
        value = value[key]
            
    if isinstance(value, float):
        value = round(value, 10)
    return value

def genePlots(glo_cfg, task_mode):
    score_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["score_dir"])
    
    plot_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["statistic_dir"])
    plot_path = build_dir(plot_path, "PlotsSummary")
    
    sta_cfg_path = "Statistics/config.toml"
    sta_cfg = toml.load(sta_cfg_path)
    
    methods = sta_cfg["Methods"]
    datasets = sta_cfg["Datasets"]
    
    for dataset in datasets:
        plotdata_path = build_dir(plot_path, dataset)
        data_dir = os.path.join(glo_cfg["dataset_dir"]["path"], "UTS", dataset)
        curve_names = os.listdir(data_dir)
        for curve_name in curve_names:
            plotcurve_dir = build_dir(plotdata_path, curve_name)
            plotcurve_path = os.path.join(plotcurve_dir, task_mode)
            
            label_path = os.path.join(data_dir, curve_name, "test_label.npy")
            curve_path = os.path.join(data_dir, curve_name, "test.npy")
            label = np.load(label_path)
            curve = np.load(curve_path)
            
            scores = aggre_scores(score_path, dataset, curve_name, task_mode, methods)
            
            plot_uts_summary(curve, scores, label, plotcurve_path, methods)
            
            
def geneCDFs(glo_cfg, task_mode):
    score_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["score_dir"])
    
    plot_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["statistic_dir"])
    plot_path = build_dir(plot_path, "CDFSummary")
    
    sta_cfg_path = "Statistics/config.toml"
    sta_cfg = toml.load(sta_cfg_path)
    
    methods = sta_cfg["Methods"]
    datasets = sta_cfg["Datasets"]
    
    for dataset in datasets:
        plotdata_path = build_dir(plot_path, dataset)
        data_dir = os.path.join(glo_cfg["dataset_dir"]["path"], "UTS", dataset)
        curve_names = os.listdir(data_dir)
        for curve_name in curve_names:
            plotcurve_dir = build_dir(plotdata_path, curve_name)
            plotcurve_path = os.path.join(plotcurve_dir, task_mode)
            
            label_path = os.path.join(data_dir, curve_name, "test_label.npy")
            label = np.load(label_path)
            
            scores = aggre_scores(score_path, dataset, curve_name, task_mode, methods)
            
            plot_cdf_summary(scores, label, plotcurve_path, methods)
            
def geneDistributionsEachCurve(glo_cfg, task_mode):
    score_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["score_dir"])
    
    plot_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["statistic_dir"])
    plot_path = build_dir(plot_path, "DistributionEachCurve")
    
    sta_cfg_path = "Statistics/config.toml"
    sta_cfg = toml.load(sta_cfg_path)
    
    methods = sta_cfg["Methods"]
    datasets = sta_cfg["Datasets"]
    thres_source = sta_cfg["Thresholds"]["Source"]
    
    eval_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["evaluation_dir"])
    
    for dataset in datasets:
        evals = aggre_thresholds(eval_path, dataset, task_mode, methods)
        
        plotdata_path = build_dir(plot_path, dataset)
        data_dir = os.path.join(glo_cfg["dataset_dir"]["path"], "UTS", dataset)
        curve_names = os.listdir(data_dir)
        for curve_name in curve_names:
            # collect thresholds
            thresholds = []
            for method in methods:
                if evals[method] is None:
                    thresholds.append(0)
                else:
                    thresholds.append(get_value_in_dict(evals[method][curve_name], thres_source))
            
            plotcurve_dir = build_dir(plotdata_path, curve_name)
            plotcurve_path = os.path.join(plotcurve_dir, task_mode)
            
            label_path = os.path.join(data_dir, curve_name, "test_label.npy")
            label = np.load(label_path)
            
            scores = aggre_scores(score_path, dataset, curve_name, task_mode, methods)
            
            plot_distribution_each_curve(scores, thresholds, label, plotcurve_path, methods)
            

def geneDistributionSummary(glo_cfg, task_mode):
    score_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["score_dir"])
    
    plot_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["statistic_dir"])
    plot_path = build_dir(plot_path, "DistributionSummary")
    
    sta_cfg_path = "Statistics/config.toml"
    sta_cfg = toml.load(sta_cfg_path)
    
    methods = sta_cfg["Methods"]
    datasets = sta_cfg["Datasets"]
    thres_source = sta_cfg["Thresholds"]["Source"]
    
    eval_path = build_dir(glo_cfg["Analysis"]["base_dir"], glo_cfg["Analysis"]["evaluation_dir"])
    
    for dataset in datasets:
        scores_dict = {}
        thresholds_dict = {}
        labels_dict = {}
        
        data_dir = os.path.join(glo_cfg["dataset_dir"]["path"], "UTS", dataset)
        curve_names = os.listdir(data_dir)
        
        for curve_name in curve_names:
            label_path = os.path.join(data_dir, curve_name, "test_label.npy")
            labels_dict[curve_name] = np.load(label_path)
        
        for method in methods:
            scores_dict[method] = {}
            thresholds_dict[method] = {}
            
            json_path_t = os.path.join(eval_path, method, dataset, task_mode) + ".json"
            if os.path.exists(json_path_t):
                with open(json_path_t, 'r') as f:
                    json_t = json.load(f)
            else:
                json_t = None
            
            for curve_name in curve_names:
                score_path_t = os.path.join(score_path, method, task_mode, dataset, curve_name) + ".npy"
                if os.path.exists(score_path_t):
                    scores_dict[method][curve_name] = np.load(score_path_t)
                else:
                    scores_dict[method][curve_name] = None
                
                if json_t:
                    thresholds_dict[method][curve_name] = get_value_in_dict(json_t[curve_name], thres_source)
        
        
        plotdata_path = build_dir(plot_path, dataset)
        plot_curve_path = os.path.join(plotdata_path, task_mode)
        plot_distribution_datasets(scores_dict, thresholds_dict, labels_dict, save_path=plot_curve_path, methods=methods)
        