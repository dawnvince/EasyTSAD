import os
import numpy as np
from matplotlib import pyplot as plt
import toml

from utils.util import build_dir
from Analysis.Plot import plot_uts_summary, plot_cdf_summary

def aggre_scores(score_path, dataset, curve_name, mode, methods):
    scores = []
    for method in methods:
        score_path_t = os.path.join(score_path, method, mode, dataset, curve_name) + ".npy"
        if os.path.exists(score_path_t):
            scores.append(np.load(score_path_t))
        else:
            scores.append(None)
    return scores


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