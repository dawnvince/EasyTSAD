import numpy as np
import os
import toml
import sys

from DataFactory import TSData

# config_path = "./GlobalConfig.toml"
# config = toml.load(config_path)
# data_dir = config["dataset_dir"]["path"]

def specific_metrics():
    pass

def all_metrics(data_dir, types, dataset, train_proportion:float=1, valid_proportion:float=0, preprocess="min_max"):
    base_path = os.path.join(data_dir, types, dataset)
    
    tsDatas = {}
    for curve in os.listdir(base_path):
        ## Generate TSData instance from the numpy files
        tsData = TSData.buildfrom(types=types, dataset=dataset, data_name=curve, data_dir=data_dir, train_proportion=train_proportion, valid_proportion=valid_proportion)
        
        if preprocess == "min-max":
            tsData.min_max_norm()
        elif preprocess == "z-score":
            tsData.z_score_norm()
        elif preprocess == "raw":
            pass
        elif preprocess is None:
            pass
        else:
            raise ValueError("Unknown preprocess, must be one of min-max, z-score, raw")
        
        tsDatas[curve] = tsData
    
    return tsDatas
        
def all_dataset(data_dir, types, datasets,
                train_proportion:float=1, 
                valid_proportion:float=0, 
                preprocess="min_max"):
    
    print("=== [Load Data] DataSets:", ','.join(datasets), "===")
    tsDatas = {}
    for dataset in datasets:
        tsDatas[dataset] = all_metrics(data_dir, types, dataset, train_proportion, valid_proportion, preprocess)
    
    return tsDatas