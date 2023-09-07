import os
import importlib
import random
import math
from typing import Dict

def build_dir(path1, path2):
    path = os.path.join(path1, path2)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def get_method_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def dict_split(src:Dict, proportion: float, seed=1):
    keys = list(src.keys())
    random.seed(seed)
    random.shuffle(keys)
    split_idx = math.ceil(proportion * len(keys))
    print("=== dataset is split ===")
    print("training set is ", keys[:split_idx])
    print("test set is ", keys[split_idx:])
    
    d1, d2 = {}, {}
    for k in keys[:split_idx]:
        d1[k] = src[k]
    for k in keys[split_idx:]:
        d2[k] = src[k]
        
    return d1, d2
    