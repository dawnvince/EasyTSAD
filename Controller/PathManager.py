import toml
import os
import json

import logging

logger = logging.getLogger("logger")

def build_dir(path1, path2):
    path = os.path.join(path1, path2)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def check_and_build(path):
    if not os.path.exists(path):
        os.makedirs(path)

class PathManager:
    '''
    PathManager manages the paths related to this project. It will automatically build directory for newly introduced methods. Also, you can easily get access to the name of any file you want using the given methods.
    
    NOTE: This class obeys Singleton Pattern. Only one instance exists for global access.
    '''
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PathManager, cls).__new__(cls)
        else:
            logger.error("Multiple PathManager instances. Violate Singleton Pattern.")
        return cls._instance
    
    @staticmethod
    def get_instance():
        return PathManager._instance
    
    @staticmethod
    def del_instance():
        PathManager._instance = None
    
    def __init__(self, glo_cfg) -> None:                
        glo_cfg = glo_cfg["Paths"]
        self.glo_cfg = glo_cfg
        current_dir = os.getcwd()
        
        self.__data_p = None
        self.__res_p = build_dir(current_dir, self.glo_cfg["Results"]["base_dir"])
        
        self.__score_dir = None
        self.__y_hat_dir = None
        self.__plot_dir = None
        self.__runtime_dir = None
        self.__eval_dir = None
        
    def load_dataset_path(self, data_dir):
        self.__data_p = data_dir
            
    def get_dataset_path(self, types, dataset):
        return os.path.join(self.__data_p, types, dataset)
    
    def get_dataset_curves(self, types, dataset):
        return os.listdir(os.path.join(self.__data_p, types, dataset))
    
    def get_dataset_one_curve(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve)
    
    def get_dataset_train_set(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve, "train.npy")
    
    def get_dataset_test_set(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve, "test.npy")
    
    def get_dataset_train_label(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve, "train_label.npy")
    
    def get_dataset_test_label(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve, "test_label.npy")
    
    def get_dataset_train_timestamp(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve, "train_timestamp.npy")
    
    def get_dataset_test_timestamp(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve, "test_timestamp.npy")
    
    def get_dataset_info(self, types, dataset, curve):
        return os.path.join(self.__data_p, types, dataset, curve, "info.json")
    
    def get_eval_json_all(self, method, schema, dataset):
        if self.__eval_dir == None:
            self.__eval_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["eval_dir"])
        path = os.path.join(self.__eval_dir, method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "all.json")
    
    def get_eval_json_avg(self, method, schema, dataset):
        if self.__eval_dir == None:
            self.__eval_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["eval_dir"])
        path = os.path.join(self.__eval_dir, method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "avg.json")
    
    def get_score_curves(self, method, schema, dataset):
        return os.listdir(os.path.join(self.__score_dir, method, schema, dataset))
        
    def get_score_path(self, method, schema, dataset, curve_name):
        if self.__score_dir == None:
            self.__score_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["score_dir"])
        path = os.path.join(self.__score_dir, method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "%s.npy"%curve_name)
    
    def get_yhat_path(self, method, schema, dataset, curve_name):
        if self.__y_hat_dir == None:
            self.__y_hat_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["y_hat_dir"])
        path = os.path.join(self.__y_hat_dir, method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "%s.npy"%curve_name)
    
    def get_plot_path_score_only(self, method, schema, dataset, curve_name):
        if self.__plot_dir == None:
            self.__plot_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["plot_dir"])
        path = os.path.join(self.__plot_dir, "score_only", method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "%s.pdf"%curve_name)
    
    def get_plot_path_with_yhat(self, method, schema, dataset, curve_name):
        if self.__plot_dir == None:
            self.__plot_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["plot_dir"])
        path = os.path.join(self.__plot_dir, "with_yhat", method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "%s.pdf"%curve_name)
    
    ## AggerX plots a figure containing all curves in a dataset for one method
    def get_plot_path_aggreX(self, method, schema, dataset):
        if self.__plot_dir == None:
            self.__plot_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["plot_dir"])
        path = os.path.join(self.__plot_dir, "AggregationX", method, schema)
        check_and_build(path)
        return os.path.join(path, "%s.pdf"%dataset)
    
    ## AggerY plots a figure containing all methods for one curve
    def get_plot_path_aggreY(self, schema, dataset, curve_name):
        if self.__plot_dir == None:
            self.__plot_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["plot_dir"])
        path = os.path.join(self.__plot_dir, "AggregationY", schema, dataset)
        check_and_build(path)
        return os.path.join(path, "%s.pdf"%curve_name)
    
    def get_rt_time_path(self, method, schema, dataset):
        if self.__runtime_dir == None:
            self.__runtime_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["runtime_dir"])
        path = os.path.join(self.__runtime_dir, method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "time.json")
    
    def get_rt_statistic_path(self, method, schema, dataset):
        if self.__runtime_dir == None:
            self.__runtime_dir = build_dir(self.__res_p, self.glo_cfg["Results"]["runtime_dir"])
        path = os.path.join(self.__runtime_dir, method, schema, dataset)
        check_and_build(path)
        return os.path.join(path, "model_statistic.txt")
    
    def get_default_config_path(self, method):
        cur_path = os.path.abspath(__file__)
        cur_dir = os.path.dirname(cur_path)
        cur_dir = os.path.dirname(cur_dir)
        cur_path = os.path.join(cur_dir, "Methods", method, "config.toml")
        return cur_path
    
    def check_valid(self, path, msg):
        if not os.path.exists(path):
            raise FileNotFoundError(msg)