import os
import json
import csv
import logging

import numpy as np

from .Controller import PathManager
from .Plots import plot_uts_summary_aggreY, plot_uts_summary_aggreX

class Summary:
    def __init__(self) -> None:
        self.pm = PathManager.get_instance()
        self.logger = logging.getLogger("logger")

        
    def to_csv(self, datasets, methods, training_schema, eval_items):
        """
        Generates a CSV file based on the provided datasets, methods, training schema, and evaluation items.

        Args:
            datasets (list): List of dataset names.
            methods (list): List of method names.
            training_schema (str): Training schema name.
            eval_items (list): List of evaluation items, where each item is a list containing the path to the final value in Eval JSONs, 
                e.g. [
                        ["event-based f1 under pa with mode log", "f1"],
                        ["best f1 under pa", "f1"]
                    ]

        Returns:
            None

        Raises:
            FileNotFoundError: If the JSON file for a specific method, dataset, and training schema does not exist.

        """
        self.logger.info("Generating CSV file for Method[{}], Datasets[{}], Schema[{}].".format(', '.join(methods), ', '.join(datasets), training_schema))
        self.logger.info("[CSV] Containing Eval items: [{}]".format(', '.join([item[0] for item in eval_items])))
        
        csv_path = self.pm.get_csv_path(training_schema)
        eval_len = len(eval_items)
        
        def gen_table_head():
            first_heads = [" "]
            second_heads = ["Methods"]
            for dataset in datasets:
                # generate first head
                first_heads += [dataset] * eval_len
            
                # generate second head
                for item in eval_items:
                    second_heads.append(item[0])
                    
            return first_heads, second_heads
    
        def gen_each_line(method):
            content = [method]
            for dataset in datasets:
                json_path = self.pm.get_eval_json_avg(method, training_schema, dataset, build=False)
                if not os.path.exists(json_path):
                    self.logger.error("File Not Founded when creating csv. Fil path: {}".format(json_path))
                    content += ["Not Founded"] * eval_len
                    continue
                
                with open(json_path, 'r') as f:
                    evals = json.load(f)
                
                for eval_item in eval_items:
                    value = evals
                    for key in eval_item:
                        value = value[key]
                        
                    if isinstance(value, float):
                        value = round(value, 4)
                        
                    content.append(value)
            
            return content
                    
        with open(csv_path, "w") as f:
            w = csv.writer(f)
            first_head, second_head = gen_table_head()
            w.writerow(first_head)
            w.writerow(second_head)
            
            for method in methods:
                w.writerow(gen_each_line(method))
    
    ## AggerX plots a figure containing all curves in a dataset for one method 
    
    ## Problem: Generated Plots are too big.    
    def __plot_aggreX(self, types, datasets, methods, training_schema):
        
        def aggreX_scores(dataset, method, curve_names):
            scores = []
            labels = []
            raws = []
            for curve_name in curve_names:
                curve_path = self.pm.get_dataset_test_set(types, dataset, curve_name)
                raws.append(np.load(curve_path))
                
                label_path = self.pm.get_dataset_test_label(types, dataset, curve_name)
                labels.append(np.load(label_path))
                
                score_path = self.pm.get_score_path(method, training_schema, dataset, curve_name, build=False)
                if os.path.exists(score_path):
                    scores.append(np.load(score_path))
                else:
                    self.logger.warning("   [{}] [{}]-{}: Score File Not Founded".format(method, dataset, curve_name))
                    scores.append(None)
                    
            return raws, scores, labels
                
        
        for method in methods:
            for dataset in datasets:
                plot_path = self.pm.get_plot_path_aggreX(method, training_schema, dataset)
                curve_names = self.pm.get_dataset_curves(types, dataset)
                
                raws, scores, labels = aggreX_scores(dataset, method, curve_names)
                
                plot_uts_summary_aggreX(raws, scores, labels, plot_path, curve_names)
                
                
                
                
    def plot_aggreY(self, types, datasets, methods, training_schema):
        """
        Plots aggregated anomaly scores for the specified types, datasets, methods, and training schema.

        Args:
            types (str): Types of the data.
            datasets (list): List of dataset names.
            methods (list): List of method names.
            training_schema (str): Training schema name.

        Returns:
            None

        """
        self.logger.info("Plotting Aggregated Anomaly scores for Method[{}], Datasets[{}], Schema[{}].".format(', '.join(methods), ', '.join(datasets), training_schema))
        def aggreY_scores(dataset, methods, curve_name):
            scores = []
            
            curve_path = self.pm.get_dataset_test_set(types, dataset, curve_name)
            raw = np.load(curve_path)
            
            label_path = self.pm.get_dataset_test_label(types, dataset, curve_name)
            label = np.load(label_path)
            
            for method in methods:           
                score_path = self.pm.get_score_path(method, training_schema, dataset, curve_name, build=False)
                if os.path.exists(score_path):
                    scores.append(np.load(score_path))
                else:
                    self.logger.warning("   [{}] [{}]-{}: Score File Not Founded".format(method, dataset, curve_name))
                    scores.append(None)
                    
            return raw, scores, label
        
        for dataset in datasets:
            curve_names = self.pm.get_dataset_curves(types, dataset)
            self.logger.info("  Plotting [{}]".format(dataset))
            for curve_name in curve_names:
                plot_path = self.pm.get_plot_path_aggreY(training_schema, dataset, curve_name)
                raw, scores, label = aggreY_scores(dataset, methods, curve_name)
                
                plot_uts_summary_aggreY(raw, scores, label, plot_path, methods)
            