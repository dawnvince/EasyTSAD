from EasyTSAD.Controller import TSADController

def run_only_once(gctrl, methods, training_schema):
    """============= [EXPERIMENTAL SETTINGS] ============="""
    for method in methods:
        # run models
        gctrl.run_exps(
            method=method,
            training_schema=training_schema
        )

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
    
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
    dataset_types = "UTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="/path/to/datasets",
        datasets=datasets,
    )

    # Specifying methods and training schemas
    from EasyTSAD.Methods import AE, Donut, AR
    
    methods = ["AR", "AE", "Donut"]
    training_schema = "naive"
    
    # If your have run this function and don't change the params, you can skip this step.
    run_only_once(gctrl, methods, training_schema)
       
        
    """============= Implement your evaluations ============="""
    
    from EasyTSAD.Evaluations import EvalInterface, MetricInterface
    from dataclasses import dataclass
    
    """
    First, implement `Metric` class inherited from `MetricInterface` for easy statistics.
    
    NOTE: If your metric is based on F1, auprc or auroc, you can use them by introducing related packages:
        from EasyTSAD.Evaluations.Metrics import F1class, Auprc, Auroc
    """
    # if your metric is not based on F1, auprc or auroc, please define Metric first.
    @dataclass
    class NewMetric(MetricInterface):
        name:str
        num = 1
        value_1:float
        value_2:float
        ...
        
        def add(self, other):
            self.value_1 += other.value_1
            self.value_2 += other.value_2
            self.num += 1
            
        def avg(self):
            if self.num != 0:
                self.value_1 /= self.num
                self.value_2 /= self.num
                
        def to_dict(self):
            return {
                self.name: {
                    "value_1": self.value_1,
                    "value_2": self.value_2
                }
            }
        
    '''
    Then, define your Evaluation protocols inherited from `EvalInterface`.
    '''
    class YourEval(EvalInterface):
        def __init__(self, param1) -> None:
            super().__init__()
            self.name = "Your eval name"
            self.param1 = param1
            
        def calc(self, scores, labels, margins) -> type[MetricInterface]:
            value_1 = ...
            value_2 = ...
            return NewMetric(
                self.name,
                value_1=value_1,
                value_2=value_2
            )
    
    '''
    Finally, perform evaluations on offline score files.
    
    If all score files exist, you needn't to call run_exps again when performing new evaluations.
    '''
    # Specifying evaluation protocols
    param1 = 0
    
    gctrl.set_evals(
        [
            YourEval(param1),
            ...
        ]
    )

    for method in methods:
        gctrl.do_evals(
            method=method,
            training_schema=training_schema
        )
    
    """============= Generate CSVs ============="""
    gctrl.summary.to_csv(
        datasets=datasets,
        methods=methods,
        training_schema=training_schema,
        eval_items=[
            ["Your eval name", "value_1"]
        ]
    )
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    for method in methods:
        gctrl.plots(
            method=method,
            training_schema=training_schema
        )
