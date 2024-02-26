from EasyTSAD.Controller import TSADController

def run_only_once(gctrl, methods, training_schema):
    """============= [EXPERIMENTAL SETTINGS] ============="""
    # Specifying methods and training schemas
    
    for method in methods:
        # run models
        gctrl.run_exps(
            method=method,
            training_schema=training_schema
        )
       
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze")
        ]
    )

    for method in methods:
        gctrl.do_evals(
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
    
    from EasyTSAD.Methods import AE, Donut, AR
    
    methods = ["AR", "AE"]
    training_schema = "naive"
    
    # If your have run this function and don't change the params, you can skip this step.
    run_only_once(gctrl=gctrl, methods=methods, training_schema=training_schema)
    
    
    """============= [Aggregation Plots] ============="""
    gctrl.summary.plot_aggreY(
        types=dataset_types,
        datasets=datasets,
        methods=methods,
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
    
    
