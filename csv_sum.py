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

def do_evals(gctrl, methods, training_schemas):
        
    """============= [EVALUATION SETTINGS] ============="""
    
    from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA, EventKthF1PA, EventKthPrcPA, PointAuprcPA, EventPrcPA, PointKthF1PA
    # Specifying evaluation protocols
    gctrl.set_evals(
        [
            PointF1PA(),
            EventF1PA(),
            EventF1PA(mode="squeeze"),
            
            PointKthF1PA(3),
            PointKthF1PA(10),
            PointKthF1PA(20),
            PointKthF1PA(50),
            PointKthF1PA(150),
            PointAuprcPA(),
            EventKthPrcPA(3, mode="raw"),
            EventKthPrcPA(10, mode="raw"),
            EventKthPrcPA(20, mode="raw"),
            EventKthPrcPA(50, mode="raw"),
            EventKthPrcPA(150, mode="raw"),
            
            EventKthF1PA(3, mode="log", base=3),
            EventKthF1PA(10, mode="log", base=3),
            EventKthF1PA(20, mode="log", base=3),
            EventKthF1PA(50, mode="log", base=3),
            EventKthF1PA(150, mode="log", base=3),
            EventPrcPA(mode="log", base=3),
            EventKthPrcPA(3, mode="log", base=3),
            EventKthPrcPA(10, mode="log", base=3),
            EventKthPrcPA(20, mode="log", base=3),
            EventKthPrcPA(50, mode="log", base=3),
            EventKthPrcPA(150, mode="log", base=3),
            
            EventKthF1PA(3, mode="squeeze"),
            EventKthF1PA(10, mode="squeeze"),
            EventKthF1PA(20, mode="squeeze"),
            EventKthF1PA(50, mode="squeeze"),
            EventKthF1PA(150, mode="squeeze"),
            EventPrcPA(mode="squeeze"),
            EventKthPrcPA(3, mode="squeeze"),
            EventKthPrcPA(10, mode="squeeze"),
            EventKthPrcPA(20, mode="squeeze"),
            EventKthPrcPA(50, mode="squeeze"),
            EventKthPrcPA(150, mode="squeeze"),
        ]
    )

    for method in methods:
        for training_schema in training_schemas:
            gctrl.do_evals(
                method=method,
                training_schema=training_schema
            )
        
        
if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
    
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["AIOPS", "NAB", "TODS", "WSD", "Yahoo", "UCR", "NEK"]
    dataset_types = "UTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="UTS",
        dirname="datasets",
        datasets=datasets,
    )
    
    from EasyTSAD.Methods import *
    
    methods = ["AR", "LSTMADalpha", "LSTMADbeta", "AE", "EncDecAD", "Donut", "FCVAE", "TimesNet", "SRCNN", "TFAD", "OFA", "AnomalyTransformer", "FITS", "DCdetector", "SubLOF", "TranAD", "MatrixProfile", "SAND"]
    schemas = ["naive", "all_in_one", "zero_shot"]
    
    # do_evals(gctrl, methods, schemas)
    
    # If your have run this function and don't change the params, you can skip this step.
    # run_only_once(gctrl=gctrl, methods=methods, training_schema=training_schema)
    
    
    """============= [Aggregation Plots] ============="""
    # gctrl.summary.plot_aggreY(
    #     types=dataset_types,
    #     datasets=datasets,
    #     methods=methods,
    #     training_schema=training_schema
    # )
    
    """============= Generate CSVs ============="""
    for training_schema in schemas:
        gctrl.summary.to_csv(
            datasets=datasets,
            methods=methods,
            training_schema=training_schema,
            eval_items=[
                ["best f1 under pa", "f1"],
                ["event-based f1 under pa with mode log", "f1"],
                ["event-based f1 under pa with mode squeeze", "f1"],
                ["event-based f1 under 3-delay pa with mode log", "f1"],
                ["event-based f1 under 10-delay pa with mode log", "f1"],
                ["event-based f1 under 20-delay pa with mode log", "f1"],
                ["event-based f1 under 50-delay pa with mode log", "f1"],
                ["event-based f1 under 150-delay pa with mode log", "f1"],
                
                ["point-based auprc pa"],
                ["event-based auprc under pa with mode log"],
                ["event-based auprc under pa with mode squeeze"],
                ["3-th auprc under event-based pa with mode log"],
                ["10-th auprc under event-based pa with mode log"],
                ["20-th auprc under event-based pa with mode log"],
                ["50-th auprc under event-based pa with mode log"],
                ["150-th auprc under event-based pa with mode log"],
            ]
        )
    
    
