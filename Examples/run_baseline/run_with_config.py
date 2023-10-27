import sys
sys.path.append("/home/sht/aiops/TestTSAD")

from EasyTSAD.Controller import TSADController
'''
We recommend to use toml file to config the experiments. For baselines' config files' specific configuration, please refer to our Github.

'''
if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
    
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["TODS", "UCR", "AIOPS", "NAB", "Yahoo", "WSD"]
    datasets = ["TODS"]
    dataset_types = "UTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    gctrl.set_dataset(
        datasets=datasets,
        dirname="../../datasets"
    )
    
    
    """============= [EXPERIMENTAL SETTINGS] ============="""
    # Specifying methods and training schemas
    from EasyTSAD.Methods import AE, Donut, AR
    
    method = "AR"
    training_schema = "one_by_one"

    # run models
    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        cfg_path="{}_config.toml".format(method),
        cuda=True
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

    gctrl.do_evals(
        method=method,
        training_schema=training_schema
    )
        
        
    """============= [PLOTTING SETTINGS] ============="""
    
    # plot anomaly scores for each curve
    gctrl.plots(
        method=method,
        training_schema=training_schema
    )
