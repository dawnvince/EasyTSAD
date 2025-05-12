from typing import Dict
import numpy as np
import os
from EasyTSAD.Controller import TSADController

if __name__ == "__main__":
    
    # Create a global controller
    gctrl = TSADController()
        
    """============= [DATASET SETTINGS] ============="""
    # Specifying datasets
    datasets = ["SWaT"]
    dataset_types = "MTS"
    
    # set datasets path, dirname is the absolute/relative path of dataset.
    
    # Use all curves in datasets:
    gctrl.set_dataset(
        dataset_type="MTS",
        dirname="/home/safari/EasyTSAD/datasets",
        datasets=datasets
    )


    """============= Impletment your algo. ============="""
    from EasyTSAD.Methods import MTSExample
    from EasyTSAD.Methods import CAD
    
    """============= Run your algo. ============="""
    # Specifying methods and training schemas
    training_schema = "mts"
    method = "CAD"  # Use CAD Method

    gctrl.run_exps(
        method=method,
        training_schema=training_schema,
        preprocess="raw",
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