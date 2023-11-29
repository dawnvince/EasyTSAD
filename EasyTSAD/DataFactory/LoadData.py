from . import TSData
from ..Controller import PathManager
import logging

logger = logging.getLogger("logger")    

def __load_all_curve_in_dataset(types, dataset, train_proportion:float=1, valid_proportion:float=0, preprocess="z-score", diff_p=0):
    """
    Retrieves and preprocesses all time series in the specific dataset.

    Args:
        types (str): 
            UTS or MTS
        dataset (str): 
            Name of the dataset.
        train_proportion (float, optional): 
            Proportion of data to use for training. Defaults to 1 (uses all data).
        valid_proportion (float, optional): 
            Proportion of data to use for validation. Defaults to 0 (no validation data).
        preprocess (str, optional): 
            Preprocessing method to apply to the data. Options: "min-max", "z-score", "raw", None.
            Defaults to "z-score".
        diff_p (int, optional): 
            Order of differencing to apply to the data. Defaults to 0 (no differencing).

    Returns:
        dict: Dictionary containing TSData instances for each metric, with metric names as keys.

    Raises:
        ValueError: If an unknown preprocessing method is specified.

    """
    pm = PathManager.get_instance()
    curves = pm.get_dataset_curves(types, dataset)
    
    tsDatas = {}
    for curve in curves:
        ## Generate TSData instance from the numpy files
        tsData = TSData.buildfrom(types=types, dataset=dataset, data_name=curve, train_proportion=train_proportion, valid_proportion=valid_proportion)
        
        if diff_p > 0 and isinstance(diff_p, int):
            tsData.differential(diff_p)
        
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


def __load_all_datasets(types, datasets,
                train_proportion:float=1, 
                valid_proportion:float=0, 
                preprocess="min_max", diff_p=0):
    """
    Loads and preprocesses time series data from multiple datasets.

    Args:
        types (str): 
            UTS or MTS
        datasets (list): 
            List of dataset names to load.
        train_proportion (float, optional): 
            Proportion of data to use for training. Defaults to 1 (uses all data).
        valid_proportion (float, optional): 
            Proportion of data to use for validation. Defaults to 0 (no validation data).
        preprocess (str, optional): 
            Preprocessing method to apply to the data. Options: "min-max", "z-score", "raw", None.
            Defaults to "z-score".
        diff_p (int, optional): 
            Order of differencing to apply to the data. Defaults to 0 (no differencing).

    Returns:
        dict: Dictionary containing TSData instances for each dataset, with dataset names as keys.

    """
    logger.info("    [Load Data (All)] DataSets: %s "%(','.join(datasets)))
    tsDatas = {}
    for dataset in datasets:
        tsDatas[dataset] = __load_all_curve_in_dataset(types, dataset, train_proportion, valid_proportion, preprocess, diff_p)
    
    return tsDatas

def __load_specific_curves(types, dataset, curve_names, train_proportion:float=1, valid_proportion:float=0, preprocess="z-score", diff_p=0):
    """
    Retrieves and preprocesses specific time series data from a dataset.

    Args:
        types (str): 
            UTS or MTS
        dataset (str): 
            Name of the dataset.
        curve_names (list[str]):
            Name of the curves.
        train_proportion (float, optional): 
            Proportion of data to use for training. Defaults to 1 (uses all data).
        valid_proportion (float, optional): 
            Proportion of data to use for validation. Defaults to 0 (no validation data).
        preprocess (str, optional): 
            Preprocessing method to apply to the data. Options: "min-max", "z-score", "raw", None.
            Defaults to "z-score".
        diff_p (int, optional): 
            Order of differencing to apply to the data. Defaults to 0 (no differencing).

    Returns:
        dict: Dictionary containing TSData instances for each dataset, with dataset names as keys.

    Raises:
        ValueError: If an unknown preprocessing method is specified.

    """
    logger.info("    [Load Data (Specify)] DataSets: %s "%(dataset))
    tsDatas = {dataset: {}}
    for curve in curve_names:
        ## Generate TSData instance from the numpy files
        tsData = TSData.buildfrom(types=types, dataset=dataset, data_name=curve, train_proportion=train_proportion, valid_proportion=valid_proportion)
        
        if diff_p > 0 and isinstance(diff_p, int):
            tsData.differential(diff_p)
        
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
        
        tsDatas[dataset][curve] = tsData
    
    return tsDatas


def load_data(dc, preprocess, diff_p):
    """
    Loads the data based on the specified configuration.

    Args:
        - `use_diff` (bool, optional): Whether to use differential order. Defaults to True.

    Returns:
        dict: A dictionary containing the loaded time series data.
    """
    
    if dc["specify_curves"]:
        tsDatas = __load_specific_curves(
            types=dc["dataset_type"],
            dataset=dc["datasets"][0],
            curve_names=dc["curves"],
            train_proportion=dc["train_proportion"], 
            valid_proportion=dc["valid_proportion"],
            preprocess=preprocess,
            diff_p=diff_p
        )
        return tsDatas

    else:
        tsDatas = __load_all_datasets(
            types=dc["dataset_type"],
            datasets=dc["datasets"],
            train_proportion=dc["train_proportion"], 
            valid_proportion=dc["valid_proportion"],
            preprocess=preprocess,
            diff_p=diff_p
        )
        return tsDatas