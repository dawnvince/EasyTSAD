import os
import logging

def setup_logger(path=None, level="info"):
    """
    Set up a logger with the specified log level and log file.

    Args:
        path (str, optional): Path to the log file. If not provided, a default log file named "TSADEval.log" will be created in the current working directory. Defaults to None.
        level (str, optional): Log level to set for the logger. Options: "debug", "info", "warning", "error". Defaults to "info".

    Returns:
        logging.Logger: Configured logger instance.

    """
    if level == "debug":
        loglevel = logging.DEBUG
    elif level == "info":
        loglevel = logging.INFO
    elif level == "warning":
        loglevel = logging.WARNING
    elif level == "error":
        loglevel = logging.ERROR
    else:
        raise ValueError("Invalid level, one of \"debug\", \"info\", \"warning\", \"error\". Defaults to \"info\".\n")
        
    logger = logging.getLogger("logger")
    logger.setLevel(loglevel)
    if path is None:
        log_path = os.path.join(os.getcwd(), "TSADEval.log")
    else:
        log_path = os.path.abspath(path)
        os.makedirs(log_path)

    fh = logging.FileHandler(log_path)
    fh.setLevel(loglevel)
    
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    
    formatter = logging.Formatter('(%(asctime)s) [%(levelname)s]: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger