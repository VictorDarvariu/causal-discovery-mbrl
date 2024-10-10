import ast
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy
import numpy as np
import pandas as pd

date_format = "%Y-%m-%d-%H-%M-%S"

def get_memory_usage_str():
    """Get the memory usage in megabytes as a string."""
    import psutil
    mb_used = psutil.Process(os.getpid()).memory_info().vms / 1024 ** 2
    return f"Process memory usage: {mb_used} MBs."

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_current_time_millis():
    """Return epoch time in milliseconds."""
    return int(time.time() * 1000)

def print_time_from(dt):
    """Print time elapsed from a datetime object."""
    started_str = dt.strftime(date_format)
    print(f"started at {started_str}")

    experiment_ended_datetime = datetime.now()
    ended_str = experiment_ended_datetime.strftime(date_format)
    print(f"ended at {ended_str}")
    print(f"took {(experiment_ended_datetime - dt).total_seconds(): .3f} seconds.")

class NpEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder to enable encoding objects containing NumPy datatypes.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def read_file_as_np_array(file_path):
    file_pathlib = Path(file_path)
    if file_pathlib.suffix == ".npy":
        np_arr = np.load(file_path)
    elif file_pathlib.suffix == ".csv":
        np_arr = pd.read_csv(file_path, header=None).values
    else:
        raise ValueError(f"Unsupported data file type {file_pathlib.suffix}.")

    return np_arr