import ast
import os
import time
from datetime import datetime
import json
import numpy as np

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
