import contextlib
import functools
import logging
import os
import platform
import sys
import traceback
from logging import StreamHandler
from logging.handlers import RotatingFileHandler

import numpy as np

date_format = "%Y-%m-%d-%H-%M-%S"
logging_format = "%(asctime)s - [PID%(process)d] %(message)s"
# logging_format = "%(hostname)s %(asctime)s - [PID%(process)d] %(message)s"


@contextlib.contextmanager
def local_seed(seed):
    """Context manager for using a specified numpy seed within a block."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_logger_instance(filename):
    """Generate a logger instance."""
    root_logger = logging.getLogger('causal-discovery-dv')
    root_logger.propagate = False
    root_logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt=logging_format, datefmt=date_format)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    return root_logger