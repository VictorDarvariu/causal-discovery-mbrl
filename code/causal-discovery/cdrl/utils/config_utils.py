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
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.INFO)

    has_stdout = False
    has_file = False
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            has_file = True
        if isinstance(handler, StreamHandler):
            has_stdout = True

    formatter = logging.Formatter(fmt=logging_format, datefmt=date_format)

    if not has_stdout:
        sh = StreamHandler(sys.stdout)
        sh.addFilter(HostnameFilter())
        sh.setFormatter(formatter)
        root_logger.addHandler(sh)

    if filename is not None and not has_file:
        fh = RotatingFileHandler(filename,
                                 maxBytes=32*1024*1024,
                                 backupCount=10)
        fh.addFilter(HostnameFilter())
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    return root_logger


class HostnameFilter(logging.Filter):
    hostname = platform.node()
    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True