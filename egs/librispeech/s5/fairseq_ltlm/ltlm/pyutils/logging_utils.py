# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
import sys
import logging
import os


def setup_logger(exp_dir=None, log_fn="training.log", stream=sys.stdout, logger_level=logging.INFO, logfile_mode="w"):
    logger = logging.getLogger()
    logger.setLevel(logger_level)
    logger.handlers.clear()

    # default output
    c_handler = logging.StreamHandler(stream)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # Also writes logs to the experiment directory
    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)
        log_path = os.path.join(exp_dir, log_fn)
        f_handler =logging.FileHandler(log_path, mode=logfile_mode, encoding="utf-8")
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
