import subprocess
import threading
import shlex
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
import logging
from threading import Event
from logging.handlers import RotatingFileHandler
from .sac2spec_pyv.main import sac2spec_pyv
import time
import glob
from datetime import datetime
import os


def setup_logger(log_file_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:  # 避免重复添加handler
        handler = RotatingFileHandler(log_file_path, maxBytes=1048576, backupCount=5)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def run_or_print_cmd(cmd: str, debug: bool = False, logger=None):
    if logger is None:
        print("Logger is not initialized.")
        return

    if debug:
        logger.debug(f"Command (not executed in debug mode): {cmd}")
    else:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(result.stdout)
            else:
                logger.error(result.stderr)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")


def read_cmds_from_file(file_path: str) -> List[str]:
    try:
        with open(file_path, "r") as f:
            cmds = [line.strip() for line in f if line.strip()]
        return cmds
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
        return []


def check_and_print_file_count(base_dir, stop_event, interval, total_tasks):
    """
    Periodically check and print the number of files in the given directory (recursively).

    Args:
    - base_dir (str): Directory to check.
    - interval (int): Time interval (in seconds) to wait between checks.
    - stop_event (threading.Event): An event that signals the thread to stop.
    """
    while not stop_event.is_set():
        file_count = sum([len(files) for _, _, files in os.walk(base_dir)])
        print(
            f"[{datetime.now()}]: Current number of spectrums written out: {file_count}/{total_tasks}"
        )
        time.sleep(interval)


def sac2spec_cmd_deployer(xc_param: Dict):
    log_file_path = xc_param["log_file_path"]
    output_dir = xc_param["output_dir"]
    debug = xc_param["debug"]
    parallel = xc_param["parallel"]
    segspec_dir = os.path.join(output_dir, "segspec")
    cmd_list_file = os.path.join(output_dir, "cmd_list", "sac2spec_cmds.txt")
    logger = setup_logger(log_file_path, "main_logger")
    cmds = read_cmds_from_file(cmd_list_file)

    sac_spec_list_dir = os.path.join(output_dir, "sac_spec_list")
    sac_lists = glob.glob(sac_spec_list_dir + "/*sac_list*")
    total_tasks = sum(1 for sac_list in sac_lists for line in open(sac_list))

    # Create an Event object
    stop_event = Event()

    # Start the thread to check and print file count
    check_thread = threading.Thread(
        target=check_and_print_file_count,
        args=(segspec_dir, stop_event, 10, total_tasks),
    )
    check_thread.start()

    if xc_param["norm_special"] != "PYV":
        ######### CUDA VERSION ################
        if parallel:
            # parallel execution
            with ThreadPoolExecutor(max_workers=len(cmds)) as executor:
                for cmd in cmds:
                    executor.submit(run_or_print_cmd, cmd, debug, logger)
        else:
            # serial execution
            for cmd in cmds:
                run_or_print_cmd(cmd, debug, logger)
        ########### END OF CUDA VERSION ################
    else:
        # PYV version
        params = []
        for cmd in cmds:
            # 使用shlex分割命令行为列表，保持引号内内容不分割
            parts = shlex.split(cmd)
            # 剩余部分是参数
            params.append(" ".join(parts[1:]))
        for param in params:
            if debug:
                print(f"Command (not executed in debug mode): {param}")
            else:
                sac2spec_pyv(param)
        # end of PYV version

    # Set the event to signal the thread to stop
    stop_event.set()
    check_thread.join()

    # Print the final file count
    file_count = sum([len(files) for _, _, files in os.walk(segspec_dir)])
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Total number of spectrums written out: {file_count}")

    print(f"\n[{formatted_time}]: Finish doing SAC2SPEC !.\n")
