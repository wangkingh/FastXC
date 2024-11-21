import subprocess
from typing import List, Dict
import logging
from multiprocessing import Pool
from logging.handlers import RotatingFileHandler
from tqdm import tqdm
from datetime import datetime
import os


# PART0: UTILS
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



#def run_or_print_cmd(cmd: str, debug: bool = False, logger=None):
#    if logger is None:
#        print("Logger is not initialized.")
#        return
#
#    if debug:
#        logger.debug(f"Command (not executed in debug mode): {cmd}")
#    else:
#        try:
#            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
#            if result.returncode == 0:
#                logger.info(result.stdout)
#            else:
#                logger.error(result.stderr)
#        except Exception as e:
#            logger.error(f"An unexpected error occurred: {e}")


def run_or_print_cmd(cmd, debug):
    if debug:
        print(f"Would run: {cmd}")
    else:
        os.system(cmd)



def read_cmds_from_file(file_path: str) -> List[str]:
    try:
        with open(file_path, "r") as f:
            cmds = [line.strip() for line in f if line.strip()]
        return cmds
    except FileNotFoundError:
        logging.error(f"The file {file_path} was not found.")
        return []


def worker(cmd_debug):
    cmd, debug = cmd_debug
    run_or_print_cmd(cmd, debug)
    return True


def execute_in_batches(pool, cmds: List[str], batch_size: int, debug, information):
    start_index = 0
    total_cmds = len(cmds)
    while start_index < total_cmds:
        end_index = min(start_index + batch_size, total_cmds)
        current_batch = [(cmd, debug) for cmd in cmds[start_index:end_index]]
        list(tqdm(pool.imap_unordered(worker, current_batch), total=len(current_batch), desc=information))
        start_index = end_index

def execute_commands(pool, cmds: List[str], debug, information):
    current_batch = [(cmd, debug) for cmd in cmds]
    for _ in tqdm(pool.imap_unordered(worker, current_batch), total=len(current_batch), desc=information):
        pass

def stack_cmd_deployer(xc_param: Dict):
    output_dir = xc_param["output_dir"]
    cmd_list_file = os.path.join(output_dir, "cmd_list", "stack_cmds.txt")
    debug = xc_param["debug"]
    parallel = xc_param["parallel"]
    threads = xc_param["cpu_count"]
    stack_cmds = read_cmds_from_file(cmd_list_file)
    if parallel:
        with Pool(processes=threads) as pool:
            # execute_in_batches(pool, stack_cmds, 5000, debug, "[Stacking]")  # 调整批处理大小
            execute_commands(pool, stack_cmds, debug, "[Stacking]")
    else:
        for cmd in tqdm(stack_cmds, desc="[Stacking]"):
            run_or_print_cmd(cmd, debug)

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Finish doing STACK .\n")

def rotate_cmd_deployer(xc_param: Dict):
    output_dir = xc_param["output_dir"]
    debug = xc_param["debug"]
    parallel = xc_param["parallel"]
    threads = xc_param["cpu_count"]
    rotate_cmds = read_cmds_from_file(os.path.join(output_dir, "cmd_list", "rotate_cmds.txt"))
    if parallel:
        with Pool(processes=threads) as pool:
            execute_in_batches(pool, rotate_cmds, 5000, debug, "[Rotating]")  # 调整批处理大小
    else:
        for cmd in tqdm(rotate_cmds, desc="[Rotating]"):
            run_or_print_cmd(cmd, debug)

    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Finish doing ROTATE !!!.\n")
