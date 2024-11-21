import subprocess
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from datetime import datetime
from tqdm import tqdm
from typing import Dict
import threading
import logging
import time
import os


def setup_logger(log_file_path):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)  # Or another level

    # Use a thread-safe RotatingFileHandler
    handler = RotatingFileHandler(log_file_path, maxBytes=1048576, backupCount=5)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class ProgressTracker:
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.tqdm_bar = tqdm(total=total_tasks, desc="Progress", unit="task")

    def update(self):
        self.tqdm_bar.update(1)

    def close(self):
        self.tqdm_bar.close()


class GPUWorker:
    def __init__(self, gpu_id, function, max_workers, progress_tracker):
        self.gpu_id = gpu_id
        self.function = function
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.progress_tracker = progress_tracker

    def submit(self, task):
        if not self.executor._shutdown:
            future = self.executor.submit(self.function, task)
            future.add_done_callback(lambda x: self.progress_tracker.update())

    def shutdown(self):
        self.executor.shutdown(wait=True)
        print(f"GPU Worker {self.gpu_id} has been shut down.")


class MultiGPUProcessor:
    def __init__(self, function, gpu_ids, max_workers_per_gpu, progress_tracker):
        self.workers = {
            gpu_id: GPUWorker(gpu_id, function, max_workers, progress_tracker)
            for gpu_id, max_workers in zip(gpu_ids, max_workers_per_gpu)
        }

    def distribute_tasks(self, gpu_cmd_dict):
        for gpu_id, cmd_list in gpu_cmd_dict.items():
            for cmd in cmd_list:
                self.workers[gpu_id].submit(cmd)

    def shutdown(self):
        for worker in self.workers.values():
            worker.shutdown()


# 修改 periodic_progress_display 函数来适配进度条的关闭而不是打印
def periodic_progress_display(tracker, interval):
    while tracker.tqdm_bar.n < tracker.total_tasks:
        time.sleep(interval)
    tracker.close()  # 当所有任务完成后关闭进度条


def xc_cmd_deployer(xc_param: Dict, gpu_info: Dict):
    output_dir = xc_param["output_dir"]
    max_workers_per_gpu = gpu_info["gpu_task_num"]
    gpu_mem = gpu_info["gpu_mem_info"]
    gpu_list = gpu_info["gpu_list"]
    debug = xc_param["debug"]
    xc_cmd_file = os.path.join(output_dir, "cmd_list", "xc_cmds.txt")
    logger = setup_logger(xc_param["log_file_path"])

    with open(xc_cmd_file, "r") as f:
        xc_cmd_list = [line.strip() for line in f if line.strip()]

    total_tasks = len(xc_cmd_list)
    progress_tracker = ProgressTracker(total_tasks)

    # Define the function that will be executed by the workers
    def run_cmd(exe_cmd: str):
        result = subprocess.run(
            exe_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        return result.returncode  # 确保返回值正确处理

    def print_cmd(exe_cmd: str):
        print(exe_cmd)
        logger.debug(f"Command: {exe_cmd}")
        return 0  # 假定这里总是成功

    run_or_print = print_cmd if debug else run_cmd

    # Ensure max_workers_per_gpu and gpu_mem_info are of same length
    num_gpus = len(gpu_mem)
    gpu_mem_info = dict(zip(gpu_list, gpu_mem))
    if len(max_workers_per_gpu) < num_gpus:
        # If max_workers_per_gpu is shorter, extend it with 1s
        max_workers_per_gpu.extend([1] * (num_gpus - len(max_workers_per_gpu)))
    elif len(max_workers_per_gpu) > num_gpus:
        # If max_workers_per_gpu is longer, only keep the smallest ones
        max_workers_per_gpu = sorted(max_workers_per_gpu)[:num_gpus]

    # sort the gpu_mem_info by memory size
    sorted_gpus = sorted(gpu_mem_info.keys())
    max_workers_per_gpu = [
        x
        for _, x in sorted(
            zip(gpu_mem_info.values(), max_workers_per_gpu), key=lambda pair: pair[0]
        )
    ]

    # Launch the workers
    processor = MultiGPUProcessor(
        run_or_print, sorted_gpus, max_workers_per_gpu, progress_tracker
    )

    total_mem = sum(gpu_mem_info.values())
    gpu_portions = {gpu_id: mem / total_mem for gpu_id, mem in gpu_mem_info.items()}
    cmd_counts = {
        gpu_id: round(len(xc_cmd_list) * portion)
        for gpu_id, portion in gpu_portions.items()
    }

    cmd_index = 0

    # 启动周期性进度打印线程
    progress_thread = threading.Thread(
        target=periodic_progress_display, args=(progress_tracker, 10)
    )
    progress_thread.start()

    try:
        for gpu_id, count in cmd_counts.items():
            for _ in range(count):
                if cmd_index < total_tasks:
                    cmd = f"{xc_cmd_list[cmd_index]} -G {gpu_id}"
                    processor.workers[gpu_id].submit(cmd)
                    cmd_index += 1

        while cmd_index < total_tasks:
            for gpu_id in sorted_gpus:
                if cmd_index >= total_tasks:
                    break
                cmd = f"{xc_cmd_list[cmd_index]} -G {gpu_id}"
                processor.workers[gpu_id].submit(cmd)
                cmd_index += 1
        # after all the commands are submitted, shutdown the workers
        processor.shutdown()
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{formatted_time}]: Finish doing Cross Correlation !!!.\n")

    except KeyboardInterrupt:
        print("Detected KeyboardInterrupt. Cleaning up...")

        progress_thread.join()
        processor.shutdown()
        print("All processes stopped cleanly.")

    finally:
        # 停止所有工作并清理资源
        progress_thread.join()
        processor.shutdown()
        print("All processes stopped cleanly.")
