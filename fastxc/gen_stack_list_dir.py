from .SeisHandler import SeisArray
from typing import Dict, List, Tuple
from multiprocessing import Pool
import logging
import itertools
from tqdm import tqdm
from datetime import datetime
from contextlib import contextmanager
from datetime import datetime, timedelta
import os


def convert_to_datetime(date_str):
    """
    Converts a date string of a specific format into a datetime object.

    The input date string should be in the format 'year.day_of_year.hourminute'
    (e.g., '2023.045.2359').
    Logs an error and returns None if the string format is incorrect.

    Parameters:
    date_str (str): The date string to convert.

    Returns:
    datetime: The corresponding datetime object.
    """
    try:
        # split string to get year jday and time infomation
        parts = date_str.split(".")
        year = int(parts[0])
        day_of_year = int(parts[1])
        time_part = parts[2]
        hour = int(time_part[:2])
        minute = int(time_part[2:])
        dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        return datetime(year, dt.month, dt.day, hour, minute)
    except ValueError as e:
        logging.error(f"datetime convertine error {date_str}: {e}")


@contextmanager
def suppress_logging():
    """
    A context manager for temporarily suppressing logging.

    Within this context manager, all logs below the CRITICAL level will not be output.
    The previous logging level is restored upon exit from the context.

    Example Usage:
    with suppress_logging():
        # Perform operations here that would generate log output; they will be suppressed.
    """
    prev_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(prev_level)


def get_all_sta_comp(
    segspec_dir: str, double_array: bool
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Retrieves all station-component pairs from a given segspec directory.

    This function discerns and organizes station-component pairs into two lists:
    source station-component pairs and receiver station-component pairs,
    based on the provided segspec files. If set to single-array mode and
    the receiver station list is empty, the source station list will be copied to it.

    Parameters:
    segspec_dir (str): The directory containing segspec files.
    double_array (bool): Whether to process dual-array data.

    Returns:
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: A tuple containing two lists,
    one for source station-component pairs and the other for receiver station-component pairs.
    """
    pattern = "{home}/{network}/{*}/{station}.{YYYY}.{JJJ}.{HH}{MI}.{component}.segspec"

    with suppress_logging():
        spec_array = SeisArray(array_dir=segspec_dir, pattern=pattern)
        spec_array.match()

    spec_array.group(labels=["network", "station", "component"], filtered=False)
    array_files = spec_array.files_group

    src_component_pairs = []
    sta_component_pairs = []
    for file_info in array_files:
        network, station, component = file_info
        if network == "array1":
            src_component_pairs.append((station, component))
        if network == "array2":
            sta_component_pairs.append((station, component))
    # if
    if sta_component_pairs == [] and not double_array:
        sta_component_pairs = src_component_pairs

    return src_component_pairs, sta_component_pairs


# Write stacking list for stacking calculation
def gen_stack_list(info_pair, ncf_dir, time_list, stack_list_dir, double_array: bool):
    """
    Generates a stack list file for stacking calculations.

    Based on the information pair for the source and receiver stations, it generates a list
    of file paths that may exist in the ncf directory. These paths are written into a new
    text file for use in subsequent stacking calculations.

    Parameters:
    info_pair (Tuple): A tuple containing the source and receiver station information.
    ncf_dir (str): The directory containing .sac files.
    time_list (List[str]): A list containing time labels.
    stack_list_dir (str): The directory to store stack lists.
    double_array (bool): Whether to process dual-array data.

    No return value, but a file will be created or updated.
    """
    all_path = []
    src_info, sta_info = info_pair
    src_name, src_component = src_info
    sta_name, sta_component = sta_info

    # In single array condition, there will be no src_name > sta_name
    # e.g. only "AAKH-YCWH"; no "YCWH-AAKH"
    # if not double_array and src_name > sta_name:
    #     return

    file_name = f"{src_name}-{sta_name}.{src_component}-{sta_component}.sac"
    for time_label in time_list:
        possible_path = os.path.join(ncf_dir, time_label, file_name)
        if os.path.exists(possible_path):
            all_path.append(possible_path)

    if all_path == []:
        return

    stack_list_name = f"{src_name}-{sta_name}.{src_component}-{sta_component}.stack.txt"
    stack_list = os.path.join(stack_list_dir, stack_list_name)

    with open(stack_list, "w") as f:
        for path in all_path:
            f.write(path + "\n")
    return


def gen_stack_list_dir(xc_param: Dict, SeisArrayInfo: Dict):
    """
    Generates stack list directories based on cross-correlation parameters and seismic array configuration.

    Retrieves station-component information pairs from the segspec directory and filters
    the relevant time labels from the ncf directory according to the specified start and end times.
    It then calls gen_stack_list using the filtered time labels and station-component pairs
    to generate stack lists.

    Parameters:
    xc_param (Dict): Contains parameters for cross-correlation calculations.
    executing (Dict): A dictionary containing execution information.
    seis_array_1 (Dict): Contains the configuration for the first seismic array.

    No return value, but multiple files will be created for subsequent processing.
    """

    start_str = SeisArrayInfo["start"]
    end_str = SeisArrayInfo["end"]
    output_dir = xc_param["output_dir"]
    thread_num = xc_param["cpu_count"]

    start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")

    segspec_dir = os.path.join(output_dir, "segspec")

    ncf_dir = os.path.join(output_dir, "ncf")

    stack_list_dir = os.path.join(output_dir, "stack_list")
    os.makedirs(stack_list_dir, exist_ok=True)

    double_array = SeisArrayInfo["sac_dir_2"] != "NONE"
    src_info, sta_info = get_all_sta_comp(segspec_dir, double_array)

    time_list_raw = os.listdir(ncf_dir)

    time_list = []

    for time_label in time_list_raw:
        time_object = convert_to_datetime(time_label)
        if time_object >= start and time_object <= end:
            time_list.append(time_label)

    tasks = [
        (info_pair, ncf_dir, time_list, stack_list_dir, double_array)
        for info_pair in itertools.product(src_info, sta_info)
    ]

    def update_bar(*a):
        # This function will be called after the finish of each process,
        # to update the progress bar
        pbar.update()

    # create a progress bar
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{formatted_time}]: Creating stacking lists \n")

    pbar = tqdm(total=len(tasks), desc="[Creating stacking lists]")
    # creating a parallel pool
    with Pool(processes=thread_num) as pool:
        results = []
        for task in tasks:
            result = pool.apply_async(gen_stack_list, args=task, callback=update_bar)
            results.append(result)

        # waiting for compolete of all tasks
        for result in results:
            result.wait()

    # close progress bar
    pbar.close()
    formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{formatted_time}]: Finish creating stacking lists \n")
