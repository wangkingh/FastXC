import logging
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler, printing to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(console_handler)


def filter_file(file_info: Dict, criteria_with_time: Optional[List[tuple]] = None,
                criteria_without_time: Optional[Dict[str, List]] = None) -> Optional[Dict]:
    """
    Filters a single file based on time range and additional criteria.
    Returns file_info [a dict] if the file is valid, otherwise returns None.
    """
    if criteria_without_time is not None:
        for key, values in criteria_without_time.items():
            if file_info.get(key) not in values:
                return None

    # Check if the file's time falls within any of the time intervals
    if criteria_with_time is not None:
        file_time = file_info.get('time')  # time is a datetime object
        if not any(start_time <= file_time <= end_time for start_time, end_time in criteria_with_time):
            return None

    return file_info


def filter_files(file_list: List[Dict],
                 criteria: Optional[Dict[str, List]] = None,
                 num_threads: int = 1):
    """
    Filters a list of files based on time range and additional criteria.
    Uses multiprocessing for parallel filtering.
    """
    if criteria is None:
        criteria = {}
    if not file_list:
        logger.warning("No files provided for filtering.")
        return []
    sample_dict = file_list[0]

    # Check if criteria is valid
    for key, values in criteria.items():
        if not isinstance(values, list):
            logger.error(f"Criteria values for {key} must be a list")
            return []

    keys_to_delete = [key for key in criteria if key not in sample_dict]
    for key in keys_to_delete:
        logger.error(f"Criteria key {key} not found in file info. Will be deleted.")
        del criteria[key]


    criteria_with_time = None
    criteria_without_time = None

    # Separate 'time' criteria and validate before processing files
    if criteria is not None:
        criteria_without_time = {k: v for k, v in criteria.items() if k != "time"}

        if 'time' in criteria:
            ... # do nothing
            criteria_time = criteria['time']

            # Ensure criteria_time is even, if it is odd, discard the last one
            if len(criteria_time) % 2 != 0:
                criteria_time = criteria_time[:-1]

            # Convert criteria_time to datetime and group into tuple pairs
            criteria_with_time = []
            
            for i in range(0, len(criteria_time), 2):
                try:
                    start_time = datetime.strptime(criteria_time[i], "%Y-%m-%d %H:%M:%S")
                    end_time = datetime.strptime(criteria_time[i + 1], "%Y-%m-%d %H:%M:%S")
                    criteria_with_time.append((start_time, end_time))
                except ValueError:
                    logger.error(f"Invalid date format for {criteria_time[i]} or {criteria_time[i + 1]}, "
                                 "expected format is '%Y-%m-%d %H:%M:%S'")
                    return False

    if criteria:
        logger.info(f"Filtering files based on criteria: {criteria}")

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Apply the function to each file info dictionary
        all_results = list(filter(None, executor.map(
            lambda file_info: filter_file(file_info, criteria_with_time, criteria_without_time), file_list)))
        
    logger.info("Finished filtering files.")
    logger.info(f"Filtered {len(all_results)} files.")
    return all_results
