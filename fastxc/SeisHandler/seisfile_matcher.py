import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Dict
from datetime import datetime, timedelta

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


def gen_time_from_fields(fields) -> datetime:
    """
    Get the datetime object from the given fields
    """
    year = fields.get("year")
    month = fields.get("month")
    day = fields.get("day")
    jday = fields.get("jday")
    hour = fields.get("hour", "0")
    minute = fields.get("minute", "0")

    if year is not None:
        if len(year) == 2:
            year = 2000 + int(year)  # Assume 20xx
        else:
            year = int(year)

    if jday is not None:
        jday = int(jday)

    if month is not None:
        month = int(month)

    if day is not None:
        day = int(day)

    # Since we've set default values for hour and minute, these will not be None
    hour = int(hour)
    minute = int(minute)

    if year and jday:
        time = datetime(year, 1, 1) + timedelta(days=jday - 1, hours=hour, minutes=minute)
        # del fields["year"]
        # del fields["jday"]
        # del fields["hour"]
        # del fields["minute"]
        return time
    elif year and month and day:
        time = datetime(year, month, day, hour, minute)
        # del fields["year"]
        # del fields["month"]
        # del fields["day"]
        # del fields["hour"]
        # del fields["minute"]
        return time
    else:
        raise ValueError("Invalid time fields")


def get_files(directory: str) -> List[str]:
    file_list = []
    count = 0
    logger.info(f"Searching for files in {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
            count += 1
    logger.info(f"Finish. {count} files found in {directory}")
    return file_list


def match_file(file_path: str, regex_pattern: str) -> Dict:
    """
    Match a file with the given regex_pattern
    """
    # Match the file name with the regex pattern
    fields = {}
    try:
        match = re.match(regex_pattern, file_path)
        if match:
            fields = match.groupdict()
            fields["time"] = gen_time_from_fields(fields)  # Add time to the fields
            fields["path"] = file_path  # Add the file path to the fields
    except Exception as e:
        logger.error(f"An error occurred while processing the file {file_path}: {e}")
    return fields


def match_files(file_paths: list, regex_pattern: str, num_threads: int) -> List[Dict]:
    """
    Match a list of files with the given pattern
    """

    # Create a partial function with the pattern as the first argument
    partial_match_file = partial(match_file, regex_pattern=regex_pattern)

    # Initialize a list to store the results
    all_results = []
    logger.info("Start file pattern matching...")
    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:

        # Use 'map' to apply the partial function to each file path
        future_results = executor.map(partial_match_file, file_paths)

        # Iterate through the results
        for result in future_results:
            if result:  # Filters out None or empty results
                all_results.append(result)

    logger.info("File pattern matching completed.")
    logger.info(f"{len(all_results)} files matched.")
    return all_results
