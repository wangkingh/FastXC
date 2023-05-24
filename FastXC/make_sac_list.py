# make_sac_list.py is a python script that can be used to create a list of SAC files based on the given pattern.
import os
import re
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

# create_regex pattern from the input of user and match files based on the pattern


class MakeSacList:
    def __init__(self, sac_directory: str) -> None:

        self.sac_directory = sac_directory

        self.pattern = None
        self.regex_pattern = None

        self.network_list = None
        self.kcmpnm = None
        self.suffix = None

        self.station_list_file = None
        self.station_list = None

        self.start_time = None
        self.end_time = None

        self.matched_files = None
        self.numthreads = None

    @property
    def sac_directory(self):
        return self._sac_directory

    @sac_directory.setter
    def sac_directory(self, value):
        self._sac_directory = value

    @property
    def pattern(self):
        return self._pattern

    @pattern.setter
    def pattern(self, value):
        self._pattern = value

    @property
    def network_list(self):
        return self._network_list

    @network_list.setter
    def network_list(self, value):
        self._network_list = value

    @property
    def station_list_file(self):
        return self._station_list_file

    @station_list_file.setter
    def station_list_file(self, value):
        self._station_list_file = value

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = value

    @property
    def end_time(self):
        return self._end_time

    @end_time.setter
    def end_time(self, value):
        self._end_time = value

    @property
    def kcmpnm(self):
        return self._kcmpnm

    @kcmpnm.setter
    def kcmpnm(self, value):
        self._kcmpnm = value

    @property
    def numthreads(self):
        return self._numthreads

    @numthreads.setter
    def numthreads(self, value):
        self._numthreads = value

    @property
    def suffix(self):
        return self._suffix

    def suffix(self, value):
        self._suffix = value

    def load_list_from_file(self, file_path: str):
        with open(file_path, 'r') as f:
            self.station_list = []
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    self.station_list.append(line)

    # create regex pattern
    def create_regex_pattern(self, pattern: str) -> str:
        """
        Create the regex pattern based on the given pattern string
        """
        # mapping the input str to regex pattern
        field_to_regex = OrderedDict({
            "YYYY": r"(?P<year>\d{4})",  # 4 digits for year
            "YY": r"(?P<year>\d{2})",  # 2 digits for year
            "MM": r"(?P<month>\d{2})",  # 2 digits for month
            "DD": r"(?P<day>\d{2})",  # 2 digits for day
            "JJJ": r"(?P<jday>\d{3})",  # 3 digits for day of year
            "HH": r"(?P<hour>\d{2})",  # 2 digits for hour
            "MI": r"(?P<minute>\d{2})",  # 2 digits for minute
            "knetwk": r"(?P<knetwk>\w+)",  # for network code
            "kstnm": r"(?P<kstnm>\w+)",  # for station name
            "kcmpnm": r"(?P<kcmpnm>\w+)",  # for component name
            "suffix": r"(?P<suffix>\w+)",  # for file extension
        })

        # Replace field names with corresponding regex patterns
        for field_name, regex in field_to_regex.items():
            pattern = pattern.replace('{' + field_name + '}', regex)

        # Replace '?' (any character wildcard) with regex for any characters except for special characters
        pattern = pattern.replace('{?}', '[^. _-]*')

        # Escape special characters and compile the final regex pattern
        pattern = pattern.replace('.', r'\.')
        pattern = pattern.replace('_', r'\_')

        # Replace escaped wildcard regex with original
        pattern = pattern.replace('\[^. \_-\]\*', '[^. _-]*')

        return r"{}".format(pattern)

    # check the inputs
    def check_inputs(self):
        # Check sac_directory
        if not os.path.isdir(self.sac_directory):
            raise ValueError("sac_directory must be a valid directory.")

        # Check pattern
        if self.pattern is not None:
            if not self.pattern:  # Check if list is empty
                self.pattern = None
            else:
                self.regex_pattern = self.create_regex_pattern(self.pattern)

        # Check network_list
        if self.network_list is not None:
            if not isinstance(self.network_list, list):
                raise ValueError("network_list must be a list.")
            if not self.network_list:  # Check if list is empty
                self.network_list = None

        # Check kcmpnm
        if self.kcmpnm is not None:
            if not isinstance(self.kcmpnm, str):
                raise ValueError("kcmpnm must be a string.")
            if not self.kcmpnm.strip():  # Check if string is empty
                self.kcmpnm = None

        # Check suffix
        if self.suffix is not None:
            if not isinstance(self.suffix, str):
                raise ValueError("suffix must be a string.")

        # Check station_list_file
        if self.station_list_file is not None:
            if not os.path.isfile(self.station_list_file):
                raise ValueError(
                    "station_list_file must be a valid file path.")
            self.load_list_from_file(self.station_list_file)
            if not self.station_list:
                self.station_list = None

        # Chekck thread number
        if self.numthreads is None:
            self.numthreads = 1
        if not isinstance(self.numthreads, int):
            print("[Warning]: numthreads must be an integer.")
            print("[Warning]: numthreads is set to 1.")
            self.numthreads = 1
        if self.numthreads < 1:
            print("[Warning]: numthreads must be greater than 0.")
            print("[Warning]: numthreads is set to 1.")
            self.numthreads = 1

        # Check start_time and end_time
        date_format = "%Y-%m-%d %H:%M:%S"
        far_past = "1970-01-01 00:00:00"
        far_future = "2299-12-31 23:59:59"

        if self.start_time is None and self.end_time is None:
            print("[Warning]: start_time and end_time are not set.")

        if self.start_time is not None and self.end_time is None:
            try:
                datetime.strptime(self.start_time, date_format)
            except ValueError:
                raise ValueError(
                    "start_time must be in format: 'YYYY-mm-dd HH:MM:SS'")
            print(
                '[Warning]: end_time is not set. Default end_time is set to "2299-12-31 23:59:59"')
            self.end_time = far_future

        if self.start_time is None and self.end_time is not None:
            try:
                self.end = datetime.strptime(self.end_time, date_format)
            except ValueError:
                raise ValueError(
                    "end_time must be in format: 'YYYY-mm-dd HH:MM:SS'")
            self.start = datetime.strptime(far_past, date_format)
            print(
                '[Warning]: start_time is not set. Default start_time is set to "1970-01-01 00:00:00"')

        if self.start_time is not None and self.end_time is not None:
            try:
                datetime.strptime(self.start_time, date_format)
                datetime.strptime(self.end_time, date_format)
            except ValueError:
                raise ValueError(
                    "start_time and end_time must be in format: 'YYYY-mm-dd HH:MM:SS'")
        return

    def print_criteria(self):
        print("Matching criteria:")
        print("  Pattern: " + self.pattern)
        # print(f"  Regex pattern: {self.regex_pattern}")
        print(f"  Network list: {self.network_list}")
        print(f"  Start time: {self.start_time}")
        print(f"  End time: {self.end_time}")
        print(f"  Component name: {self.kcmpnm}")
        print(f"  Suffix: {self.suffix}")
        print(f"  Number of threads: {self.numthreads}")
        if self.station_list_file == None:
            print("  Station list file: None")
            return
        # Format and print the station list
        print("  Station list:")
        for i in range(min(10, len(self.station_list))):
            print("    " + self.station_list[i])
        if len(self.station_list) > 50:
            print("    ...")
            print(
                f"    Please check the {self.station_list_file} for the full station list.")

   # process the file name
    def process_file_name(self, file_path):
        """
        Process the file name and return the network, station and time.
        """
        file_name = os.path.basename(file_path)
        if not self.regex_pattern:
            return None, None, None, None, None
        try:
            details = re.match(self.regex_pattern, file_name).groupdict()
        except AttributeError:
            # print("[Warning]: The file name does not match the pattern.")
            # print(f"[Warning]: File path: {file_path}")
            return '', '', '', '', ''

        network = details.get("knetwk")
        station = details.get("kstnm")
        kcmpnm = details.get("kcmpnm")
        suffix = details.get("suffix")
        # processing time information
        try:
            year = int(
                details["year"]) if "year" in details and details["year"] is not None else None
            month = int(
                details["month"]) if "month" in details and details["month"] is not None else None
            day = int(
                details["day"]) if "day" in details and details["day"] is not None else None
            jday = int(
                details["jday"]) if "jday" in details and details["jday"] is not None else None
            hour = int(
                details["hour"]) if "hour" in details and details["hour"] is not None else None
            minute = int(
                details["minute"]) if "minute" in details and details["minute"] is not None else None
        except ValueError:
            raise ValueError(
                "The format of the file name is not correct. Please check the pattern.")

        time = None
        if year and jday:
            time = datetime(year, 1, 1) + timedelta(days=jday - 1)
        elif year and month and day:
            time = datetime(year, month, day)
        elif year and jday and hour and minute:
            time = datetime(year, 1, 1) + timedelta(days=jday -
                                                    1, hours=hour, minutes=minute)
        elif year and month and day and hour and minute:
            time = datetime(year, month, day, hour, minute)

        return network, station, time, kcmpnm, suffix

    # match a single file
    def match_single_file(self, file_path):
        """
        Process a single file, check if it matched the pattern.
        If it matches, return the file_path otherwise return None.
        """

        # Get the network, station and time
        try:
            network, station, time, kcmpnm, suffix = self.process_file_name(
                file_path)
        except ValueError:
            return None

        if network == '' and station == '' and time == '' and kcmpnm == '' and suffix == '':
            return None

        # Check if the file matches the pattern
        if self.network_list and network not in self.network_list:
            return None

        if self.station_list and station not in self.station_list:
            return None

        if self.kcmpnm and kcmpnm != self.kcmpnm:
            return None

        if self.suffix and suffix != self.suffix:
            return None

        if self.start_time and self.end_time:
            start_time = datetime.strptime(
                self.start_time, "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            if not (start_time <= time <= end_time):
                return None

        return file_path    # Return the file path if it matches

    # match the files
    def match_files(self):
        """
        Match the files based on the given pattern and return the matched files.
        """
        if not self.regex_pattern:
            print(
                "[Warning]: No matching pattern set. All files by regex_pattern will be matched.")

        self.matched_files = []
        # Get all files in sac_directory
        all_files = [os.path.join(root, file) for root, dirs, files in os.walk(
            self.sac_directory) for file in files]

        with ThreadPoolExecutor(max_workers=self._numthreads) as executor:
            futures = {executor.submit(self.match_single_file, file)
                       for file in all_files}

            for future in tqdm(as_completed(futures), total=len(futures), desc='Matching files'):
                file_path = future.result()
                if file_path is not None:
                    self.matched_files.append(file_path)

        return self.matched_files

    # write the matched files to a file
    def write_matched_files(self, sac_list_file: str):
        if not self.matched_files:
            print("[Warning]: No matched files found. saclist will be empty.")
            print("[Advice] : Please check the matching criteria.")
        try:
            with open(sac_list_file, 'w') as f:
                for file in self.matched_files:
                    f.write(file + '\n')
            print(f"Matched files written to: {sac_list_file}")
        except IOError:
            raise IOError("Error writing to the SAC list file.")

    # prevent the user from accessing the attributes
    def __getattr__(self, name):
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
