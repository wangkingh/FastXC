from typing import Dict, List, Optional
from .pattern_utils import check_pattern
from .seisfile_matcher import get_files, match_files
from .seisfile_filter import filter_files
from .seisfile_organizer import group_by_labels, organize_by_labels


class SeisArray:
    """
    SeisArray class is designed for organizing the noise data into a virtual array.
    """

    def __init__(self, array_dir: str, pattern: str):
        self.array_dir = array_dir
        self.pattern = check_pattern(array_dir, pattern)
        self.files = None
        self.filtered_files = None
        self.pattern_filter = None
        self.files_group = None
        self.virtual_array = None

    def match(self, threads: int = 1):
        """
        Matching files according to the pattern
        """
        # Get all files in the directory
        file_list = get_files(self.array_dir)

        # Match all files with the pattern
        self.files = match_files(file_list, self.pattern, num_threads=threads)

    def filter(self, criteria: Optional[Dict[str, List]] = None, threads: int = 1):
        """
        Apply the file filter to the directory, and store the matched files.
        """

        if self.files is None:
            print("[Error] Please match the files first.")
            return None

        self.filtered_files = filter_files(self.files, criteria, num_threads=threads)

    def group(self, labels: list, sort_labels: list = None, filtered=True):
        """
        re-organize the array files according to the order
        """
        if filtered:
            files = self.filtered_files
            if files is None:
                print("[Error] Please filter the files first.")
                return None
        else:
            files = self.files
            if files is None:
                print("[Error] Please match the files first.")
                return None
        files_group = group_by_labels(files, labels, sort_labels)
        self.files_group = files_group.to_dict(orient='index')

    def organize(self, label_order: list, output_type='dict', filtered=True):
        """
        re-organize the array files according to the order
        """
        if filtered:
            files = self.filtered_files
            if files is None:
                print("[Error] Please filter the files first.")
                return None
        else:
            files = self.files
            if files is None:
                print("[Error] Please match the files first.")
                return None
        if output_type not in ['path', 'dict']:
            print("[Error] flag should be 'path' or 'dict'.")
            output_type = 'dict'
        self.virtual_array = organize_by_labels(files, label_order, output_type)
