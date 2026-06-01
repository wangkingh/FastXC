import logging
from collections import OrderedDict
from typing import Dict, List, Optional

from .files import FileFilter, FileMatcher
from .patterns import DEFAULT_BASE_FIELDS, FieldRegistry, check_pattern

logger = logging.getLogger(__name__)


class SeisArray:
    """Match, filter, and group seismic file paths under one root directory."""

    def __init__(
        self,
        array_dir: str,
        pattern: str,
        custom_fields: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
    ):
        self.registry = FieldRegistry(OrderedDict(DEFAULT_BASE_FIELDS))
        for field_name, regex_str in (custom_fields or {}).items():
            self.registry.add_field(field_name, regex_str, overwrite=overwrite)

        self.array_dir = array_dir
        self.pattern = check_pattern(array_dir, pattern, self.registry)
        self.files: Optional[List[Dict]] = None
        self.filtered_files: Optional[List[Dict]] = None
        self.files_group: Optional[Dict] = None

    def match(self, threads: int = 1) -> List[Dict]:
        matcher = FileMatcher(directory=self.array_dir, regex_pattern=self.pattern)
        self.files = matcher.match_files(num_threads=threads)
        self.filtered_files = None
        self.files_group = None
        return self.files

    def filter(
        self,
        criteria: Optional[Dict[str, dict]] = None,
        threads: int = 1,
    ) -> Optional[List[Dict]]:
        if self.files is None:
            logger.warning("Please match the files first.")
            return None

        file_filter = FileFilter(criteria=criteria, num_threads=threads)
        self.filtered_files = file_filter.filter_files(self.files)
        self.files_group = None
        return self.filtered_files

    def group(
        self,
        labels: List[str],
        sort_labels: Optional[List[str]] = None,
        filtered: bool = True,
    ) -> Optional[Dict]:
        files = self.filtered_files if filtered else self.files
        if files is None:
            logger.error(
                "Please filter the files first." if filtered else "Please match the files first."
            )
            return None

        self.files_group = _group_files(files, labels, sort_labels)
        return self.files_group


def _group_files(
    files: List[Dict],
    labels: List[str],
    sort_labels: Optional[List[str]] = None,
) -> Dict:
    if not labels:
        raise ValueError("group labels must not be empty")
    if not files:
        return {}

    sort_labels = sort_labels or []
    required_fields = set(labels) | set(sort_labels)
    missing_fields = {
        field
        for file_info in files
        for field in required_fields
        if field not in file_info
    }
    if missing_fields:
        raise ValueError(f"group fields not present in matched files: {missing_fields}")

    ordered_files = list(files)
    if sort_labels:
        ordered_files.sort(key=lambda item: tuple(item[label] for label in sort_labels))

    grouped: Dict = OrderedDict()
    label_set = set(labels)
    for file_info in ordered_files:
        key_values = tuple(file_info[label] for label in labels)
        key = key_values[0] if len(key_values) == 1 else key_values
        group = grouped.setdefault(key, {})
        for field_name, value in file_info.items():
            if field_name in label_set:
                continue
            group.setdefault(field_name, []).append(value)

    return dict(grouped)
