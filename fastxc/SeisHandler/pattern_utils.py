import os
import re
import sys
from collections import OrderedDict, Counter

# making this a global variable
field_to_regex = OrderedDict({
    "YYYY": r"(?P<year>\d{4})",  # 4 digits for year
    "YY": r"(?P<year>\d{2})",  # 2 digits for year
    "MM": r"(?P<month>\d{2})",  # 2 digits for month
    "DD": r"(?P<day>\d{2})",  # 2 digits for day
    "JJJ": r"(?P<jday>\d{3})",  # 3 digits for day of year
    "HH": r"(?P<hour>\d{2})",  # 2 digits for hour
    "MI": r"(?P<minute>\d{2})",  # 2 digits for minute
    "home": r"(?P<home>\w+)",  # for home directory
    "network": r"(?P<network>\w+)",  # for network code
    "event": r"(?P<event>\w+)",  # for network code
    "station": r"(?P<station>\w+)",  # for station name
    "component": r"(?P<component>\w+)",  # for component name
    "sampleF": r"(?P<sampleF>\w+)",  # for sampling frequency
    "quality": r"(?P<quality>\w+)",  # for quality indicator
    "locid": r"(?P<locid>\w+)",  # for location ID
    "suffix": r"(?P<suffix>\w+)",  # for file extension
    "label0": r"(?P<label0>\w+)",  # for file label0
    "label1": r"(?P<label1>\w+)",  # for file label1
    "label2": r"(?P<label2>\w+)",  # for file label2
    "label3": r"(?P<label3>\w+)",  # for file label3
    "label4": r"(?P<label4>\w+)",  # for file label4
    "label5": r"(?P<label5>\w+)",  # for file label5
    "label6": r"(?P<label6>\w+)",  # for file label6
    "label7": r"(?P<label7>\w+)",  # for file label7
    "label8": r"(?P<label8>\w+)",  # for file label8
    "label9": r"(?P<label9>\w+)",  # for file label9
})


def create_regex_pattern(pattern: str) -> str:
    """
    Create the regex pattern based on the given pattern string
    """
    # Replace field names with corresponding regex patterns
    for field_name, regex in field_to_regex.items():
        pattern = pattern.replace('{' + field_name + '}', regex)
    # Escape special characters and compile the final regex pattern
    pattern = pattern.replace('.', r'\.')
    pattern = pattern.replace('_', r'\_')
    pattern = pattern.replace('/', r'\/')
    # Replace '?' (any character wildcard) with regex for any characters except for special characters
    pattern = pattern.replace('{?}', '[^. _/]*')
    # Replace '*'(any character wildcard) with regex for any characters
    pattern = pattern.replace('{*}', '.*')
    return r"{}".format(pattern)


def check_pattern(array_dir: str, pattern: str) -> str:
    """
    Check if pattern is a valid string and return a dictionary with
    """
    # check if all fields in the pattern are valid
    pattern_fields = re.findall(r"\{(\w+)}", pattern)
    fild_counts = Counter(pattern_fields)

    # avoid duplicate fields
    duplicate_fields = [field for field, count in fild_counts.items() if count > 1]
    if duplicate_fields:
        raise ValueError(f"pattern contains duplicate fields: {duplicate_fields}")

    pattern_fields = set(re.findall(r"\{(\w+)}", pattern))
    valid_fields = set(field_to_regex.keys())
    if not pattern_fields.issubset(valid_fields):
        invalid_fields = pattern_fields - valid_fields
        raise ValueError(f"pattern contains invalid fields: {invalid_fields}")

    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")
    necessary_fields = ["{home}"]
    date_fields0 = ["{YYYY}", "{MM}", "{DD}"]
    date_fields1 = ["{YYYY}", "{JJJ}"]
    date_fields2 = ["{YY}", "{MM}", "{DD}"]
    date_fields3 = ["{YY}", "{JJJ}"]
    # check the necessary fields
    for field in necessary_fields:
        if field not in pattern:
            raise ValueError(f"pattern must contain {field}")

    # check if one of the date fields is in the pattern
    if not any(field in pattern for field in date_fields0 + date_fields1 + date_fields2 + date_fields3):
        raise ValueError(
            f"pattern must contain one of the following fields:\n"
            f"{date_fields0 + date_fields1 + date_fields2 + date_fields3}")

    # check is sac_dir is a dir, else warning
    if not os.path.isdir(array_dir):
        print(f"Warning: {array_dir} is not a directory", file=sys.stderr)
    # Apply the sac_dir to the pattern
    array_dir = os.path.normpath(array_dir)
    pattern = pattern.replace("{home}", array_dir)
    regex_pattern = create_regex_pattern(pattern)
    return regex_pattern
