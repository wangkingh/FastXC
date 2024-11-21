from collections import defaultdict
import pandas as pd


def group_by_labels(matched_files, labels, sort_labels):
    """
    Organize the matched files into a multi-level dictionary according to the order.

    matched_files: a list of file paths, every file is a dictionary contains fields and path
    """
    if not matched_files:
        raise ValueError("No files matched. Matching First!.")

    df = pd.DataFrame(matched_files)

    # Check if all the keys in order are in fields of file_info
    if not all(field in df.columns for field in labels):
        raise ValueError("The provided order has fields not present in the file_info")

        # Check if all the keys in sort_labels are in fields of file_info
    if sort_labels is not None:
        if not all(field in df.columns for field in sort_labels):
            raise ValueError("The provided sort_labels has fields not present in the file_info")

        # Check if sort_labels and labels are conflict
        if not set(sort_labels).isdisjoint(set(labels)):
            raise ValueError("The provided sort_labels and labels should not intersect")

    df_grouped = df.groupby(labels).agg(list)

    if sort_labels is not None:
        df_grouped = df_grouped.sort_values(by=sort_labels)

    return df_grouped


def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)


def add_path(multi_dict, keys, value):
    for key in keys[:-1]:
        multi_dict = multi_dict[key]
    # add the value into the last level of the dictionary
    if not multi_dict[keys[-1]]:
        multi_dict[keys[-1]] = []
    multi_dict[keys[-1]].append(value)


def organize_by_labels(matched_files, order, information_type):
    if not matched_files:
        raise ValueError("No files matched. Matching First!.")
    if information_type not in ['path', 'dict']:
        raise ValueError("The information_type should be 'path' or 'dict'")
    df = pd.DataFrame(matched_files)
    multi_dict = recursive_defaultdict()
    for _, row in df.iterrows():
        keys = [row[field] for field in order]
        if information_type == 'path':
            add_path(multi_dict, keys, row['path'])
        elif information_type == 'dict':
            add_path(multi_dict, keys, row.to_dict())
    return multi_dict
