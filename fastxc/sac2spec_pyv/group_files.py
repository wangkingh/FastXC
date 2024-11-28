def read_file(file_path):
    """读取文件内容，并将每行作为一个元素存储在列表中"""
    with open(file_path, "r") as file:
        lines = [line.strip() for line in file.readlines()]
    return lines


def group_files(input_file, output_file, group_size):
    """
    根据输入和输出文件路径以及组大小，生成文件组列表。

    :param input_file: 输入文件路径，文件中的每一行是一个输入文件的绝对路径
    :param output_file: 输出文件路径，文件中的每一行是一个输出文件的绝对路径
    :param group_size: 每组包含的文件数
    :return: 文件组列表，每个文件组是一个包含输入文件列表和输出文件列表的元组
    """
    input_files = read_file(input_file)
    output_files = read_file(output_file)

    if len(input_files) != len(output_files):
        raise ValueError(
            "The number of input files must match the number of output files."
        )

    grouped_files = []
    for i in range(0, len(input_files), group_size):
        input_group = input_files[i : i + group_size]
        output_group = output_files[i : i + group_size]
        grouped_files.append((input_group, output_group))

    return grouped_files