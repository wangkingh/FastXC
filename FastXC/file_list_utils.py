# file_list_utils.py is a module that contains functions to generate list files for the SAC and NCF files.

import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def write_files(parameters):
    # unpack the parameters
    file_direcrory, list_directory, file_extension = parameters

    # Generate the "list_file" path
    list_file_path = os.path.join(
        list_directory, os.path.basename(file_direcrory))

    # Open the "list_file" and write file paths into it
    allfiles = glob.glob(file_direcrory + '/*' + file_extension)
    allfiles.sort()
    with open(list_file_path, 'w') as file:
        for file_path in allfiles:
            file.write(file_path + '\n')


def MakeSpectrumList(spectrum_directory: str, spectrum_list_directory: str, num_threads: int = 1):
    if not os.path.exists(spectrum_directory):
        raise ValueError(
            f"The sac_directory {spectrum_directory} does not exist.")
    # Create the list folder of spectrum if it does not exist
    if not os.path.exists(spectrum_list_directory):
        os.makedirs(spectrum_list_directory)

    # Get all the spectrum directories in the spectrum folder
    spectrum_time_dirs = glob.glob(spectrum_directory+'/*')

    # Create a ProcessPoolExecutor to parallelize writing list
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = []
        with tqdm(total=len(spectrum_time_dirs), desc='Making spectrum list') as pbar:
            for result in executor.map(write_files, zip(spectrum_time_dirs, [spectrum_list_directory]*len(spectrum_time_dirs), ['segspec']*len(spectrum_time_dirs))):
                results.append(result)
                pbar.update()


def MakeStackList(ncf_directory: str, stack_list_directory: str, num_threads: int = 1):
    # Create the list folder of stacklist if it does not exist
    if not os.path.exists(stack_list_directory):
        os.makedirs(stack_list_directory)

    # Get all the ncf_stapair directories in the ncf folder
    ncf_pair_dirs = glob.glob(ncf_directory+'/*')

    # Create a ProcessPoolExecutor to parallelize writing list
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = []
        with tqdm(total=len(ncf_pair_dirs), desc='Making stack list') as pbar:
            for result in executor.map(write_files, zip(ncf_pair_dirs, [stack_list_directory]*len(ncf_pair_dirs), ['sac']*len(ncf_pair_dirs))):
                results.append(result)
                pbar.update()


#  make stack list list: a list of stack list files
def MakeStackListList(stack_list_directory: str, stack_list_list: str):
    # check if the stack_list_directory exists, else rasie error
    if not os.path.exists(stack_list_directory):
        raise ValueError(
            f"The stack_list_directory {stack_list_directory} does not exist.")

    # Get all the stack_list files in the stack_list_directory
    stack_list_files = glob.glob(stack_list_directory+'/*')

    # Create the stack_list_list file
    with open(stack_list_list, 'w') as file:
        for stack_list_file in stack_list_files:
            file.write(stack_list_file+'\n')


# Match the spec_list file under two folders if they have the same basename (represents same time)
def MatchDoubleList(spec_list_folder1: str, spec_list_folder2: str):
    # check wheter the two folders exist, else raise error
    if not os.path.exists(spec_list_folder1):
        raise ValueError(
            f"The spec_list_folder1 {spec_list_folder1} does not exist.")
    if not os.path.exists(spec_list_folder2):
        raise ValueError(
            f"The spec_list_folder2 {spec_list_folder2} does not exist.")
    # match the spec_list files under the two folders
    spec_list_files1 = glob.glob(spec_list_folder1+'/*')
    spec_list_files2 = glob.glob(spec_list_folder2+'/*')
    spec_list_files1.sort()
    spec_list_files2.sort()

    # find out the matched spec_list files and return a list which each elements contain the two matched path
    matched_spec_list = []
    for spec_list_file1 in spec_list_files1:
        for spec_list_file2 in spec_list_files2:
            if os.path.basename(spec_list_file1) == os.path.basename(spec_list_file2):
                matched_spec_list.append([spec_list_file1, spec_list_file2])

    # check if the list is empty, if it is, raise error
    if len(matched_spec_list) == 0:
        raise ValueError(
            f"The spec_list_folder1 {spec_list_folder1} and spec_list_folder2 {spec_list_folder2} have no matched files.")

    # return the matched list
    return matched_spec_list
