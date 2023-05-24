# Description: This file contains the class definitions for the command generators for specxc_mg.
# Purpose: Generate command for specxc_mg
import os
import warnings


class SingleListCmdGen:
    def __init__(self, virt_src_list: str, half_cc_length: float, output_directory: str, gpu_id: int):
        # Initial parameter values
        self.virt_src_list = virt_src_list
        self.half_cc_length = half_cc_length
        self.output_directory = output_directory
        self.gpu_id = gpu_id

        # Optional parameters
        self.checkhead = True
        self.do_cross_correlation = True  # if false, do auto-correlation
        self.save_by_pair = True

        # Command template
        self.command = 'specxc_mg '

    # Getters and setters for each parameter

    @property
    def virt_src_list(self):
        return self._virt_src_list

    @virt_src_list.setter
    def virt_src_list(self, value):
        self._virt_src_list = value

    @property
    def half_cc_length(self):
        return self._half_cc_length

    @half_cc_length.setter
    def half_cc_length(self, value):
        self._half_cc_length = value

    @property
    def output_directory(self):
        return self._output_directory

    @output_directory.setter
    def output_directory(self, value):
        self._output_directory = value

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        self._gpu_id = value

    @property
    def checkhead(self):
        return self._checkhead

    @checkhead.setter
    def checkhead(self, value):
        self._checkhead = value

    @property
    def do_cross_correlation(self):
        return self._do_cross_correlation

    @do_cross_correlation.setter
    def do_cross_correlation(self, value):
        self._do_cross_correlation = value

    @property
    def save_by_pair(self):
        return self._save_by_pair

    @save_by_pair.setter
    def save_by_pair(self, value):
        self._save_by_pair = value

    # Method to generate command

    def generate_command(self) -> str:
        # Start with base command
        command_parts = [self.command, '-A', self.virt_src_list, '-C', str(self.half_cc_length),
                         '-O', self.output_directory, '-G', str(self.gpu_id)]

        # Add optional parts of command
        if not self.checkhead:
            command_parts.append('-K')
        if self.save_by_pair:
            command_parts.append('-S')
        if self.do_cross_correlation:
            command_parts.append('-X')

        return ' '.join(command_parts)

    # New method to print selected parameters

    def check_input(self):
        if not self.virt_src_list:
            warnings.warn("virt_src_list is not specified.")
        elif not os.path.isfile(self.virt_src_list) or os.stat(self.virt_src_list).st_size == 0:
            print(
                "[Warning]: Invalid virt_src_list file. File does not exist, is empty, or has no content.")

        if not isinstance(self.half_cc_length, (int, float)):
            print("[Warning]: half_cc_length must be a number.")

        if not self.output_directory:
            print("[Warning]: output_directory is not specified.")

        if not isinstance(self.gpu_id, int):
            print("[Warning]: gpu_id must be an integer.")

        if not isinstance(self.checkhead, bool):
            print("[Warning]: checkhead must be a boolean.")

        if not isinstance(self.do_cross_correlation, bool):
            print("[Warning]: do_cross_correlation must be a boolean.")

        if not isinstance(self.save_by_pair, bool):
            print("[Warning]: save_by_pair must be a boolean.")

    def print_selected_parameters(self):
        print(f"Selected Parameters:")
        print(f"virt_src_list: {self.virt_src_list}")
        print(f"half_cc_length: {self.half_cc_length}")
        print(f"output_directory: {self.output_directory}")
        print(f"gpu_id: {self.gpu_id}")
        print(f"checkhead: {self.checkhead}")
        print(f"do_cross_correlation: {self.do_cross_correlation}")
        print(f"save_by_pair: {self.save_by_pair}")

    # prevent the user from accessing the attributes
    def __getattr__(self, name):
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")


class DoubleListCmdGen:
    def __init__(self, virt_src_list: str, virt_sta_list: str, half_cc_length: float, output_directory: str, gpu_id: int):
        # Initial parameter values
        self._virt_src_list = virt_src_list
        self._virt_sta_list = virt_sta_list
        self._half_cc_length = half_cc_length
        self._output_directory = output_directory
        self._gpu_id = gpu_id

        # Optional parameters
        self._do_cross_correlation = True
        self._save_by_pair = False
        self._checkhead = True

        # Command template
        self._command = 'specxc_mg '

    # Getters and setters for each parameter
    # Getters and setters for each parameter
    @property
    def virt_src_list(self):
        return self._virt_src_list

    @virt_src_list.setter
    def virt_src_list(self, value):
        self._virt_src_list = value

    @property
    def virt_sta_list(self):
        return self._virt_sta_list

    @virt_sta_list.setter
    def virt_sta_list(self, value):
        self._virt_sta_list = value

    @property
    def half_cc_length(self):
        return self._half_cc_length

    @half_cc_length.setter
    def half_cc_length(self, value):
        self._half_cc_length = value

    @property
    def output_directory(self):
        return self._output_directory

    @output_directory.setter
    def output_directory(self, value):
        self._output_directory = value

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, value):
        self._gpu_id = value

    @property
    def do_cross_correlation(self):
        return True

    def do_cross_correlation(self, value: bool):
        self._do_cross_correlation = value

    @property
    def save_by_pair(self):
        return self._save_by_pair

    @save_by_pair.setter
    def save_by_pair(self, value: bool):
        self._save_by_pair = value

    @property
    def checkhead(self):
        return self._checkhead

    @checkhead.setter
    def checkhead(self, value: bool):
        self._checkhead = value

    def generate_command(self) -> str:
        command_parts = [self._command, '-A', self.virt_src_list, '-B', self.virt_sta_list, '-O', self.output_directory, '-C',
                         str(self.half_cc_length), '-G', str(self.gpu_id)]

        if self.do_cross_correlation:
            command_parts.append('-X')

        if not self.checkhead:
            command_parts.append('-K')

        if self.save_by_pair:
            command_parts.append('-S')

        return ' '.join(command_parts)

    # Method to print selected parameters

    def print_selected_parameters(self):
        print(f"Selected Parameters:")
        print(f"virt_src_list: {self._virt_src_list}")
        print(f"virt_sta_list: {self._virt_sta_list}")
        print(f"half_cc_length: {self._half_cc_length}")
        print(f"output_directory: {self._output_directory}")
        print(f"gpu_id: {self._gpu_id}")
        print(f"do_cross_correlation: {self._do_cross_correlation}")
        print(f"save_by_pair: {self._save_by_pair}")
        print(f"checkhead: {self._checkhead}")

    def check_input(self):
        if not self.virt_src_list:
            print("[Warning]: virt_src_list is not specified.\n")
        elif not os.path.isfile(self.virt_src_list) or os.stat(self.virt_src_list).st_size == 0:
            print(
                "[Warning]: Invalid virt_src_list file. File does not exist, is empty, or has no content.\n")

        if not self.virt_sta_list:
            print("[Warning]: virt_sta_list is not specified.\n")
        elif not os.path.isfile(self.virt_sta_list) or os.stat(self.virt_sta_list).st_size == 0:
            print(
                "[Warning]: Invalid virt_sta_list file. File does not exist, is empty, or has no content.\n")

        if not isinstance(self.half_cc_length, (int, float)):
            print("[Warning]: half_cc_length must be a number.\n")

        if not self.output_directory:
            print("[Warning]: output_directory is not specified.\n")

        if not isinstance(self.gpu_id, int):
            print("[Warning]: gpu_id must be an integer.\n")

        if not isinstance(self.do_cross_correlation, bool):
            print("[Warning]: do_cross_correlation must be a boolean.\n")

        if not isinstance(self.save_by_pair, bool):
            print("[Warning]: save_by_pair must be a boolean.\n")
    # Prevent the user from accessing the attributes

    def __getattr__(self, name):
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
