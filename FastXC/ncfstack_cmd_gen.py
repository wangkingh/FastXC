# Purpose: Generate the command to run ncfstack_mpi
import os


class GenStackCmd:
    def __init__(self, ncf_list_list: str, output_directory: str):
        self.ncf_list_list = ncf_list_list
        self.output_directory = output_directory
        self.normalize_output = True
        self.command = 'ncfstack_mpi'

    @property
    def ncf_list_list(self):
        return self._ncf_list_list

    @ncf_list_list.setter
    def ncf_list_list(self, value):
        self._ncf_list_list = value

    @property
    def output_directory(self):
        return self._output_directory

    @output_directory.setter
    def output_directory(self, value):
        self._output_directory = value

    @property
    def normalize_output(self):
        return self._normalize_output

    @normalize_output.setter
    def normalize_output(self, value):
        self._normalize_output = value
    
    @property
    def command(self):
        return self._command
    
    @command.setter
    def command(self, value):
        self._command = value

    def check_input(self):
        
        # check ncf_list_list
        if not os.path.isfile(self.ncf_list_list) or os.stat(self.ncf_list_list).st_size == 0:
            print(
                "[Warning] Invalid NCF list list. File does not exist, is empty or has no content.")

        # check output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            print(f"Output directory created: {self.output_directory}")

        # check normalize_output
        if not isinstance(self.normalize_output, bool):
            print(
                "[Warning] normalize_output should be a boolean. Setting it to True by default.")
            self.normalize_output = True

    def generate_command(self) -> str:
        command_parts = [self.command, '-I',
                         self.ncf_list_list, '-O', self.output_directory]

        if not self.normalize_output:
            command_parts.append('-A')

        return ' '.join(command_parts)
