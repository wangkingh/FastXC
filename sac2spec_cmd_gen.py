# sac2spec_cmnd_gen.py
# Description: This file contains the class sac2spec, which is used to store the parameters for the sac2spec program.
import os
import glob


class Sac2SpecCmdGen:
    ALLOWED_WHITEN_TYPES = ['None', 'Before', 'After', 'Both']
    ALLOWED_NORMALIZE_TYPES = ['None', 'Onebit', 'Runabs']

    # Initialize the class wtih required parameters
    def __init__(self, sac_list: str, output_directory: str, window_length: int) -> None:
        self.sac_list = sac_list  # path to the SAC list
        self.output_directory = output_directory  # output directory path
        self.window_length = window_length  # window length in seconds
        self.gpu_list = []  # list of GPUs to be used
        self.freq_bands = []  # list of frequency bands
        self.whiten_type = 'None'  # whiten type
        self.normalize_type = 'None'  # normalize type
        self.skip_check_npts = False  # whether skip check npts
        self.save_by_time = False  # whether save by time
        self.command = 'sac2spec_mg '  # command to be executed
        self.CUDA_whiten = 0
        self.CUDA_normalize = 0

    # property gatters and setters
    # These methods are used to get and set the property values with necessary validation
    @property
    def sac_list(self):
        return self._sac_list

    @sac_list.setter
    def sac_list(self, value):
        self._sac_list = value

    @property
    def output_directory(self):
        return self._output_directory

    @output_directory.setter
    def output_directory(self, value):
        self._output_directory = value

    @property
    def window_length(self):
        return self._window_length

    @window_length.setter
    def window_length(self, value):
        self._window_length = value

    @property
    def gpu_list(self):
        return self._gpu_list

    @gpu_list.setter
    def gpu_list(self, value):
        self._gpu_list = value

    @property
    def freq_bands(self):
        return self._freq_bands

    @freq_bands.setter
    def freq_bands(self, value):
        self._freq_bands = value

    @property
    def whiten_type(self):
        return self._whiten_type

    @whiten_type.setter
    def whiten_type(self, value):
        self._whiten_type = value

    @property
    def normalize_type(self):
        return self._normalize_type

    @normalize_type.setter
    def normalize_type(self, value):
        self._normalize_type = value

    @property
    def skip_check_npts(self):
        return self._skip_check_npts

    @skip_check_npts.setter
    def skip_check_npts(self, value):
        self._skip_check_npts = value

    @property
    def save_by_time(self):
        return self._save_by_time

    @save_by_time.setter
    def save_by_time(self, value):
        self._save_by_time = value

    def check_input(self) -> None:
        if not self.sac_list:
            print("[Warning]: SAC List is not specified.")
        if not os.path.exists(self.sac_list):
            print("[Warning]: The specified SAC list path does not exist.")
        if os.path.isfile(self.sac_list) and os.stat(self.sac_list).st_size == 0:
            print("[Warning]: The specified SAC list file is empty.")

        if not self.output_directory:
            print("[Warning]: Output Directory is not specified.")
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            print("Output directory created: {0}".format(
                self.output_directory))

        # check whiten_type
        if self.whiten_type not in self.ALLOWED_WHITEN_TYPES:
            # set to none
            self.whiten_type = 'None'
            print("[Warning]: Whiten Type is not valid. Set to None.")
            self.CUDA_whiten = 0
        else:
            self.CUDA_whiten = self._get_cuda_c_whiten_type()

        # check normalize_type
        if self.normalize_type not in self.ALLOWED_NORMALIZE_TYPES:
            # set to none
            self.normalize_type = 'None'
            print("[Warning]: Normalize Type is not valid. Set to None.")
            self.CUDA_normalize = 0
        else:
            self.CUDA_normalize = self._get_cuda_c_normalize_type()

        if not self.freq_bands:
            raise ValueError("Frequency Bands are not specified.")

        if not all(isinstance(freq, str) and '/' in freq for freq in self.freq_bands):
            raise ValueError(
                "Frequency Bands must be strings in the format '2/3'.")

        if not self.window_length:
            raise ValueError("Window Length is not specified.")

        if not self.gpu_list:
            raise ValueError("GPU List is not specified.")

        print("Input parameters are valid.")

    # methods to print the selected parameters
    def print_selected_parameters(self) -> None:
        print("Selected Parameters:")
        print("- SAC List: {0}".format(self.sac_list))
        print("- Output Directory: {0}".format(self.output_directory))
        print(
            "- Window Length: {0} (unit: seconds)".format(self.window_length))

        if self.gpu_list:
            print("- GPU List: {0}".format(self.gpu_list))
        else:
            print("Warning: GPU List is not specified.")

        if self.freq_bands:
            print("- Frequency Bands: {0} (unit: Hz)".format(self.freq_bands))
        else:
            print("Warning: Frequency Bands are not specified.")

        print("- Whiten Type (Python): {0}".format(self.whiten_type))
        print(
            "- Whiten Type (CUDA-C): {0}".format(self.CUDA_whiten))
        print("- Normalize Type (Python): {0}".format(self.normalize_type))
        print(
            "- Normalize Type (CUDA-C): {0}".format(self.CUDA_normalize))
        print("- Skip Check NPTS: {0}".format(self.skip_check_npts))
        print("- Save By Time: {0}".format(self.save_by_time))

    def generate_command(self) -> str:

        command_parts = [self.command, '-I', self.sac_list, '-O',
                         self.output_directory, '-L', str(self.window_length)]

        if self.gpu_list:
            gpu_list_str = '/'.join(str(gpu) for gpu in self.gpu_list)
            command_parts += ['-G', gpu_list_str]

        if self.freq_bands:
            freq_bands_str = ' '.join(freq for freq in self.freq_bands)
            # add a dummy freq band to suit the format of sac2spec
            freq_bands_str = freq_bands_str+' -1/-1'
            command_parts += ['-F', freq_bands_str]

        command_parts += ['-W', self.CUDA_whiten, '-N', self.CUDA_normalize]

        if self.skip_check_npts:
            command_parts += ['-K']

        if self.save_by_time:
            command_parts += ['-T']

        return ' '.join(command_parts)

    def _get_cuda_c_whiten_type(self):
        # return CUDA-C's whiten_type value
        if self.whiten_type == 'None':
            return '0'
        elif self.whiten_type == 'Before':
            return '1'
        elif self.whiten_type == 'After':
            return '2'
        elif self.whiten_type == 'Both':
            return '3'

    def _get_cuda_c_normalize_type(self):
        # return CUDA-C's normalize_type value
        if self.normalize_type == 'None':
            return '0'
        elif self.normalize_type == 'Onebit':
            return '1'
        elif self.normalize_type == 'Runabs':
            return '2'

    # prevent the user from accessing the attributes
    def __getattr__(self, name):
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
