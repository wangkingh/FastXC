from .config_parse import parse_and_check_ini_file
from .design_filter import design_filter
from .gen_sac2spec_list_dir import gen_sac2spec_list_dir
from .gen_xc_list_dir import gen_xc_list_dir
from .gen_stack_list_dir import gen_stack_list_dir
from .gen_rotate_list_dir import gen_rotate_list_dir
from .cmd_sac2spec_utils import gen_sac2spec_cmd
from .cmd_xc_utils import gen_xc_cmd
from .cmd_StackRotate_utils import gen_stack_cmd, gen_rotate_cmd
from .deployer_sac2spec import sac2spec_cmd_deployer
from .deployer_xc import xc_cmd_deployer
from .deployer_StackRotate import stack_cmd_deployer, rotate_cmd_deployer
from .deployer_sac2dat import sac2dat_deployer
import os
import shutil


class FastXC:
    def __init__(self, config_path):
        # Parse the configuration file
        (
            self.SeisArrayInfo,
            self.parameters,
            self.command,
            self.gpu_info,
        ) = parse_and_check_ini_file(config_path)

    def generate_filter(self):
        design_filter(self.parameters)

    def generate_sac2spec_list_dir(self):
        gen_sac2spec_list_dir(self.SeisArrayInfo, self.parameters, self.gpu_info)

    def generate_sac2spec_cmd(self):
        gen_sac2spec_cmd(
            self.SeisArrayInfo, self.command, self.parameters, self.gpu_info
        )

    def deploy_sac2spec_cmd(self):
        sac2spec_cmd_deployer(self.parameters)

    def generate_xc_list_dir(self):
        gen_xc_list_dir(self.parameters)

    def generate_xc_cmd(self):
        gen_xc_cmd(self.command, self.parameters, self.gpu_info)

    def deploy_xc_cmd(self):
        xc_cmd_deployer(self.parameters, self.gpu_info)

    def generate_stack_list_dir(self):
        calculate_type = self.parameters["calculate_style"]
        if calculate_type == "DUAL":
            print(
                "Dual type has already stacked the files in the step of cross-correlation."
            )
        elif calculate_type == "MULTI":
            gen_stack_list_dir(self.parameters, self.SeisArrayInfo)

    def generate_stack_cmd(self):
        calculate_type = self.parameters["calculate_style"]
        if calculate_type == "DUAL":
            print(
                "Dual type has already stacked the files in the step of cross-correlation."
            )
        elif calculate_type == "MULTI":
            gen_stack_cmd(self.command, self.parameters)

    def deploy_stack_cmd(self):
        stack_cmd_deployer(self.parameters)

    def generate_rotate_list_dir(self):
        return gen_rotate_list_dir(self.SeisArrayInfo, self.parameters)

    def generate_rotate_cmd(self):
        gen_rotate_cmd(self.command, self.parameters)

    def deploy_rotate_cmd(self):
        rotate_cmd_deployer(self.parameters)

    def deploy_sac2dat(self):
        sac2dat_deployer(self.parameters)

    @staticmethod
    def generate_template_config(output_path="template_config.ini.copy"):
        """
        Generates a template configuration file at the given output path.

        :param output_path: Path to save the template config file. Default is "template_config.ini.copy".
        """
        # Determine the path to the template file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "template_config.ini")

        # Copy the template to the desired output location
        shutil.copy2(template_path, output_path)

        print(f"Template configuration file has been copied to: {output_path}")

    def run(self):
        # Run the entire process in sequence
        # generate filter
        self.generate_filter()

        # sac2spec
        self.generate_sac2spec_list_dir()
        self.generate_sac2spec_cmd()
        self.deploy_sac2spec_cmd()

        # cross correlation
        self.generate_xc_list_dir()
        self.generate_xc_cmd()
        self.deploy_xc_cmd()

        # stacking
        self.generate_stack_list_dir()
        self.generate_stack_cmd()
        self.deploy_stack_cmd()

        # rotating
        if self.generate_rotate_list_dir():
            self.generate_rotate_cmd()
            self.deploy_rotate_cmd()
