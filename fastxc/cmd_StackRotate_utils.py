from typing import List, Dict
import os


def gen_linear_stack_cmd(command_dict: Dict, xc_param: Dict) -> List[str]:
    """
    Generate commands for stack
    """
    cmd_list = []
    output_dir = xc_param["output_dir"]
    command = command_dict["stack"]

    stack_list_dir = os.path.join(output_dir, "stack_list")
    stack_dir = os.path.join(output_dir, "stack", "linear")
    cmd_list_dir = os.path.join(output_dir, "cmd_list")

    stack_lists = [
        os.path.join(stack_list_dir, fname) for fname in os.listdir(stack_list_dir)
    ]

    os.makedirs(stack_dir, exist_ok=True)

    for stack_list in stack_lists:
        stack_list_name = os.path.basename(stack_list)
        info = stack_list_name.split(".")
        output_fname = f"{info[0]}.{info[1]}.ncf.sac"
        output_path = os.path.join(stack_dir, info[0], output_fname)

        cmd = f"{command} -I {stack_list} -O {output_path} "

        cmd_list.append(cmd)

    os.makedirs(cmd_list_dir, exist_ok=True)
    cmd_list_path = os.path.join(cmd_list_dir, "stack_cmds.txt")
    with open(cmd_list_path, "w") as f:
        f.write("\n".join(cmd_list))
    return cmd_list


def gen_stack_cmd(command_dict: Dict, xc_param: Dict) -> List[str]:
    save_flag = xc_param["save_flag"]
    save_linear = bool(int(save_flag[0]))
    save_pws = bool(int(save_flag[1]))
    save_tfpws = bool(int(save_flag[2]))
    save_segments = bool(int(save_flag[3]))
    if save_linear:
        gen_linear_stack_cmd(command_dict, xc_param)
    if save_pws:
        pass
    if save_tfpws:
        pass
    if save_segments:
        pass


# generate terminal rotate cmd
def gen_rotate_cmd(command_dict: Dict, xc_param: Dict) -> List[str]:
    cmd_list = []
    output_dir = xc_param["output_dir"]
    rotate_list_dir = os.path.join(output_dir, "rotate_list")
    cmd_list_dir = os.path.join(output_dir, "cmd_list")
    command = command_dict["rotate"]

    for rotate_list in os.listdir(rotate_list_dir):
        # Get first key in the enz_group dictionary
        target_dir = os.path.join(rotate_list_dir, rotate_list)
        inlist = os.path.join(target_dir, "enz_list.txt")
        outlist = os.path.join(target_dir, "rtz_list.txt")
        cmd = f"{command} -I {inlist} -O {outlist}"
        cmd_list.append(cmd)

    os.makedirs(cmd_list_dir, exist_ok=True)
    with open(os.path.join(cmd_list_dir, "rotate_cmds.txt"), "w") as f:
        f.write("\n".join(cmd_list))

    return cmd_list
