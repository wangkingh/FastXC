from .arguproc import parse_arguments
from .group_files import group_files
from .process_a_group import process_a_group
from multiprocessing import Pool


def process_group_wrapper(args):
    # 解包参数，适用于 imap_unordered 的单参数传递
    group_file, args_dict = args
    return process_a_group(group_file, args_dict)


def sac2spec_pyv(command_line=None):
    print("\n\n[INFO]: This is a Python Version doing the Preprocessing of Data!!\n")
    args = parse_arguments(command_line)
    sac_lst = args.sac_lst
    spec_lst = args.spec_lst
    num_ch = args.num_ch
    thread_num = args.thread_num
    args_dict = vars(args)
    grouped_files = group_files(sac_lst, spec_lst, num_ch)
    # for grouped_file in grouped_files:  # Sequential processing
    #     process_a_group(grouped_file, args_dict)
    # 并行处理
    with Pool(thread_num) as pool:
        params = [(group_file, args_dict) for group_file in grouped_files]
        for param in params:
            pool.apply_async(process_group_wrapper, args=(param,))

        pool.close()
        pool.join()


if __name__ == "__main__":
    sac2spec_pyv()
