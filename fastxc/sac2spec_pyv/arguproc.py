import argparse
import shlex


def parse_arguments(command_line=None):
    parser = argparse.ArgumentParser(
        description="Process SAC files to generate spectrum files with optional GPU acceleration.",
        epilog="Last updated by wangjx@20240710",
    )
    parser.add_argument(
        "-I",
        dest="sac_lst",
        required=True,
        help="Input list of sac files of multi/single components.",
    )
    parser.add_argument(
        "-O",
        dest="spec_lst",
        required=True,
        help="Output spec list of segment spectrum files.",
    )

    # Optional arguments
    parser.add_argument(
        "-C", dest="num_ch", type=int, help="Number of channels to process."
    )
    parser.add_argument(
        "-B", dest="filter_file", help="File containing Butterworth filter parameters."
    )
    parser.add_argument(
        "-L",
        dest="seglen",
        type=float,
        required=True,
        help="Length of segment window in seconds, usually 7200s (2 hours).",
    )
    parser.add_argument("-G", dest="gpu_id", type=int, help="Index of GPU device.")
    parser.add_argument(
        "-F",
        dest="freq_band",
        type=parse_frequency_band,
        help="Frequency bands for spectral whitening in Hz, using the format f_low/f_high.",
    )
    parser.add_argument(
        "-W",
        dest="whiten_type",
        type=int,
        choices=[0, 1, 2, 3],
        help="Whiten type. 0: no whitening; 1: pre-time domain normalization; 2: post-time domain normalization; 3: both.",
    )
    parser.add_argument(
        "-N",
        dest="normalize_type",
        type=int,
        choices=[0, 1, 2],
        help="Normalization type. 0: no normalization; 1: runabs; 2: one-bit.",
    )
    parser.add_argument(
        "-Q",
        dest="skip_steps",
        type=parse_skip_steps,
        help="Skip segment steps. Use -1 to indicate the end of input sequence.",
    )
    parser.add_argument(
        "-T",
        dest="thread_num",
        type=int,
        default=1,
        help="GPU thread number. Default is 1.",
    )
    parser.add_argument("-U", dest="gpu_num", type=int, help="Number of GPUs to use.")
    
    if command_line:
        # 使用 shlex.split 来智能拆分命令行参数字符串
        args = parser.parse_args(shlex.split(command_line))
    else:
        # 没有提供 command_line 时，从命令行读取参数
        args = parser.parse_args()
        
    return args


def parse_frequency_band(value):
    try:
        freq_low, freq_high = map(float, value.split("/"))
        if freq_low >= freq_high:
            raise argparse.ArgumentTypeError("Invalid frequency band range")
        return freq_low, freq_high
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid frequency band format")


def parse_skip_steps(value):
    try:
        steps = []
        for val in value.split("/"):
            num = int(val)
            if num == -1:
                break  # Stop parsing if -1 is found
            steps.append(num)
        return steps
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid skip step format")


if __name__ == "__main__":
    args = parse_arguments()
    print(
        args
    )  # Just a placeholder for demonstration. Replace with actual function calls.
