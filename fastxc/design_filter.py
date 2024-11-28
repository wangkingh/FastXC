import os
from typing import Dict
from scipy.signal import butter


def design_filter(xc_param: Dict):
    """
    Design filter for fastxc, save the filter coefficients to file.

    Parameters
    ----------
    xc_param : dict, including frequency bands, e.g. {'bands': '0.1/0.5 0.5/1.0 1.0/2.0'}
     : dict, including output_dir, e.g. {'output_dir': 'output'}

    Returns
    -------
    None
    """

    # Parsing parameters
    bands = xc_param['bands']  # frequency bands
    output_dir = xc_param['output_dir']  # output directory
    fs = 1.0 / xc_param['delta']  # sampling frequency
    f_nyq = fs / 2.0  # Nyquist frequency
    order = 2  # filter order

    # check output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'butterworth_filter.txt')
    bands_str = bands.split()

    all_freqs = []
    for band_str in bands_str:
        freq_low, freq_high = map(float, band_str.split('/'))
        all_freqs.append((freq_low, freq_high))

    # Get the overall min and max frequencies
    overall_min = min(freq[0] for freq in all_freqs)
    overall_max = max(freq[1] for freq in all_freqs)
    
    # Check if the overall band is valid
    if not (0 < overall_min < overall_max < f_nyq):
        print(f'Error: Overall frequency band {overall_min}/{overall_max} is not valid.')
        raise ValueError
    
    # Normalize frequencies
    overall_min_norm = overall_min / f_nyq
    overall_max_norm = overall_max / f_nyq

    # Design the overall bandpass filter
    b, a = butter(order, [overall_min_norm, overall_max_norm], btype='bandpass')
    
        # Write filters to file
    try:
        with open(output_file, 'w') as f:
            # Write the overall filter first
            f.write(f'# {overall_min}/{overall_max}\n')
            f.write('\t'.join(f'{b_i:.18e}' for b_i in b) + '\n')
            f.write('\t'.join(f'{a_i:.18e}' for a_i in a) + '\n')

            # Now write the individual band filters
            for band_str in bands_str:
                freq_low, freq_high = map(float, band_str.split('/'))
                freq_low_norm = freq_low / f_nyq
                freq_high_norm = freq_high / f_nyq
                b, a = butter(order, [freq_low_norm, freq_high_norm], btype='bandpass')

                line_b = '\t'.join(f'{b_i:.18e}' for b_i in b)
                line_a = '\t'.join(f'{a_i:.18e}' for a_i in a)

                f.write(f'# {band_str}\n')
                f.write(line_b + '\n')
                f.write(line_a + '\n')
    except IOError as e:
        print(f"Filter file writing error: {e}")


if __name__ == "__main__":
    design_filter({
        'bands': '0.2/0.5 0.6/0.8',  # 定义频率带
        'output_dir': './',          # 定义输出目录
        'delta': 0.01                # 定义采样间隔
    })

