from FastXC.sac2spec_cmd_gen import Sac2SpecCmdGen

# 创建Sac2SpecCmdGen对象并设置参数
cmd_gen = Sac2SpecCmdGen(
    sac_list='path/to/sac_list.txt',
    output_directory='path/to/output',
    window_length=10
)
cmd_gen.gpu_list = [0, 1]
cmd_gen.freq_bands = ["0.2/0.5", "0.5/1", "1/2"]
cmd_gen.whiten_type = 'After'
cmd_gen.normalize_type = 'Onebit'
cmd_gen.skip_check_npts = False
cmd_gen.save_by_time = True
cmd_gen.nsmooth = 50

# 检查输入参数的有效性
cmd_gen.check_input()

# 打印选择的参数
cmd_gen.print_selected_parameters()

# 生成命令字符串
command = cmd_gen.generate_command()
print("Generated command: ", command)
