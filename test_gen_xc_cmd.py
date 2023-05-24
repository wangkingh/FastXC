from FastXC.specxc_cmd_gen import SingleListCmdGen, DoubleListCmdGen
# 创建 SingleListCmdGen 实例
single_cmd_gen = SingleListCmdGen(virt_src_list='path/to/virt_src_list.txt', half_cc_length=0.5,
                                  output_directory='path/to/output', gpu_id=0)

# 设置其他参数
single_cmd_gen.checkhead = False
single_cmd_gen.do_cross_correlation = False
single_cmd_gen.save_by_pair = False

# 生成命令
command = single_cmd_gen.generate_command()
print(command)

# 检查参数有效性
single_cmd_gen.check_input()

# 打印已选参数
single_cmd_gen.print_selected_parameters()

# 创建 DoubleListCmdGen 实例
double_cmd_gen = DoubleListCmdGen(virt_src_list='path/to/virt_src_list.txt', virt_sta_list='path/to/virt_sta_list.txt',
                                  half_cc_length=1.0, output_directory='path/to/output', gpu_id=1)

# 设置其他参数
double_cmd_gen.checkhead = True
double_cmd_gen.do_cross_correlation = True
double_cmd_gen.save_by_pair = True

# 生成命令
command = double_cmd_gen.generate_command()
print(command)

# 检查参数有效性
double_cmd_gen.check_input()

# 打印已选参数
double_cmd_gen.print_selected_parameters()
