from FastXC.ncfstack_cmd_gen import GenStackCmd

# Instantiate the class with the path to the NCF list file and the output directory
ncf_command = GenStackCmd("/path/to/ncf_list_file.txt",
                          "/path/to/output_directory")

# You can also set the properties manually
ncf_command.ncf_list_list = "/another/path/to/ncf_list_file.txt"
ncf_command.output_directory = "./output_directory"
ncf_command.normalize_output = False

# Check the input
ncf_command.check_input()

# Generate the command
command = ncf_command.generate_command()
print(command)  # Will print the generated command
