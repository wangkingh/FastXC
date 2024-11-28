from fastxc.fastxc import FastXC

ini_file = "./config/test.ini"

job = FastXC(ini_file)

# generate filter
job.generate_filter()

# sac2spec
job.generate_sac2spec_list_dir()

job.generate_sac2spec_cmd()

job.deploy_sac2spec_cmd()

# xc
job.generate_xc_list_dir()

job.generate_xc_cmd()

job.deploy_xc_cmd()

# stack
job.generate_stack_list_dir()

job.generate_stack_cmd()

job.deploy_stack_cmd()

# rotate
if job.generate_rotate_list_dir():
    job.generate_rotate_cmd()
    job.deploy_rotate_cmd()

# sac2dat
job.deploy_sac2dat()
