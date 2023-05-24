from FastXC.make_sac_list import MakeSacList

# create MakeSacList object
path_to_sac_files = '/storage/HOME/yaolab/data/wjx/datas/tl-wf'
sac_list = MakeSacList(sac_directory=path_to_sac_files)

# set pattern
sac_list.pattern = '{kstnm}.{YYYY}.{JJJ}.{HH}{MI}.{kcmpnm}.{suffix}'


# set criteria
# sac_list.network_list = ['NET1', 'NET2']  # only match network code in the list
sac_list.kcmpnm = 'U'  # only match kcmpnm
# only match files after the start time
sac_list.start_time = '2017-08-01 00:00:00'
# only match files before the end time
# sac_list.end_time = '2017-09-02 23:59:59'
sac_list.suffix = 'sac'  # only match suffix

# check inputs
sac_list.check_inputs()

# print criteria
sac_list.print_criteria()

# match files
matched_files = sac_list.match_files()

# for file in matched_files:
#    print(file)

# write matched files to a list
path_to_sac_list = './sac_list.txt'
sac_list.write_matched_files(path_to_sac_list)
