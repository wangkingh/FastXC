from FastXC import MakeSacList
from FastXC import MakeSpectrumList
from FastXC import MakeStackList
from FastXC import MakeStackListList
from FastXC import Sac2SpecCmdGen
from FastXC import SingleListCmdGen
from FastXC import GenStackCmd

# MakeSacList
sacdir = '/storage/HOME/yaolab/data/wjx/datas/tl-wf'
job1 = MakeSacList(sacdir)
job1.pattern = '{kstnm}.{YYYY}.{JJJ}.{HH}{MI}.{kcmpnm}.{suffix}'
job1.suffix = 'sac'
job1.start_time = '2017-08-01 00:00:00'
job1.end_time = '2017-08-09 00:00:00'
job1.kcmpnm = 'Z'
job1.station_list_file = './stalist.txt'

job1.check_inputs()
job1.print_criteria()
job1.match_files()
job1.write_matched_files('./saclist.txt')

