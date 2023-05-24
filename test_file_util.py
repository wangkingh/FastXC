import os
from FastXC.file_list_utils import MakeStackList
from FastXC.file_list_utils import MatchDoubleList
from FastXC.file_list_utils import MakeStackListList
from FastXC.file_list_utils import MakeSpectrumList

# create dummy directories and files
os.makedirs("spectrum_directory/subdir", exist_ok=True)
os.makedirs("ncf_directory/subdir", exist_ok=True)


# create dummy .segspec files for spectrum directory
with open("spectrum_directory/subdir/file1.segspec", 'w') as f:
    f.write("dummy data")

# create dummy .sac files for ncf directory
with open("ncf_directory/subdir/file1.sac", 'w') as f:
    f.write("dummy data")

# use the functions

# MakeSpectrumList
MakeSpectrumList("spectrum_directory", "spectrum_list_directory", num_threads=1)

# MakeStackList
MakeStackList("ncf_directory", "stack_list_directory", num_threads=1)

# MakeStackListList
MakeStackListList("stack_list_directory", "stack_list_list.txt")

# MatchDoubleList
matched_files = MatchDoubleList("spectrum_list_directory", "stack_list_directory")
print(matched_files)
