import glob
import os
from shutil import copyfile
all_files=glob.glob('/home/qiushou/Documents/CHS_data/miniscope/raw_data/*/*/*/*Deep*csv')
all_except_videos = [i for i in all_files if "20190905" not in i]
except_videos = [i for i in all_files if "20190905" in i]

for video in all_except_videos:
    base = video.split("/raw_data")[1]
    dst_dir = r'/run/user/1000/gvfs/smb-share:server=10.10.46.135,share=lab_members/XuChun/Lab Projects/01_Intra Hippocampus/Miniscope_CFC/RawData'
    dst = dst_dir+base
    copyfile(video,dst)

    print(f"{video} \r finish copy!")
