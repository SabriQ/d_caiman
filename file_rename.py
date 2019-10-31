import glob,os
from shutil import copyfile
xy_pathes = glob.glob(r'/run/user/1000/gvfs/smb-share:server=10.10.46.135,share=lab_members/XuChun/Lab Projects/01_Intra Hippocampus/Miniscope_CFC/RawData/*/*/*/xy.txt')
#%%
def add_prefix(xy_path,prefix="behave_video_"):
    dirname = os.path.dirname(xy_path)
    videoname=os.path.splitext(os.path.basename(glob.glob(os.path.join(dirname,"*[0-9].AVI"))[0]))[0]
    


#    tspath = glob.glob(os.path.join(dirname,"*ts.txt"))[0]
#    tsname = os.path.splitext(os.path.basename(glob.glob(os.path.join(dirname,"*ts.txt"))[0]))[0]
#
#    trackpath = glob.glob(os.path.join(dirname,"*1000000.csv"))[0]
#    trackname = os.path.splitext(os.path.basename(glob.glob(os.path.join(dirname,"*1000000.csv"))[0]))[0]
    
#    newtrackpath = os.path.join(dirname,prefix+str(trackname)+".csv")
#    new_tspath = os.path.join(dirname,prefix+tsname+".txt")
    new_xypath = os.path.join(dirname,prefix+str(videoname)+"_arena.txt")
    #copyfile(trackpath,newtrackpath)
    copyfile(xy_path,new_xypath)
    #copyfile(tspath,new_tspath)
    print(f"rename successfully!")
for xy_path in xy_pathes:
    add_prefix(xy_path)   
