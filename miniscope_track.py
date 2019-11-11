# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:05:55 2019

@author: Sabri
"""
#%% build a videolists which contains what we want:
#    all_videolists
#    no_ts_videos
#    untracked_videos
import glob,os 
#video_dir = glob.glob(r"")
#%%
data_dir = r"/run/user/1000/gvfs/smb-share:server=10.10.46.135,share=share/Qiushou/12_Miniscope/Raw_data/20191110/*"
#data_dir2= r"/run/user/1000/gvfs/smb-share:server=10.10.46.135,share=share/zhangna/3. EPM and open field/open_field"
#all_videolists = glob.glob(data_dir)
all_videolists = glob.glob(data_dir+r'/*[0-9].mp4')
#all_videolists2 = glob.glob(data_dir2+r'/*[0-9].mp4')
ts_txts  = glob.glob(data_dir+r'/*_ts.txt')
tracked_h5s = glob.glob(data_dir+r'/*Deep*.h5')
print("all_videos: ",len(all_videolists))
#print(len(ts_txts))
#print(len(tracked_h5s))
#%%

def dst_videolists(all_videolists,ts_txts,tracked_h5s):
    ts_videos = []
    tracked_videos = []
    for video in all_videolists :
        basename = os.path.splitext(os.path.basename(video))[0]
#        print(basename)
        for ts_txt in ts_txts:
#            print(ts_file)
            if basename in ts_txt:
                ts_videos.append(video)
#                print(">>>>>>"+video)
        for tracked_h5 in tracked_h5s:
#            print(tracked_video)
            if basename in tracked_h5:
                tracked_videos.append(video)
#                print("<<<<<<"+video)
    return [ i for i in all_videolists if i not in ts_videos],[ i for i in all_videolists if i not in tracked_videos]

no_ts_videos, untracked_videos =  dst_videolists(all_videolists,ts_txts,tracked_h5s)
print("no_ts_videos:",len(no_ts_videos))
print("untracted_videos: ",len(untracked_videos))

#%%  generate *_ts.txt 
import platform, subprocess
def generate_ts_txt(videoPath):
    tsPath = os.path.splitext(videoPath)[0]+'_ts.txt'
    if (platform.system()=="Linux"):
        command = 'ffprobe -i %s -show_frames -select_streams v  -loglevel quiet| grep pkt_pts_time= | cut -c 14-24 > %s' % (videoPath,tsPath)
        child = subprocess.Popen(command,shell=True)
    if (platform.system()=="Windows"):
        powershell=r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        # command below relys on powershell so we open powershell with a process named child and input command through stdin way.
        child = subprocess.Popen(powershell,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        command = r'ffprobe.exe -i %s -show_frames -loglevel quiet |Select-String media_type=video -context 0,4 |foreach{$_.context.PostContext[3] -replace {.*=}} |Out-File %s' % (videoPath, tsPath)
        child.stdin.write(command.encode('utf-8'))
        out = child.communicate()[1].decode('gbk') # has to be 'gbk'
        print(out)
        child.wait()
#generate_ts_txt(no_ts_videos[0])
for video in no_ts_videos:
    generate_ts_txt(video)
    print(f"{video} is generating ts file...")
#%% generate  tracked file
def generate_track_h5(config_path,untracked_videos,suffix=".mp4"):    
    os.environ["DLClight"]='True' # all the child process have this env but not its father process
    if (platform.system()=="Linux"):
        import deeplabcut
        deeplabcut.analyze_videos(config_path,untracked_videos,shuffle=1,save_as_csv=True,videotype=suffix)
        deeplabcut.plot_trajectories(config_path,untracked_videos)
        deeplabcut.create_labeled_video(config_path,untracked_videos)
#print(video_dir)
generate_track_h5(config_path=r'/home/qiushou/Documents/dlcmodels/linear_track_40cm_AB-QS-2019-09-26/config.yaml',untracked_videos=untracked_videos,suffix=".mp4")
#generate_track_h5(config_path=r'/home/qiushou/Documents/dlcmodels/epm-ZN-2019-07-25/config.yaml',untracked_videos=all_videolists,suffix=".mp4")
#generate_track_h5(config_path=r'/home/qiushou/Documents/dlcmodels/open_field-ZN-2019-07-25/config.yaml',untracked_videos=all_videolists2,suffix=".mp4")
