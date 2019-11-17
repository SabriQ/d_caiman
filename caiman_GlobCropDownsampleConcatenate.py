try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
    get_ipython().magic(u'matplotlib qt')
except:
    pass
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format=
                          "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
                    # filename="/tmp/caiman.log",
                    level=logging.DEBUG)

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
import cv2

try:
    cv2.setNumThreads(0)
except:
    pass
import bokeh.plotting as bpl
import holoviews as hv
bpl.output_notebook()
#%%
import glob
import re
import os,sys
animal_id = "191174"
notes = '20191110-1112all'
common_dir = os.path.join(r'/run/user/1000/gvfs/smb-share:server=10.10.46.135,share=data_archive/qiushou/miniscope/2019111[0-2]',animal_id,"H*M*S*")
resultDir = '/home/qiushou/Documents/QS_data/miniscope/miniscope_result'
msFileList = glob.glob(os.path.join(common_dir,"msCam*.avi"))
tsFileList = glob.glob(os.path.join(common_dir,"timestamp.dat"))
show_cropped_img = False 
test = False 
def sort_key(s):
    if s:
        try:
            date = re.findall('\d{8}', s)[0]
        except:
            date = -1            
        try:
            H = re.findall('H(\d+)',s)[0]
        except:
            H = -1            
        try:
            M = re.findall('M(\d+)',s)[0]
        except:
            M = -1            
        try:
            S = re.findall('S(\d+)',s)[0]
        except:
            S = -1            
        try:
            ms = re.findall('msCam(\d+)',s)[0]
        except:
            ms = -1
        return [int(date),int(H),int(M),int(S),int(ms)]
msFileList.sort(key=sort_key)
tsFileList.sort(key=sort_key)
if test == True:
    msFileList = msFileList[0:2]
print(msFileList,tsFileList)


#%% get coordinates for crop videos


import moviepy.video as mpv
from moviepy.editor import *
import matplotlib as mpl
import cv2
import pickle

newpath_parent=os.path.join(resultDir,'Results'+'_'+animal_id)
if not os.path.exists(newpath_parent):
    os.makedirs(newpath_parent)
cropfilename=os.path.join(newpath_parent,'crop_param.pkl')
 
clip = VideoFileClip(msFileList[0])
im=clip.get_frame(1)
if not os.path.exists(cropfilename):
    r=cv2.selectROI(im,fromCenter=False)
    x1=int(r[0])
    x2=int(r[0]+r[2])
    y1=int(r[1])
    y2=int(r[1]+r[3])
    imcrop=im[y1:y2,x1:x2]
    if show_cropped_img:
        cv2.imshow('Image',imcrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('Done')
    np.shape(imcrop)
    crop_coord=[x1,x2,y1,y2]
    print(crop_coord)
    with open(cropfilename,'wb') as output:
        pickle.dump(crop_coord,output,pickle.HIGHEST_PROTOCOL)
else:
    with open(cropfilename, "rb") as f:
        r= pickle.load(f) 
    print(r)
    x1=r[0]
    x2=r[1]
    y1=r[2]
    y2=r[3]
    imcrop=im[y1:y2,x1:x2]
    if show_cropped_img:
        cv2.imshow('Image',imcrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print('Done')
    np.shape(imcrop)
    crop_coord=[x1,x2,y1,y2]
    print(crop_coord)

#%% concatenate videos in multi-threads
import datetime    
newpath=os.path.join(newpath_parent,datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'_'+notes)
if not os.path.exists(newpath):
    os.makedirs(newpath)                    
videoconcat=os.path.join(newpath,'msCam_concat.avi')
spatial_downsampling=2
temporal_downsampling=3
cropped_clip_list=[]
iframe=0
for video in msFileList:
    print('Concatenating '+video)
    clip = VideoFileClip(video)
    cropped_clip=mpv.fx.all.crop(clip,x1=x1,y1=y1,x2=x2,y2=y2)
    if spatial_downsampling!=1:
        cropped_clip=cropped_clip.resize(1/spatial_downsampling)
    cropped_clip_list.append(cropped_clip)
final_clip=concatenate_videoclips(cropped_clip_list)

if temporal_downsampling>1:
    final_clip=mpv.fx.all.speedx(final_clip, factor=temporal_downsampling)
final_clip.write_videofile(videoconcat,codec='rawvideo',audio=False,threads=8)

print(f'concatenated video is located at {videoconcat}')

#%% concatenate timestamps of tsFileList
#newpath=r'/home/qiushou/Documents/QS_data/miniscope/miniscope_result/Results_191172/20191110_160835_all'
ms_ts_name = os.path.join(newpath,'ms_ts.pkl')
print(ms_ts_name)

import pandas as pd
import pickle
if not os.path.exists(ms_ts_name):
    ts_session=[]
    for tsFile in tsFileList:
        datatemp=pd.read_csv(tsFile,sep = "\t", header = 0)
        ts_session.append(datatemp['sysClock'].values)    
    ttemp=np.hstack(ts_session)[::temporal_downsampling]
    # remporally downsample for each video
    # [i[::3] for i in ts_session][0]
    session_indend=(np.where(np.diff(ttemp)<0)[0]).tolist()
    session_indend.append(-1)
    ts_session_ds=[]
    i0=0
    session_indstart=[]
    for i in range(len(ts_session)):
        session_indstart.append(i0)
        ts_session_ds.append(ttemp[i0:session_indend[i]])
        i0=session_indend[i]+1
    ms_ts=np.array(ts_session_ds)    
    with open(ms_ts_name,'wb') as output:
        pickle.dump(ms_ts,output,pickle.HIGHEST_PROTOCOL)
else:
    with open(ms_ts_name, "rb") as f:
        ms_ts= pickle.load(f)
#print(f'concatenated timestamp of miniscope_video is located at {ms_ts_name}')

