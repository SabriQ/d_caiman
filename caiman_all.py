#import time
#time.sleep(1800)
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



import glob
import re
import os,sys
animal_id = sys.argv[1]
notes = '10fps'
common_dir = os.path.join('/run/user/1000/gvfs/smb-share:server=10.10.46.135,share=share/ZhuXinyue/003_VR/003_2_Miniscope/AVI/20191226',animal_id)
resultDir = '/home/qiushou/Documents/ZXY_data/miniscope/miniscope_result'
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
#    session_indend.append(-1)
    ts_session_ds=[]
    i0=0
    session_indstart=[]
    if len(session_indend)>0:
        for i in range(len(session_indend)):
            session_indstart.append(i0)
            ts_session_ds.append(ttemp[i0:(session_indend[i]+1)])
            i0=session_indend[i]+1
        ts_session_ds.append(ttemp[(session_indend[-1]+1):])
    else:
        ts_session_ds.append(ttemp[i0:])
    
    
    ms_ts=np.array(ts_session_ds)    
    with open(ms_ts_name,'wb') as output:
        pickle.dump(ms_ts,output,pickle.HIGHEST_PROTOCOL)
else:
    with open(ms_ts_name, "rb") as f:
        ms_ts= pickle.load(f)

timestamps_frames = sum([len(i) for i in ms_ts])
cap = cv2.VideoCapture(videoconcat)
concated_video_frames = int(cap.get(7))
cap.release()
if timestamps_frames == concated_video_frames:
    print("concatenated video and timestamps have the same frame_Nos")
else:
    print(f"Attention: concatenated video {concated_video_frames}frames and timestamps {timestamps_frames}frames have different frame_Nos")
print(f'concatenated timestamp of miniscope_video is located at {ms_ts_name}')

#%% #for motion correction                                                                                                        
fnames=[videoconcat]      
m_orig = cm.load_movie_chain(fnames)
# start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():   
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
                          
# dataset dependent parameters
fr = 10    # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds
                          
motion_correct = True            # flag for motion correction
# motion correction parameters
pw_rigid = False                # flag for pw-rigid motion correction
                          
gSig_filt = (8,8)   # size of filter, in general gSig (see below),
#                      change this one if algorithm does not work
max_shifts = (15,15)  # maximum allowed rigid shift ## generally it's (15,15) 
strides = (96,96)  # start a new patch for pw-rigid motion correction every x pixels
overlaps = (32,32)  # overlap between pathes (size of patch strides+overlaps)
# maximum deviation allowed for patch with respect to rigid shifts
max_deviation_rigid = 5   
border_nan = 'copy'       
                          
mc_dict = {               
    'fnames': fnames,     
    'fr': fr,             
    'decay_time': decay_time,
    'pw_rigid': pw_rigid, 
    'max_shifts': max_shifts,
    'gSig_filt': gSig_filt,
    'strides': strides,   
    'overlaps': overlaps, 
    'max_deviation_rigid': max_deviation_rigid,
    'border_nan': border_nan
}                         
                          
opts = params.CNMFParams(params_dict=mc_dict)


if motion_correct:        
    # do motion correction rigid
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:          
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(np.int)
    else:                 
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)
        plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')
        plt.savefig(newpath + r'/' + "shift.pdf",edgecolor='w',format='pdf',transparent=True) 
                          
    bord_px = 0 if border_nan is 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=bord_px)
else:  # if no motion correction just memory map the file
    fname_new = cm.save_memmap(filename_reorder, base_name='memmap_',
                               order='C', border_to_0=0, dview=dview)
                          
print('Motion correction has been done!')
m_els = cm.load(fname_mc) 
                          
# save motion corrected video as mat file                         

                          
m_els.save(os.path.join(newpath,'ms_mc.avi'))
#mc_name = os.path.join(newpath,"motioncorrected.tif")
m_els.save(mc_name)

del m_els                 
cm.stop_server(dview=dview)
                          

#%% processing for source extraction
# start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():   
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
                          
# load memory mappable file
Yr, dims, T = cm.load_memmap(fname_new)
images = Yr.T.reshape((T,) + dims, order='F')
# parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = None            # upper bound on number of components per patch, in general None
gSig = (3,3)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (13,13)     # average diameter of a neuron, in general 4*gSig+1
Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .65      # merging threshold, max correlation allowed
rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 1            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .85       # min peak value from correlation image
min_pnr = 10        # min peak to noise ration from PNR image
ssub_B = 2          # additional downsampling factor in space for background
ring_size_factor = 1.5  # radius of ring is gSiz*ring_size_factor

opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': True,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                'center_psf': True,                    # leave as is for 1 photon
                                'ssub_B': ssub_B,
                                'ring_size_factor': ring_size_factor,
                                'del_duplicates': True,                # whether to remove duplicates from initialization
                                'border_pix': bord_px})                # number of pixels to not consider in the borders)
                          
                          
cn_filter, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=gSig[0], swap_dim=False) # change swap dim if output looks weird, it is a problem with tiffile
# inspect the summary images and set the parameters
#inspect_correlation_pnr(cn_filter, pnr)
                          
                          
# source extraction       
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
cnm.fit(images)           
                          

# plt(cnm.estimates.C[0],cnm.estimates.C.dim
old_RawTraces= cnm.estimates.C
# plt.plot(old_C[0,:],'k')
old_DeconvTraces = cnm.estimates.S
# plt.plot(old_S[0,:],'k')
# old_S                   
#%% COMPONENT EVALUATION  ## necessary for generate "idx_accepted = cnm.estimates.idx_components"
# the components are evaluated in three ways:
#   a) the shape of each component must be correlated with the data
#   b) a minimum peak SNR is required over the length of a transient
#   c) each shape passes a CNN based classifier
                          
min_SNR = 10           # adaptive way to set threshold on the transient size
r_values_min = 0.85    # threshold on space consistency (if you lower more components
#                        will be accepted, potentially with worst quality)
cnm.params.set('quality', {'min_SNR': min_SNR,
                           'rval_thr': r_values_min,
                           'use_cnn': False})
cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
                          
print(' ***** ')          
print('Number of total components: ', len(cnm.estimates.C))
print('Number of accepted components: ', len(cnm.estimates.idx_components))
# for saving final_result.avi
#try:                      
#    cnm.estimates.play_movie(images, q_max=99.9, magnification=1,include_bck=False,frame_range=slice(0,1000,1),                             thr=1, gain_res=1,bpx=bord_px,save_movie=True,movie_name=os.path.join(newpath,'final_result.avi'))
#except:                   
#    print("could not play video here!")


# detrend                 
cnm.estimates.detrend_df_f()
cnm.estimates.deconvolve(cnm.params,dview=dview,dff_flag=True)
                          
cnm.save(os.path.join(newpath,'result.hdf5'))
                          
#%%matplotlib inline      
                          
#How many neurons to plot 
neuronsToPlot = 30        
                          
DeconvTraces = cnm.estimates.S
RawTraces = cnm.estimates.C
SFP = cnm.estimates.A     
SFP_dims = list(dims)     
SFP_dims.append(SFP.shape[1])
print('Spatial foootprints dimensions (height x width x neurons): ' + str(SFP_dims))
numNeurons = SFP_dims[2]  
idx_accepted=cnm.estimates.idx_components
idx_deleted=cnm.estimates.idx_components_bad
dff=cnm.estimates.F_dff[idx_accepted,:]
S_dff=cnm.estimates.S_dff[idx_accepted,:]
SFP = np.reshape(SFP.toarray(), SFP_dims, order='F')
                          
maxRawTraces = np.amax(RawTraces)
plt.figure(figsize=(30,15))
plt.subplot(341);         
plt.subplot(345); plt.plot(mc.shifts_rig); plt.title('Motion corrected shifts')
plt.subplot(3,4,9);       
plt.subplot(3,4,2); plt.imshow(cn_filter); plt.colorbar(); plt.title('Correlation projection')
plt.subplot(3,4,6); plt.imshow(pnr); plt.colorbar(); plt.title('PNR')
plt.subplot(3,4,10); plt.imshow(np.amax(SFP,axis=2)); plt.colorbar(); plt.title('Spatial footprints')
                          
plt.subplot(2,2,2); plt.figure; plt.title(f'Example traces (first {neuronsToPlot} cells)')
plot_gain = 10 # To change the value gain of traces
if numNeurons >= neuronsToPlot:
    for i in range(neuronsToPlot):
        if i == 0:        
          plt.plot(RawTraces[i,:],'k')
        else:             
          trace = RawTraces[i,:] + maxRawTraces*i/plot_gain
          plt.plot(trace,'k')
else:                     
    for i in range(numNeurons):
        if i == 0:        
          plt.plot(RawTraces[i,:],'k')
        else:             
          trace = RawTraces[i,:] + maxRawTraces*i/plot_gain
          plt.plot(trace,'k')
plt.subplot(2,2,4); plt.figure; plt.title(f'Deconvolved traces (first {neuronsToPlot} cells)')
plot_gain = 20 # To change the value gain of traces
if numNeurons >= neuronsToPlot:
    for i in range(neuronsToPlot):
        if i == 0:       
          plt.plot(DeconvTraces[i,:],'k')
        else:            
          trace = DeconvTraces[i,:] + maxRawTraces*i/plot_gain
          plt.plot(trace,'k')
else:                    
    for i in range(numNeurons):
        if i == 0:       
          plt.plot(DeconvTraces[i,:],'k')
        else:            
          trace = DeconvTraces[i,:] + maxRawTraces*i/plot_gain
          plt.plot(trace,'k')
# Save summary figure
plt.savefig(newpath + '/' + 'summary_figure.pdf', edgecolor='w', format='pdf', transparent=True)

save_mat=True            
if save_mat:             
    from scipy.io import savemat
                         
    results_dict={       
        'height':dims[0],
        'width':dims[1], 
        'CorrProj':cn_filter,
        'PNR':pnr,       
        'old_sigraw':old_RawTraces.conj().transpose(),
        'old_sigdeconvolved':old_DeconvTraces.conj().transpose(),
        'sigraw':RawTraces.conj().transpose(),
        'sigdeconvolved':DeconvTraces.conj().transpose(),
        'SFP':SFP,       
        'numNeurons':SFP_dims[2],
        'ms_ts':ms_ts,  # minisope timestamps, here no,  
        'dff':dff,
        'S_dff':S_dff,   
        'idx_accepted':idx_accepted,
        'idx_deleted':idx_deleted
    }             
    SFPperm = np.transpose(SFP,[2,0,1])
    savemat(newpath + '/SFP.mat', {'SFP': SFPperm})
    savemat(newpath + '/ms.mat', {'ms': results_dict})
    print('.mat Files saved!')
                  
cm.stop_server(dview=dview)
                  
print("All done")                    
