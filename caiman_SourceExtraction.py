try:
    get_ipython().magic(u'load_ext autoreload')
    get_ipython().magic(u'autoreload 2')
    get_ipython().magic(u'matplotlib qt')
except:
    pass
import matplotlib as mpl
mpl.use('Agg')
import logging
import matplotlib.pyplot as plt
import numpy as np
import glob
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


import sys,os
import pickle
import pandas as pd

#%% motion correction with concatenated videos
videoconcat = sys.argv[1]
newpath = os.path.dirname(videoconcat)
ms_ts_name = os.path.join(newpath,"ms_ts.pkl")
if os.path.exists(ms_ts_name):
    with open(ms_ts_name, "rb") as f:
        ms_ts= pickle.load(f)
else:
    print("there is no mt_ts.pkl existed")

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

motion_correct = False            # flag for motion correction
# motion correction parameters
pw_rigid = False                # flag for pw-rigid motion correction

gSig_filt = (8,8)   # size of filter, in general gSig (see below),
#                      change this one if algorithm does not work
max_shifts = (15,15)  # maximum allowed rigid shift
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
        #plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        #plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        #plt.legend(['x shifts', 'y shifts'])
        #plt.xlabel('frames')
        #plt.ylabel('pixels')

    bord_px = 0 if border_nan is 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=bord_px)
else:  # if no motion correction just memory map the file
    bord_px = 0
    #fname_new = cm.save_memmap(fnames, base_name='memmap_',order='C',border_to_0=bord_px)# there will be slightly different from using memmap* file
    fname_new = glob.glob(os.path.join(newpath,"memmap*.mmap"))[0]# it will get the same result with the MotionCorrection_SourceExtraction.py 
print('Motion correction has been done!')
m_els = cm.load(fname_mc)

# save motion corrected video as mat/tiff/hdf5 file

mc_name = os.path.join(newpath,"motioncorrected.tif")
m_els.save(mc_name)
#vid=np.array(m_els).astype('uint8')
#try:
#	from scipy.io import savemat
#	savemat(mc_name,{'vid':vid},format="5",do_compression=True)
#finally:
m_els.save(os.path.join(newpath,'ms_mc.avi'))
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

print(r">>>>>>>>>>start cnmf.CNMF")
# source extraction
cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, Ain=Ain, params=opts)
cnm.fit(images)
print(r">>>>>>>>>>finish cnm.fit(images)")


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
neuronsToPlot = 10

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
#plt.subplot(345); plt.plot(mc.shifts_rig); plt.title('Motion corrected shifts')
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
