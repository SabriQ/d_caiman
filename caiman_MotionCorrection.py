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
fr = 30    # movie frame rate
decay_time = 0.4                 # length of a typical transient in seconds

motion_correct = True            # flag for motion correction
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
        plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels');plt.savefig(newpath + r'/' + "shift.pdf",edgecolor='w',format='pdf',transparent=True)

    bord_px = 0 if border_nan is 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=bord_px)
else:  # if no motion correction just memory map the file
    fname_new = cm.save_memmap(filename_reorder, base_name='memmap_',
                               order='C', border_to_0=0, dview=dview)

print('Motion correction has been done!')
m_els = cm.load(fname_mc)

# save motion corrected video as mat file

mc_name = os.path.join(newpath,"motioncorrected.tif")
#vid=np.array(m_els).astype('uint8')
#from scipy.io import savemat
#savemat(mc_name,{'vid':vid},format="5",do_compression=True)
m_els.save(mc_name)
m_els.save(os.path.join(newpath,'ms_mc.avi'))
del m_els
cm.stop_server(dview=dview)

