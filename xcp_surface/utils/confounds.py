# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""confound matrix selection based on Ciric et al 2007."""
import numpy as np
import nibabel as nb
import pandas as pd
import sys 

def load_confound(datafile):
    """`Load confound amd json."""
    '''
    datafile:
        real nifti or cifti file 
    confoundpd:
        confound data frame
    confoundjs: 
        confound json file
    '''
    confounds_timeseries = datafile.replace("_space-" + datafile.split("space-")[1],
                         "_desc-confounds_timeseries.tsv")
    confoundpd = pd.read_csv(confounds_timeseries, delimiter="\t", encoding="utf-8")
    confounds_json = datafile.replace("_space-" + datafile.split("space-")[1],
                         "_desc-confounds_timeseries.json")
    confoundjs = readjson(confounds_json)

    return confoundpd,confoundjs

def readjson(jsonfile):
    import json
    with open(jsonfile) as f:
        data = json.load(f)
    return data


def load_motion(confoundspd):
    """Load the 6 motion regressors."""
    motion_params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    return confoundspd[motion_params]

def load_globalS(confoundspd):
    """select global signal."""
    return confoundspd["global_signal"]

def load_WM_CSF(confoundspd):
    """select white matter and CSF nuissance."""
    return confoundspd[["csf","white_matter"]]

def load_acompcor(confoundspd, confoundjs):
    """ select WM and GM acompcor separately."""

    WM = []; CSF = []
    for key, value in confoundjs.items():
        if 'a_comp_cor' in key:
            if value['Mask']=='WM' and value['Retained']==True:
                WM.append([key,value['VarianceExplained']])
            if value['Mask']=='CSF' and value['Retained']==True:
                CSF.append([key,value['VarianceExplained']])
    # select the first five components
    csflist = []; wmlist = []
    for i in range(0,4):
        csflist.append(CSF[i][0])
        wmlist.append(WM[i][0])
    acompcor = wmlist +csflist  
    return confoundspd[acompcor]


def load_tcompcor(confoundspd, confoundjs):
    """ select tcompcor."""

    tcomp = []
    for key, value in confoundjs.items():
        if 't_comp_cor' in key:
            if value['Method']=='tCompCor' and value['Retained']==True:
                tcomp.append([key,value['VarianceExplained']])
    # sort it by variance explained
    # select the first five components
    tcomplist = [] 
    for i in range(0,6):
       tcomplist.append(tcomp[i][0])
    return confoundspd[tcomplist]


def derivative(confound):
    dat = confound.to_numpy()
    return pd.DataFrame(np.diff(dat,prepend=0))

def confpower(confound,order=2):
    return confound ** order


def load_confound_matrix(datafile,params='6P'):
    """ extract confound """
    '''
    datafile:
       cifti file or nifti file
    params: 
       confound requested based on Ciric et. al 2017
    '''
   
    confoundtsv,confoundjson = load_confound(datafile)
    if params == '2P':
        confound = load_WM_CSF(confoundtsv)
    elif params == '9P':
        motion = load_motion(confoundtsv)
        wmcsf = load_WM_CSF(confoundtsv)
        gs = load_globalS(confoundtsv)
        confound = pd.concat([motion,wmcsf,gs],axis=1)
    elif  params == '24P':
        motion = load_motion(confoundtsv)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        confound = pd.concat([mm_dev,confpower(mm_dev)],axis=1)
    elif params == '36P':
        motion = load_motion(confoundtsv)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        conf24p = pd.concat([mm_dev,confpower(mm_dev)],axis=1)
        gswmcsf = pd.concat([load_WM_CSF(confoundtsv),load_globalS(confoundtsv)],axis=1)
        gwcs_dev = pd.concat([gswmcsf,derivative(gswmcsf)],axis=1) 
        confound = pd.concat([conf24p,gwcs_dev,confpower(gwcs_dev)],axis=1)
    elif params == 'acompcor':
        motion = load_motion(confoundtsv)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        acompc = load_acompcor(confoundspd=confoundtsv, confoundjs=confoundjson)
        confound = pd.concat([mm_dev,acompc],axis=1)
    elif params == 'tcompcor':
        confound = load_tcompcor(confoundspd=confoundtsv,confoundjs=confoundjson)
    elif params == '6P':
        confound = load_motion(confoundtsv)

    return confound

    


