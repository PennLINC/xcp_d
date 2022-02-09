# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""confound matrix selection based on Ciric et al 2007."""
import numpy as np
import pandas as pd
import os  

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
    if 'space' in os.path.basename(datafile):
        confounds_timeseries = datafile.replace("_space-" + datafile.split("space-")[1],
                         "_desc-confounds_timeseries.tsv")
        confounds_json = datafile.replace("_space-" + datafile.split("space-")[1],
                         "_desc-confounds_timeseries.json")
    else:
        confounds_timeseries = datafile.split('_desc-preproc_bold.nii.gz')[0]+"_desc-confounds_timeseries.tsv"
        confounds_json = datafile.split('_desc-preproc_bold.nii.gz')[0]+"_desc-confounds_timeseries.json"
            
    confoundpd = pd.read_csv(confounds_timeseries, delimiter="\t", encoding="utf-8")
    
    confoundjs = readjson(confounds_json)

    return confoundpd,confoundjs

def readjson(jsonfile):
    import json
    with open(jsonfile) as f:
        data = json.load(f)
    return data


def load_motion(confoundspd,TR,head_radius,filtertype,
        cutoff=0.1,freqband=[0.1,0.2],order=4):
    """Load the 6 motion regressors."""
    fs = 1/TR
    head_radius = head_radius
    rot_2mm = confoundspd[["rot_x", "rot_y", "rot_z"]]*head_radius
    trans_mm = confoundspd[["trans_x", "trans_y", "trans_z"]]
    datay = pd.concat([rot_2mm,trans_mm],axis=1).to_numpy()
    datay = datay.T 
    if filtertype == 'lp' or filtertype == 'notch' :
        datay = motion_regression_filter(data=datay,fs=fs,
          filtertype=filtertype,cutoff=cutoff,freqband=freqband,order=order)
    return  pd.DataFrame(datay.T) 

def load_globalS(confoundspd):
    """select global signal."""
    return confoundspd["global_signal"]

def load_WM_CSF(confoundspd):
    """select white matter and CSF nuissance."""
    return confoundspd[["csf","white_matter"]]

def load_cosine(confoundspd):
    """select cosine for compcor"""
    return confoundspd[["cosine00","cosine01","cosine02","cosine03","cosine04","cosine05"]]

def load_acompcor(confoundspd, confoundjs):
    """ select WM and GM acompcor separately."""

    WM = []; CSF = []
    for key, value in confoundjs.items():
        if 'comp_cor' in key and 't' not in key:
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


def load_confound_matrix(datafile,TR,filtertype,cutoff=0.1,order=4,
                        freqband=[0.1,0.2],head_radius=50,params='27P'):
                    
    """ extract confound """
    '''
    datafile:
       cifti file or nifti file
    params: 
       confound requested based on Ciric et. al 2017
    '''
    confoundtsv,confoundjson = load_confound(datafile)
    if  params == '24P':
        motion = load_motion(confoundtsv,TR,head_radius,filtertype,
        cutoff=cutoff,freqband=freqband,order=order)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        confound = pd.concat([mm_dev,confpower(mm_dev)],axis=1)
    elif  params == '27P':
        motion = load_motion(confoundtsv,TR,head_radius,filtertype,
        cutoff=cutoff,freqband=freqband,order=order)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        wmcsf = load_WM_CSF(confoundtsv)
        gs = load_globalS(confoundtsv)
        confound = pd.concat([mm_dev,confpower(mm_dev),wmcsf,gs],axis=1)
    elif params == '36P':
        motion = load_motion(confoundtsv,TR,head_radius,filtertype,
        cutoff=cutoff,freqband=freqband,order=order)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        conf24p = pd.concat([mm_dev,confpower(mm_dev)],axis=1)
        gswmcsf = pd.concat([load_WM_CSF(confoundtsv),load_globalS(confoundtsv)],axis=1)
        gwcs_dev = pd.concat([gswmcsf,derivative(gswmcsf)],axis=1) 
        confound = pd.concat([conf24p,gwcs_dev,confpower(gwcs_dev)],axis=1)
    elif params == 'acompcor':
        motion = load_motion(confoundtsv,TR,head_radius,filtertype,
        cutoff=cutoff,freqband=freqband,order=order)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        acompc = load_acompcor(confoundspd=confoundtsv, confoundjs=confoundjson)
        cosine = load_cosine(confoundtsv)
        confound = pd.concat([mm_dev,acompc,cosine],axis=1)
    elif params == 'aroma':
        wmcsf=load_WM_CSF(confoundtsv)
        aroma = load_aroma(datafile=datafile)
        pd.concat([wmcsf,aroma],axis=1)
    elif params == 'aroma_gsr':
        wmcsf=load_WM_CSF(confoundtsv)
        aroma = load_aroma(datafile=datafile)
        gs = load_globalS(confoundtsv)
        pd.concat([wmcsf,aroma,gs],axis=1)
    elif params == 'acompcor_gsr':
        motion = load_motion(confoundtsv,TR,head_radius,filtertype,
        cutoff=cutoff,freqband=freqband,order=order)
        mm_dev = pd.concat([motion,derivative(motion)],axis=1)
        acompc = load_acompcor(confoundspd=confoundtsv, confoundjs=confoundjson)
        gs = load_globalS(confoundtsv)
        cosine = load_cosine(confoundtsv)
        confound = pd.concat([mm_dev,acompc,gs,cosine],axis=1)
    elif params =='custom':
        confound = pd.DataFrame() #for custom confounds with no other confounds
        
        
    return confound

def load_aroma(datafile):
    """ extract aroma confound."""
    #_AROMAnoiseICs.csv
    #_desc-MELODIC_mixing.tsv
    #
    if 'space' in os.path.basename(datafile):
        aroma_noise = datafile.replace("_space-" + datafile.split("space-")[1],
                         "_AROMAnoiseICs.csv")
        melodic_ts = datafile.replace("_space-" + datafile.split("space-")[1],
                         "_desc-MELODIC_mixing.tsv")
    else:
        aroma_noise = datafile.split('_desc-preproc_bold.nii.gz')[0]+"_AROMAnoiseICs.csv"
        melodic_ts = datafile.split('_desc-preproc_bold.nii.gz')[0]+"_desc-MELODIC_mixing.tsv"

    aroma_noise =np.genfromtxt(aroma_noise,delimiter=',',)
    aroma_noise = [np.int(i) -1  for i in  aroma_noise] # change to 0-based index
    melodic = pd.read_csv(melodic_ts,header=None, delimiter="\t", encoding="utf-8")
    aroma = melodic.drop(aroma_noise, axis=1)

    return aroma




def motion_regression_filter(data,fs,filtertype,cutoff,freqband,order=4):
    """
    apply motion filter to 6 motion.
    """


    from scipy.signal import firwin,iirnotch,filtfilt

    def lowpassfilter_coeff(cutoff, fs, order=4):
        
        nyq = 0.5 * fs
        fa = np.abs( cutoff - np.floor((cutoff + nyq) / fs) * fs)
        normalCutoff = fa / nyq
        b = firwin(order, cutoff=normalCutoff, window='hamming') 
        a = 1
        return b, a

    def iirnortch_coeff(freqband,fs):
        nyq = 0.5*fs
        fa = np.abs(freqband - np.floor((np.add(freqband,nyq) / fs) * fs))
        w0 = np.mean(fa)/nyq
        bw = np.diff(fa)/nyq
        qf = w0 / bw
        b, a = iirnotch( w0, qf )
        return b,a

    if filtertype == 'lp':
        b,a = lowpassfilter_coeff(cutoff,fs,order=4)
    elif filtertype =='notch':
        b,a = iirnortch_coeff(freqband,fs=fs)
    
    order_apply = np.int(np.floor(order/2))

    for j in range(order_apply):
        for k in range(data.shape[0]):
            data[k,:] = filtfilt(b,a,data[k,:])
        j=j+1
    
    return data 
    
    





