# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""confound matrix selection based on Ciric et al 2007."""
import numpy as np
import pandas as pd
import os
from scipy.signal import firwin, iirnotch, filtfilt


def get_confounds_tsv(datafile):
    """Find path to confounds.tsv """
    '''
    datafile:
        real nifti or cifti file
    confounds_timeseries:
        confound tsv file
    '''
    if 'space' in os.path.basename(datafile):
        confounds_timeseries = datafile.replace("_space-" + datafile.split("space-")[1],
                                                "_desc-confounds_timeseries.tsv")
    else:
        confounds_timeseries = datafile.split(
            '_desc-preproc_bold.nii.gz')[0]+"_desc-confounds_timeseries.tsv"

    return confounds_timeseries


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
        confounds_timeseries = datafile.replace(
            "_space-" + datafile.split("space-")[1],
            "_desc-confounds_timeseries.tsv")
        confounds_json = datafile.replace(
            "_space-" + datafile.split("space-")[1],
            "_desc-confounds_timeseries.json")
    else:
        confounds_timeseries = datafile.split(
            '_desc-preproc_bold.nii.gz')[0] + "_desc-confounds_timeseries.tsv"
        confounds_json = datafile.split(
            '_desc-preproc_bold.nii.gz')[0] + "_desc-confounds_timeseries.json"

    confoundpd = pd.read_csv(confounds_timeseries,
                             delimiter="\t",
                             encoding="utf-8")

    confoundjs = readjson(confounds_json)

    return confoundpd, confoundjs


def readjson(jsonfile):
    import json
    with open(jsonfile) as f:
        data = json.load(f)
    return data


def load_motion(confounds_df, TR, motion_filter_type, freqband, cutoff=0.1, motion_filter_order=4):
    """Load the 6 motion regressors."""

    # Pull out rot and trans values and concatenate them
    rot_values = confounds_df[["rot_x", "rot_y", "rot_z"]]
    trans_values = confounds_df[["trans_x", "trans_y", "trans_z"]]
    motion_confounds = pd.concat([rot_values, trans_values], axis=1).to_numpy()

    # Apply LP or notch filter
    if motion_filter_type == 'lp' or motion_filter_type == 'notch':
        motion_confounds = motion_confounds.T
        motion_confounds = motion_regression_filter(data=motion_confounds,
                                                    TR=TR,
                                                    motion_filter_type=motion_filter_type,
                                                    freqband=freqband,
                                                    cutoff=cutoff,
                                                    motion_filter_order=motion_filter_order)
        motion_confounds = motion_confounds.T  # Transpose motion confounds
    return pd.DataFrame(motion_confounds)


def load_global_signal(confounds_df):
    """select global signal."""
    return confounds_df["global_signal"]


def load_WM_CSF(confounds_df):
    """select white matter and CSF nuissance regressors."""
    return confounds_df[["csf", "white_matter"]]


def load_cosine(confounds_df):
    """select cosine values for compcor"""
    cosine = []
    for key in confounds_df.keys():  # Any colums with cosine
        if 'cosine' in key:
            cosine.append(key)
    return confounds_df[cosine]


def load_acompcor(confounds_df, confoundjs):
    """ select WM and GM acompcor separately."""

    WM = []
    CSF = []
    for key, value in confoundjs.items():  # Use the confounds json
        if 'comp_cor' in key and 't' not in key:
            # Pull out variance explained for white matter masks that are retained
            if value['Mask'] == 'WM' and value['Retained']:
                WM.append([key, value['VarianceExplained']])
            # Pull out variance explained for CSF masks that are retained
            if value['Mask'] == 'CSF' and value['Retained']:
                CSF.append([key, value['VarianceExplained']])
    # Select the first five components and add them to the list
    csflist = []
    wmlist = []
    for i in range(0, 4):
        try:
            csflist.append(CSF[i][0])
        except Exception as exc:
            pass
            print(exc)
        try:
            wmlist.append(WM[i][0])
        except Exception as exc:
            pass
            print(exc)
    acompcor = wmlist + csflist
    return confounds_df[acompcor]


def derivative(confound):
    data = confound.to_numpy()
    # Prepend 0 to the differences of the confound data
    return pd.DataFrame(np.diff(data, prepend=0))


def square_confound(confound):
    return confound**2  # Square the confound data


def load_confound_matrix(datafile,
                         original_file,
                         custom_confounds=None,
                         confound_tsv=None,
                         params='27P'):
    """ extract confound """
    '''
    original_file:
       file used to find confounds json
    datafile:
        boldfile whose confounds we want
    confound_tsv:
        confounds tsv
    custom_confounds:
        custom confounds tsv if there is one
    params:
       confound requested based on Ciric et. al 2017, default is '27P'
    '''

    #  Get the confounds dat from the json and tsv
    confoundjson = load_confound(original_file)[1]
    confoundtsv = pd.read_table(confound_tsv)

    if params == '24P':  # Get rot and trans values, as well as derivatives and square
        rot_values = confoundtsv[["rot_x", "rot_y", "rot_z"]]
        trans_values = confoundtsv[["trans_x", "trans_y", "trans_z"]]
        motion = pd.concat([rot_values, trans_values], axis=1)
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        confound = pd.concat([derivative_rot_trans, square_confound(derivative_rot_trans)], axis=1)
    elif params == '27P':  # Get rot and trans values, as well as derivatives, WM, CSF
        # global signal and square
        rot_values = confoundtsv[["rot_x", "rot_y", "rot_z"]]
        trans_values = confoundtsv[["trans_x", "trans_y", "trans_z"]]
        motion = pd.concat([rot_values, trans_values], axis=1)
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        whitematter_csf = load_WM_CSF(confoundtsv)
        global_signal = load_global_signal(confoundtsv)
        confound = pd.concat([derivative_rot_trans, square_confound(
            derivative_rot_trans), whitematter_csf, global_signal], axis=1)
    elif params == '36P':  # Get rot and trans values, as well as derivatives, WM, CSF,
        # global signal, and square. Add the square and derivative of the WM, CSF
        # and global signal as well.
        rot_values = confoundtsv[["rot_x", "rot_y", "rot_z"]]
        trans_values = confoundtsv[["trans_x", "trans_y", "trans_z"]]
        motion = pd.concat([rot_values, trans_values], axis=1)
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        square_confounds = pd.concat(
            [derivative_rot_trans, square_confound(derivative_rot_trans)], axis=1)
        global_signal_whitematter_csf = pd.concat(
            [load_WM_CSF(confoundtsv),
             load_global_signal(confoundtsv)], axis=1)
        global_signal_whitematter_csf_derivative = pd.concat(
            [global_signal_whitematter_csf, derivative(global_signal_whitematter_csf)], axis=1)
        confound = pd.concat([square_confounds, global_signal_whitematter_csf_derivative,
                              square_confound(global_signal_whitematter_csf_derivative)], axis=1)
    elif params == 'acompcor':  # Get the rot and trans values, their derivative,
        # as well as acompcor and cosine
        rot_values = confoundtsv[["rot_x", "rot_y", "rot_z"]]
        trans_values = confoundtsv[["trans_x", "trans_y", "trans_z"]]
        motion = pd.concat([rot_values, trans_values], axis=1)
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        acompcor = load_acompcor(confounds_df=confoundtsv,
                                 confoundjs=confoundjson)
        cosine = load_cosine(confoundtsv)
        confound = pd.concat([derivative_rot_trans, acompcor, cosine], axis=1)
    elif params == 'aroma':  # Get the WM, CSF, and aroma values
        whitematter_csf = load_WM_CSF(confoundtsv)
        aroma = load_aroma(datafile=datafile)
        confound = pd.concat([whitematter_csf, aroma], axis=1)
    elif params == 'aroma_gsr':  # Get the WM, CSF, and aroma values, as well as global signal
        whitematter_csf = load_WM_CSF(confoundtsv)
        aroma = load_aroma(datafile=datafile)
        global_signal = load_global_signal(confoundtsv)
        confound = pd.concat([whitematter_csf, aroma, global_signal], axis=1)
    elif params == 'acompcor_gsr':  # Get the rot and trans values, as well as their derivative,
        # acompcor and cosine values as well as global signal
        rot_values = confoundtsv[["rot_x", "rot_y", "rot_z"]]
        trans_values = confoundtsv[["trans_x", "trans_y", "trans_z"]]
        motion = pd.concat([rot_values, trans_values], axis=1)
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        acompcor = load_acompcor(confounds_df=confoundtsv,
                                 confoundjs=confoundjson)
        global_signal = load_global_signal(confoundtsv)
        cosine = load_cosine(confoundtsv)
        confound = pd.concat([derivative_rot_trans, acompcor, global_signal, cosine], axis=1)
    elif params == 'custom':
        # For custom confounds with no other confounds
        confound = pd.read_table(custom_confounds, sep='\t', header=None)
    if params != 'custom':  # For both custom and fMRIPrep confounds
        if custom_confounds is not None:
            custom = pd.read_table(custom_confounds, sep='\t', header=None)
            confound = pd.concat([confound, custom], axis=1)

    return confound


def load_aroma(datafile):
    """ extract aroma confounds"""
    #  Pull out aroma and melodic_ts files
    if 'space' in os.path.basename(datafile):
        aroma_noise = datafile.replace("_space-" + datafile.split("space-")[1],
                                       "_AROMAnoiseICs.csv")
        melodic_ts = datafile.replace("_space-" + datafile.split("space-")[1],
                                      "_desc-MELODIC_mixing.tsv")
    else:
        aroma_noise = datafile.split(
            '_desc-preproc_bold.nii.gz')[0] + "_AROMAnoiseICs.csv"
        melodic_ts = datafile.split(
            '_desc-preproc_bold.nii.gz')[0] + "_desc-MELODIC_mixing.tsv"
    # Load data
    aroma_noise = np.genfromtxt(
        aroma_noise,
        delimiter=',',
    )
    aroma_noise = [np.int(i) - 1
                   for i in aroma_noise]  # change to 0-based index

    # Load in meloditc_ts
    melodic = pd.read_csv(melodic_ts,
                          header=None,
                          delimiter="\t",
                          encoding="utf-8")

    # Drop aroma_noise from melodic_ts
    aroma = melodic.drop(aroma_noise, axis=1)

    return aroma


def motion_regression_filter(data,
                             TR,
                             motion_filter_type,
                             freqband,
                             cutoff=.1,
                             motion_filter_order=4):
    """
    Apply motion filter to trans and rot values.
    """
    # TODO: NEEDS TO BE REFACTORED AFTER CHECKING IN
    LP_freq_min = cutoff
    fc_RR_min, fc_RR_max = freqband

    TR = float(TR)
    order = float(motion_filter_order)
    LP_freq_min = float(LP_freq_min)
    fc_RR_min = float(fc_RR_min)
    fc_RR_max = float(fc_RR_max)
    if motion_filter_type:
        if motion_filter_type == 'lp':
            hr_min = LP_freq_min
            hr = hr_min
            fs = 1. / TR
            fNy = fs / 2.
            fa = np.abs(hr - (np.floor((hr + fNy) / fs)) * fs)
            # cutting frequency normalized between 0 and nyquist
            Wn = np.amin(fa) / fNy
            b_filt = firwin(int(order) + 1, Wn, pass_zero='lowpass')
            a_filt = 1.
            num_f_apply = 1.
        else:
            if motion_filter_type == 'notch':
                fc_RR_bw = np.array([fc_RR_min, fc_RR_max])
                rr = fc_RR_bw
                fs = 1. / TR
                fNy = fs / 2.
                fa = np.abs(rr - (np.floor((rr + fNy) / fs)) * fs)
                W_notch = fa / fNy
                Wn = np.mean(W_notch)
                Wd = np.diff(W_notch)
                bw = np.abs(Wd)
                b_filt, a_filt = iirnotch(Wn, Wn / bw)
                num_f_apply = np.int(np.floor(order / 2))
        for j in range(num_f_apply):
            for k in range(data.shape[0]):
                data[k, :] = filtfilt(b_filt, a_filt, data[k, :])
    else:
        data = data
    return data
