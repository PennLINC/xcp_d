# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Confound matrix selection based on Ciric et al. 2007."""
import os
import warnings

import numpy as np
import pandas as pd
from nipype import logging
from scipy.signal import filtfilt, firwin, iirnotch

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("utils")


def get_confounds_tsv(datafile):
    """Find path to confounds TSV file.

    Parameters
    ----------
    datafile : str
        Real nifti or cifti file.

    Returns
    -------
    confounds_timeseries : str
        Associated confounds TSV file.
    """
    if "space" in os.path.basename(datafile):
        confounds_timeseries = datafile.replace(
            "_space-" + datafile.split("space-")[1], "_desc-confounds_timeseries.tsv"
        )
    else:
        confounds_timeseries = (
            datafile.split("_desc-preproc_bold.nii.gz")[0]
            + "_desc-confounds_timeseries.tsv"
        )

    return confounds_timeseries


def load_confound(datafile):
    """Load confound amd json.

    Parameters
    ----------
    datafile : str
        Real nifti or cifti file.

    Returns
    -------
    confoundpd : pandas.DataFrame
        Loaded confounds TSV file.
    confoundjs : dict
        Metadata from associated confounds JSON file.
    """
    if "space" in os.path.basename(datafile):
        confounds_timeseries = datafile.replace(
            "_space-" + datafile.split("space-")[1], "_desc-confounds_timeseries.tsv"
        )
        confounds_json = datafile.replace(
            "_space-" + datafile.split("space-")[1], "_desc-confounds_timeseries.json"
        )
    else:
        confounds_timeseries = (
            datafile.split("_desc-preproc_bold.nii.gz")[0]
            + "_desc-confounds_timeseries.tsv"
        )
        confounds_json = (
            datafile.split("_desc-preproc_bold.nii.gz")[0]
            + "_desc-confounds_timeseries.json"
        )

    confoundpd = pd.read_csv(confounds_timeseries, delimiter="\t", encoding="utf-8")

    confoundjs = readjson(confounds_json)

    return confoundpd, confoundjs


def readjson(jsonfile):
    """Load JSON file into a dictionary.

    Parameters
    ----------
    jsonfile : str
        JSON file to load.

    Returns
    -------
    data : dict
        Data loaded from the JSON file.
    """
    import json

    with open(jsonfile) as f:
        data = json.load(f)
    return data


@fill_doc
def load_motion(
    confounds_df,
    TR,
    motion_filter_type=None,
    band_stop_min=None,
    band_stop_max=None,
    motion_filter_order=4,
):
    """Load the six basic motion regressors (three rotations, three translations).

    Parameters
    ----------
    confounds_df : pandas.DataFrame
        The confounds DataFrame from which to extract the six basic motion regressors.
    TR : float
        The repetition time of the associated scan.
    motion_filter_type : {"lp", "notch", None}
        The filter type to use.
        If "lp" or "notch", that filtering will be done in this function.
        Otherwise, no filtering will be applied.
    %(band_stop_min)s
    %(band_stop_max)s
    motion_filter_order : int, optional
        This only has an impact if ``motion_filter_type`` is "lp" or "notch".
        Default is 4.

    Returns
    -------
    motion_confounds : pandas.DataFrame
        The six motion regressors.
        The three rotations are listed first, then the three translations.

    References
    ----------
    .. footbibliography::
    """
    assert motion_filter_type in ("lp", "notch", None), motion_filter_type

    # Select the motion columns from the overall confounds DataFrame
    motion_confounds_df = confounds_df[
        ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    ]

    # Apply LP or notch filter
    if motion_filter_type in ("lp", "notch"):
        # TODO: Eliminate need for transpose. We control the filter function,
        # so we can make it work on RxT data instead of TxR.
        motion_confounds = motion_confounds_df.to_numpy().T
        motion_confounds = motion_regression_filter(
            data=motion_confounds,
            TR=TR,
            motion_filter_type=motion_filter_type,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_order=motion_filter_order,
        )
        motion_confounds = motion_confounds.T  # Transpose motion confounds back to RxT
        motion_confounds_df = pd.DataFrame(
            data=motion_confounds,
            columns=motion_confounds_df.columns,
        )

    return motion_confounds_df


def load_global_signal(confounds_df):
    """Select global signal from confounds DataFrame.

    Parameters
    ----------
    confounds_df : pandas.DataFrame
        The confounds DataFrame from which to extract information.

    Returns
    -------
    pandas.Series
        The global signal from the confounds.
    """
    df = pd.DataFrame(confounds_df["global_signal"])
    df.columns = ['GlobalSignal']
    return df


def load_wm_csf(confounds_df):
    """Select white matter and CSF nuisance regressors from confounds DataFrame.

    Parameters
    ----------
    confounds_df : pandas.DataFrame
        The confounds DataFrame from which to extract information.

    Returns
    -------
    pandas.DataFrame
        The CSF and WM signals from the confounds.
    """
    columns = ["CSF", "WhiteMatter"]
    df = confounds_df[["csf", "white_matter"]]
    df.columns = columns
    return df


def load_cosine(confounds_df):
    """Select discrete cosine-basis regressors for CompCor.

    Parameters
    ----------
    confounds_df : pandas.DataFrame
        The confounds DataFrame from which to extract information.

    Returns
    -------
    pandas.DataFrame
        The cosine-basis regressors from the confounds.

    Notes
    -----
    fMRIPrep does high-pass filtering before running anatomical or temporal CompCor.
    Therefore, when using CompCor regressors, the corresponding cosine_XX regressors
    should also be included in the design matrix.
    """
    cosine = []
    for key in confounds_df.keys():  # Any colums with cosine
        if "cosine" in key:
            cosine.append(key)
    return confounds_df[cosine]


def load_acompcor(confounds_df, confoundjs):
    """Select WM and GM acompcor separately.

    Parameters
    ----------
    confounds_df : pandas.DataFrame
        The confounds DataFrame from which to select the aCompCor regressors.
    confoundjs : dict
        The metadata associated with the confounds file.

    Returns
    -------
    pandas.DataFrame
        The confounds DataFrame, reduced to only include aCompCor regressors.
    """
    wm_comp_cor_retained = []
    csf_comp_cor_retained = []
    for key, value in confoundjs.items():  # Use the confounds json
        if "comp_cor" in key and "t" not in key:
            # Pull out variance explained for white matter masks that are retained
            if value["Mask"] == "WM" and value["Retained"]:
                wm_comp_cor_retained.append(key)
            # Pull out variance explained for CSF masks that are retained
            if value["Mask"] == "CSF" and value["Retained"]:
                csf_comp_cor_retained.append(key)

    # grab up to 5 acompcor values
    N_COLS_TO_GRAB = 5

    # Note that column names were changed from a_comp_cor to w_comp_cor and c_comp_cor
    # in later versions of fMRIPrep
    n_wm_comp_cor = N_COLS_TO_GRAB
    if len(wm_comp_cor_retained) < N_COLS_TO_GRAB:
        LOGGER.warning(f"Only {len(wm_comp_cor_retained)} white matter CompCor columns found.")
        n_wm_comp_cor = len(wm_comp_cor_retained)

    # ditto for csf
    n_csf_comp_cor = N_COLS_TO_GRAB
    if len(csf_comp_cor_retained) < N_COLS_TO_GRAB:
        LOGGER.warning(f"Only {len(wm_comp_cor_retained)} CSF CompCor columns found.")
        n_wm_comp_cor = len(csf_comp_cor_retained)

    acompcor_columns = (
        wm_comp_cor_retained[:n_wm_comp_cor] + csf_comp_cor_retained[:n_csf_comp_cor]
    )

    return confounds_df[acompcor_columns]


def derivative(confound):
    """Calculate derivative of a given array.

    Parameters
    ----------
    confound : pandas.DataFrame
        The confound to be modified.

    Returns
    -------
    pandas.DataFrame
        Derivative of the array, with a zero at the beginning.
        The column(s) will be untitled.
    """
    columns = confound.columns.tolist()
    new_columns = [c + "_dt" for c in columns]
    data = confound.to_numpy()
    # Prepend 0 to the differences of the confound data
    return pd.DataFrame(data=np.diff(data, prepend=0), columns=new_columns)


def square_confound(confound):
    """Square an array.

    Parameters
    ----------
    confound : array_like or int or float
        An array or value to square.

    Returns
    -------
    array_like or int or float
        The squared input data.
    """
    columns = confound.columns.tolist()
    new_columns = [c + "_sq" for c in columns]
    squared_confounds = (confound**2)
    squared_confounds.columns = new_columns
    return squared_confounds  # Square the confound data


@fill_doc
def load_confound_matrix(
    original_file, params, custom_confounds=None, confound_tsv=None
):
    """Load a subset of the confounds associated with a given file.

    Parameters
    ----------
    original_file :
       File used to find confounds json.
    %(params)s
    custom_confounds : str or None, optional
        Custom confounds TSV if there is one. Default is None.
    confound_tsv : str or None, optional
        The path to the confounds TSV file. Default is None.

    Returns
    -------
    confound : pandas.DataFrame
        The loaded and selected confounds.
    """
    #  Get the confounds dat from the json and tsv
    confounds_metadata = load_confound(original_file)[1]
    confounds_df = pd.read_table(confound_tsv)

    if params == "24P":  # Get rot and trans values, as well as derivatives and square
        motion = confounds_df[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        confound = pd.concat(
            [derivative_rot_trans, square_confound(derivative_rot_trans)], axis=1
        )
    elif params == "27P":  # Get rot and trans values, as well as derivatives, WM, CSF
        # global signal and square
        motion = confounds_df[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        whitematter_csf = load_wm_csf(confounds_df)
        global_signal = load_global_signal(confounds_df)
        confound = pd.concat(
            [
                derivative_rot_trans,
                square_confound(derivative_rot_trans),
                whitematter_csf,
                global_signal,
            ],
            axis=1,
        )
    elif params == "36P":  # Get rot and trans values, as well as derivatives, WM, CSF,
        # global signal, and square. Add the square and derivative of the WM, CSF
        # and global signal as well.
        motion = confounds_df[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        square_confounds = pd.concat(
            [derivative_rot_trans, square_confound(derivative_rot_trans)], axis=1
        )
        global_signal_whitematter_csf = pd.concat(
            [load_wm_csf(confounds_df), load_global_signal(confounds_df)], axis=1
        )
        global_signal_whitematter_csf_derivative = pd.concat(
            [global_signal_whitematter_csf, derivative(global_signal_whitematter_csf)],
            axis=1,
        )
        confound = pd.concat(
            [
                square_confounds,
                global_signal_whitematter_csf_derivative,
                square_confound(global_signal_whitematter_csf_derivative),
            ],
            axis=1,
        )
    elif params == "acompcor":  # Get the rot and trans values, their derivative,
        # as well as acompcor and cosine
        motion = confounds_df[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        acompcor = load_acompcor(confounds_df=confounds_df, confoundjs=confounds_metadata)
        cosine = load_cosine(confounds_df)
        confound = pd.concat([derivative_rot_trans, acompcor, cosine], axis=1)
    elif params == "aroma":  # Get the WM, CSF, and aroma values
        whitematter_csf = load_wm_csf(confounds_df)
        aroma = load_aroma(datafile=original_file)
        confound = pd.concat([whitematter_csf, aroma], axis=1)
    elif (
        params == "aroma_gsr"
    ):  # Get the WM, CSF, and aroma values, as well as global signal
        whitematter_csf = load_wm_csf(confounds_df)
        aroma = load_aroma(datafile=original_file)
        global_signal = load_global_signal(confounds_df)
        confound = pd.concat([whitematter_csf, aroma, global_signal], axis=1)
    elif (
        params == "acompcor_gsr"
    ):  # Get the rot and trans values, as well as their derivative,
        # acompcor and cosine values as well as global signal
        motion = confounds_df[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]]
        derivative_rot_trans = pd.concat([motion, derivative(motion)], axis=1)
        acompcor = load_acompcor(confounds_df=confounds_df, confoundjs=confounds_metadata)
        global_signal = load_global_signal(confounds_df)
        cosine = load_cosine(confounds_df)
        confound = pd.concat(
            [derivative_rot_trans, acompcor, global_signal, cosine], axis=1
        )
    elif params == "custom":
        # For custom confounds with no other confounds
        confound = pd.read_table(custom_confounds, sep="\t", header=None)
    if params != "custom":  # For both custom and fMRIPrep confounds
        if custom_confounds is not None:
            custom = pd.read_table(custom_confounds, sep="\t", header=None)
            confound = pd.concat([confound, custom], axis=1)

    return confound


def load_aroma(datafile):
    """Extract aroma confounds from a confounds TSV file.

    Parameters
    ----------
    datafile : str
        Path to the preprocessed BOLD file for which to extract AROMA confounds.

    Returns
    -------
    aroma : pandas.DataFrame
        The AROMA noise components.
    """
    #  Pull out aroma and melodic_ts files
    if "space" in os.path.basename(datafile):
        aroma_noise = datafile.replace(
            "_space-" + datafile.split("space-")[1], "_AROMAnoiseICs.csv"
        )
        melodic_ts = datafile.replace(
            "_space-" + datafile.split("space-")[1], "_desc-MELODIC_mixing.tsv"
        )
    else:
        aroma_noise = (
            datafile.split("_desc-preproc_bold.nii.gz")[0] + "_AROMAnoiseICs.csv"
        )
        melodic_ts = (
            datafile.split("_desc-preproc_bold.nii.gz")[0] + "_desc-MELODIC_mixing.tsv"
        )
    # Load data
    aroma_noise = np.genfromtxt(
        aroma_noise,
        delimiter=",",
    )
    aroma_noise = [np.int(i) - 1 for i in aroma_noise]  # change to 0-based index

    # Load in meloditc_ts
    melodic = pd.read_csv(melodic_ts, header=None, delimiter="\t", encoding="utf-8")

    # Drop aroma_noise from melodic_ts
    aroma = melodic.drop(aroma_noise, axis=1)

    return aroma


@fill_doc
def motion_regression_filter(
    data,
    TR,
    motion_filter_type,
    band_stop_min,
    band_stop_max,
    motion_filter_order=4,
):
    """Filter translation and rotation motion parameters.

    Parameters
    ----------
    data : numpy.ndarray
        Data to filter.
    TR : float
        Repetition time of the data.
    motion_filter_type : {"lp", "notch"}
        The type of motion filter to apply.
        If "notch", the frequencies between ``band_stop_min`` and ``band_stop_max`` will be
        removed.
        If "lp", the frequencies above ``band_stop_min`` will be removed.
        If not "notch" or "lp", an exception will be raised.
    %(band_stop_min)s
    %(band_stop_max)s
    motion_filter_order : int, optional
        Default is 4.

    Returns
    -------
    data : numpy.ndarray
        Filtered data.

    References
    ----------
    .. footbibliography::
    """
    assert motion_filter_type in ("lp", "notch")

    # casting all variables
    TR = float(TR)
    order = float(motion_filter_order)

    filtered_data = data.copy()

    sampling_frequency = 1.0 / TR
    nyquist_frequency = sampling_frequency / 2.0

    if motion_filter_type == "lp":  # low-pass filter
        # Remove any frequencies above band_stop_min.
        assert band_stop_min is not None
        assert band_stop_min > 0
        if band_stop_max:
            warnings.warn("The parameter 'band_stop_max' will be ignored.")

        low_pass_freq_hertz = band_stop_min / 60  # change BPM to right time unit

        # cutting frequency
        cutting_frequency = np.abs(
            low_pass_freq_hertz
            - (np.floor((low_pass_freq_hertz + nyquist_frequency) / sampling_frequency))
            * sampling_frequency
        )
        # cutting frequency normalized between 0 and nyquist
        Wn = np.amin(cutting_frequency) / nyquist_frequency  # cutoffs
        filt_num = firwin(int(order) + 1, Wn, pass_zero="lowpass")  # create b_filt
        filt_denom = 1.0
        num_f_apply = 1  # num of times to apply

    elif motion_filter_type == "notch":  # notch filter
        # Retain any frequencies *outside* the band_stop_min-band_stop_max range.
        assert band_stop_max is not None
        assert band_stop_min is not None
        assert band_stop_max > 0
        assert band_stop_min > 0
        assert band_stop_min < band_stop_max

        # bandwidth as an array
        bandstop_band = np.array([band_stop_min, band_stop_max])
        bandstop_band_hz = bandstop_band / 60  # change BPM to Hertz
        cutting_frequencies = np.abs(
            bandstop_band_hz
            - (np.floor((bandstop_band_hz + nyquist_frequency) / sampling_frequency))
            * sampling_frequency
        )

        # normalize cutting frequency
        W_notch = cutting_frequencies / nyquist_frequency
        Wn = np.mean(W_notch)
        Wd = np.diff(W_notch)
        bandwidth = np.abs(Wd)  # bandwidth
        # create filter coefficients
        filt_num, filt_denom = iirnotch(Wn, Wn / bandwidth)
        num_f_apply = np.int(np.floor(order / 2))  # how many times to apply filter

    for i_iter in range(num_f_apply):
        for j_row in range(data.shape[0]):  # apply filters across columns
            filtered_data[j_row, :] = filtfilt(filt_num, filt_denom, filtered_data[j_row, :])

    return filtered_data
