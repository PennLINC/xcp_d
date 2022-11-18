# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Confound matrix selection based on Ciric et al. 2007."""
import os
import warnings

import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds
from nipype import logging
from scipy.signal import filtfilt, firwin, iirnotch

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


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
            datafile.split("_desc-preproc_bold.nii.gz")[0] + "_desc-confounds_timeseries.tsv"
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
    confounds_df : pandas.DataFrame
        Loaded confounds TSV file.
    confounds_metadata : dict
        Metadata from associated confounds JSON file.
    """
    confounds_tsv = get_confounds_tsv(datafile)
    confound_file_base, _ = os.path.splitext(confounds_tsv)
    confounds_json = confound_file_base + ".json"
    if not os.path.isfile(confounds_json):
        raise FileNotFoundError(
            "No json found for confounds tsv.\n"
            f"\tTSV file: {confounds_tsv}\n"
            f"\tJSON file (DNE): {confounds_json}"
        )

    confounds_df = pd.read_table(confounds_tsv)
    confounds_metadata = readjson(confounds_json)

    return confounds_df, confounds_metadata


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


@fill_doc
def load_confound_matrix(params, img_files, custom_confounds=None):
    """
    Load a subset of the confounds associated with a given file.

    Parameters
    ----------
    %(params)s
    img_files : str
        The path to the bold file.
    custom_confounds : str or None, optional
        Custom confounds TSV if there is one. Default is None.

    Returns
    -------
    confound : pandas.DataFrame
        The loaded and selected confounds.

    Notes
    -------
    Switching the order of the trans and rot values in the motion columns
    can cause regression to happen incorrectly.
    """
    if params == "24P":  # Get rot and trans values, as well as derivatives and square
        confound = load_confounds(img_files, strategy=(['motion']), motion='full',
                                  scrub=100, fd_threshold=100, std_dvars_threshold=100)[0]
    elif params == "27P":  # Get rot and trans values, as well as derivatives and square, WM, CSF,
        # global signal
        confound = load_confounds(img_files, strategy=(['motion', 'wm_csf', 'global_signal']),
                                  scrub=100, fd_threshold=100, std_dvars_threshold=100,
                                  motion='full')[0]
    elif params == "36P":  # Get rot and trans values, as well as derivatives, WM, CSF,
        # global signal, and square. Add the square and derivative of the WM, CSF
        # and global signal as well.
        confound = load_confounds(img_files, strategy=(['motion', 'wm_csf', 'global_signal']),
                                  scrub=100, fd_threshold=100, std_dvars_threshold=100,
                                  motion='full', global_signal='full', wm_csf='full')[0]
    elif params == "acompcor":  # Get the rot and trans values, their derivative,
        # as well as acompcor and cosine
        confound = load_confounds(img_files, strategy=(['motion', 'high_pass', 'compcor', ]),
                                  motion='derivatives', compcor='anat_separated',
                                  scrub=100, fd_threshold=100, n_compcor=5,
                                  std_dvars_threshold=100)[0]
    elif params == "aroma":  # Get the WM, CSF, and aroma values
        confound = load_confounds(img_files, strategy=(['wm_csf', 'ica_aroma']),
                                  scrub=100, fd_threshold=100, std_dvars_threshold=100,
                                  wm_csf='basic', ica_aroma='full')[0]
    elif params == "aroma_gsr":  # Get the WM, CSF, and aroma values, as well as global signal
        confound = load_confounds(img_files, strategy=(['wm_csf', 'ica_aroma', 'global_signal']),
                                  scrub=100, fd_threshold=100, std_dvars_threshold=100,
                                  wm_csf='basic', global_signal='basic', ica_aroma='full')[0]
    elif params == "acompcor_gsr":  # Get the rot and trans values, as well as their derivative,
        # acompcor and cosine values as well as global signal
        confound = load_confounds(img_files, strategy=(['motion', 'high_pass', 'compcor',
                                                        'global_signal']),
                                  motion='derivatives', compcor='anat_separated',
                                  scrub=100, fd_threshold=100, global_signal='basic', n_compcor=5,
                                  std_dvars_threshold=100)[0]
    elif params == "custom":
        # For custom confounds with no other confounds
        confound = pd.read_table(custom_confounds, sep="\t")

    if params != "custom" and custom_confounds is not None:
        # For both custom and fMRIPrep confounds
        custom = pd.read_table(custom_confounds, sep="\t")
        confound = pd.concat([custom, confound], axis=1)

    return confound


def load_aroma(confounds_df):
    """Extract aroma confounds from a confounds TSV file.

    Parameters
    ----------
    confounds_df : :obj:`pandas.DataFrame`
        The confounds DataFrame.

    Returns
    -------
    pandas.DataFrame
        The AROMA noise components.
    """
    aroma_noise = [c for c in confounds_df.columns if c.startswith("aroma_motion_")]
    aroma = confounds_df[aroma_noise]

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


def _infer_dummy_scans(dummy_scans, confounds_file=None):
    """Attempt to determine the number of dummy scans from the confounds file.

    This function expects non-steady-state volumes flagged by the preprocessing pipeline
    to be indicated in columns starting with "non_steady_state_outlier" in the confounds file.

    Parameters
    ----------
    dummy_scans : "auto" or int
        The number of dummy scans.
        If an integer, this will be returned without modification.
        If "auto", the confounds file will be loaded and the number of non-steady-state
        volumes estimated by the preprocessing workflow will be determined.
    confounds_file : None or str
        Path to the confounds TSV file from the preprocessing pipeline.
        Only used if dummy_scans is "auto".
        Default is None.

    Returns
    -------
    dummy_scans : int
        Estimated number of dummy scans.
    """
    if dummy_scans == "auto":
        confounds_df = pd.read_table(confounds_file)

        nss_cols = [c for c in confounds_df.columns if c.startswith("non_steady_state_outlier")]

        if nss_cols:
            initial_volumes_df = confounds_df[nss_cols]
            dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
            dummy_scans = np.where(dummy_scans)[0]

            # reasonably assumes all NSS volumes are contiguous
            dummy_scans = int(dummy_scans[-1] + 1)
            LOGGER.info(f"Found {dummy_scans} dummy scans in {os.path.basename(confounds_file)}")

        else:
            LOGGER.warning(f"No non-steady-state outliers found in {confounds_file}")
            dummy_scans = 0

    return dummy_scans
