# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Confound matrix selection based on Ciric et al. 2007."""
import json
import os
import warnings

import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep.load_confounds import _load_single_confounds_file
from nipype import logging
from scipy import ndimage
from scipy.signal import butter, filtfilt, iirnotch

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


@fill_doc
def filter_motion(
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
    %(TR)s
    %(motion_filter_type)s
        If "lp" or "notch", that filtering will be done in this function.
        Otherwise, no filtering will be applied.
    %(band_stop_min)s
    %(band_stop_max)s
    %(motion_filter_order)s

    Returns
    -------
    confounds_df : pandas.DataFrame
        The six motion regressors.
        The three rotations are listed first, then the three translations.

    References
    ----------
    .. footbibliography::
    """
    if motion_filter_type not in ("lp", "notch", None):
        raise ValueError(f"Motion filter type '{motion_filter_type}' not supported.")

    confounds_df = confounds_df.copy()
    motion_columns = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]

    # Apply LP or notch filter
    if motion_filter_type in ("lp", "notch"):
        confounds_df[motion_columns] = motion_regression_filter(
            data=confounds_df[motion_columns].to_numpy(),
            TR=TR,
            motion_filter_type=motion_filter_type,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_order=motion_filter_order,
        )

    # Volterra expansion
    # Ignore pandas SettingWithCopyWarning
    with pd.option_context("mode.chained_assignment", None):
        columns_to_square = motion_columns[:]
        for col in motion_columns:
            deriv_col = f"{col}_derivative1"
            confounds_df[deriv_col] = confounds_df[col].diff()
            columns_to_square.append(deriv_col)

        for col in columns_to_square:
            square_col = f"{col}_power2"
            confounds_df[square_col] = confounds_df[col] ** 2

    return confounds_df


@fill_doc
def get_custom_confounds(custom_confounds_folder, full_confounds):
    """Identify a custom confounds file.

    Parameters
    ----------
    %(custom_confounds_folder)s
    %(full_confounds)s
        We expect the custom confounds file to have the same name.

    Returns
    -------
    %(custom_confounds_file)s
    """
    import os

    if custom_confounds_folder is None:
        return None

    if not os.path.isdir(custom_confounds_folder):
        raise FileNotFoundError(
            f"Custom confounds location does not exist: {custom_confounds_folder}"
        )

    custom_confounds_filename = os.path.basename(full_confounds)
    custom_confounds_file = os.path.abspath(
        os.path.join(
            custom_confounds_folder,
            custom_confounds_filename,
        )
    )

    if not os.path.isfile(custom_confounds_file):
        raise FileNotFoundError(f"Custom confounds file not found: {custom_confounds_file}")

    return custom_confounds_file


def _get_acompcor_confounds(confounds_file):
    confounds_df = pd.read_table(confounds_file)
    csf_compcor_columns = [c for c in confounds_df.columns if c.startswith("c_comp_cor")]
    wm_compcor_columns = [c for c in confounds_df.columns if c.startswith("w_comp_cor")]
    if not csf_compcor_columns:
        raise ValueError(f"No c_comp_cor columns in {confounds_file}")

    if not wm_compcor_columns:
        raise ValueError(f"No w_comp_cor columns in {confounds_file}")

    csf_compcor_columns = csf_compcor_columns[: min((5, len(csf_compcor_columns)))]
    wm_compcor_columns = wm_compcor_columns[: min((5, len(wm_compcor_columns)))]
    selected_columns = csf_compcor_columns + wm_compcor_columns
    return confounds_df[selected_columns]


@fill_doc
def load_confound_matrix(
    params,
    img_file,
    confounds_file,
    confounds_json_file,
    custom_confounds=None,
):
    """Load a subset of the confounds associated with a given file.

    Parameters
    ----------
    %(params)s
    img_file : :obj:`str`
        The path to the bold file. Used to load the AROMA mixing matrix, if necessary.
    confounds_file : :obj:`str`
        The fMRIPrep confounds file. Used to load most confounds.
    confounds_json_file : :obj:`str`
        The JSON file associated with the fMRIPrep confounds file.
    custom_confounds : :obj:`str` or None, optional
        Custom confounds TSV if there is one. Default is None.

    Returns
    -------
    confounds_df : :obj:`pandas.DataFrame` or None
        The loaded and selected confounds.
        If "AROMA" is requested, then this DataFrame will include signal components as well.
        These will be named something like "signal_[XX]".
        If ``params`` is "none", ``confounds_df`` will be None.
    confounds_metadata : :obj:`dict`
        Metadata for the columns in the confounds file.
    """
    PARAM_KWARGS = {
        # Get rot and trans values, as well as derivatives and square
        "24P": {
            "strategy": ["motion"],
            "motion": "full",
        },
        # Get rot and trans values, as well as derivatives and square, WM, CSF,
        "27P": {
            "strategy": ["motion", "global_signal", "wm_csf"],
            "motion": "full",
            "global_signal": "basic",
            "wm_csf": "basic",
        },
        # Get rot and trans values, as well as derivatives, WM, CSF,
        # global signal, and square. Add the square and derivative of the WM, CSF
        # and global signal as well.
        "36P": {
            "strategy": ["motion", "global_signal", "wm_csf"],
            "motion": "full",
            "global_signal": "full",
            "wm_csf": "full",
        },
        # Get the rot and trans values, their derivative,
        # as well as acompcor and cosine
        "acompcor": {
            "strategy": ["motion", "high_pass", "compcor"],
            "motion": "derivatives",
            "compcor": "anat_separated",
            "n_compcor": 5,
        },
        # Get the rot and trans values, as well as their derivative,
        # acompcor and cosine values as well as global signal
        "acompcor_gsr": {
            "strategy": ["motion", "high_pass", "compcor", "global_signal"],
            "motion": "derivatives",
            "compcor": "anat_separated",
            "global_signal": "basic",
            "n_compcor": 5,
        },
        # Get WM and CSF
        # AROMA confounds are loaded separately
        "aroma": {
            "strategy": ["wm_csf"],
            "wm_csf": "basic",
        },
        # Get WM, CSF, and global signal
        # AROMA confounds are loaded separately
        "aroma_gsr": {
            "strategy": ["wm_csf", "global_signal"],
            "wm_csf": "basic",
            "global_signal": "basic",
        },
        # Get global signal only
        "gsr_only": {
            "strategy": ["global_signal"],
            "global_signal": "basic",
        },
    }

    if params == "none":
        return None, {}

    if params in PARAM_KWARGS:
        kwargs = PARAM_KWARGS[params]

        confounds_df = _load_single_confounds_file(
            confounds_file=confounds_file,
            demean=False,
            confounds_json_file=confounds_json_file,
            **kwargs,
        )[1]

    elif params == "custom":
        # For custom confounds with no other confounds
        confounds_df = pd.read_table(custom_confounds, sep="\t")

    else:
        raise ValueError(f"Unrecognized parameter string '{params}'")

    # A workaround for the compcor bug in load_confounds with fMRIPrep v22+
    if "acompcor" in params and all("comp_cor" not in col for col in confounds_df.columns):
        LOGGER.warning("No aCompCor confounds detected with load_confounds. Extracting manually.")
        confounds_df = pd.concat((_get_acompcor_confounds(confounds_file), confounds_df), axis=1)

    if "aroma" in params:
        ica_mixing_matrix = _get_mixing_matrix(img_file)
        aroma_noise_comps_idx = _get_aroma_noise_comps(img_file)
        labeled_ica_mixing_matrix = _label_mixing_matrix(ica_mixing_matrix, aroma_noise_comps_idx)
        confounds_df = pd.concat([confounds_df, labeled_ica_mixing_matrix], axis=1)

    if params != "custom" and custom_confounds is not None:
        # For both custom and fMRIPrep confounds
        custom_confounds_df = pd.read_table(custom_confounds, sep="\t")
        confounds_df = pd.concat([custom_confounds_df, confounds_df], axis=1)

    with open(confounds_json_file, "r") as fo:
        full_confounds_metadata = json.load(fo)

    confounds_metadata = {
        k: v for k, v in full_confounds_metadata.items() if k in confounds_df.columns
    }

    return confounds_df, confounds_metadata


def _get_mixing_matrix(img_file):
    """Find AROMA (i.e., MELODIC) mixing matrix file for a given BOLD file."""
    suffix = "_space-" + img_file.split("space-")[1]

    mixing_candidates = [
        img_file.replace(suffix, "_desc-MELODIC_mixing.tsv"),
    ]

    mixing_file = [cr for cr in mixing_candidates if os.path.isfile(cr)]

    if not mixing_file:
        raise FileNotFoundError(f"Could not find mixing matrix for {img_file}")

    return mixing_file[0]


def _get_aroma_noise_comps(img_file):
    """Find AROMA noise components file for a given BOLD file."""
    suffix = "_space-" + img_file.split("space-")[1]

    index_candidates = [
        img_file.replace(suffix, "_AROMAnoiseICs.csv"),
    ]

    index_file = [cr for cr in index_candidates if os.path.isfile(cr)]

    if not index_file:
        raise FileNotFoundError(f"Could not find AROMAnoiseICs file for {img_file}")

    return index_file[0]


def _label_mixing_matrix(mixing_file, noise_index_file):
    """Prepend 'signal__' to any non-noise components in AROMA mixing matrix."""
    mixing_matrix = np.loadtxt(mixing_file, delimiter="\t")
    noise_index = np.loadtxt(noise_index_file, delimiter=",", dtype=int)
    # shift noise index to start with zero
    noise_index -= 1
    all_index = np.arange(mixing_matrix.shape[1], dtype=int)
    signal_index = np.setdiff1d(all_index, noise_index)
    noise_components = mixing_matrix[:, noise_index]
    signal_components = mixing_matrix[:, signal_index]
    # basing naming convention on fMRIPrep confounds column names
    noise_component_names = [f"aroma_motion_{i:03g}" for i in noise_index]
    signal_component_names = [f"signal__aroma_signal_{i:03g}" for i in signal_index]

    noise_component_df = pd.DataFrame(noise_components, columns=noise_component_names)
    signal_component_df = pd.DataFrame(signal_components, columns=signal_component_names)
    component_df = pd.concat((noise_component_df, signal_component_df), axis=1)
    return component_df


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
    data : (T, R) numpy.ndarray
        Data to filter. T = time, R = motion regressors
        The filter will be applied independently to each variable, across time.
    %(TR)s
    %(motion_filter_type)s
        If not "notch" or "lp", an exception will be raised.
    %(band_stop_min)s
    %(band_stop_max)s
    %(motion_filter_order)s

    Returns
    -------
    data : (T, R) numpy.ndarray
        Filtered data. Same shape as the original data.

    Notes
    -----
    Low-pass filtering (``motion_filter_type = "lp"``) is performed with a Butterworth filter,
    as in :footcite:t:`gratton2020removal`.
    The order of the Butterworth filter is determined by ``motion_filter_order``,
    although the original paper used a first-order filter.
    The original paper also used zero-padding with a padding size of 100.
    We use constant-padding, with the default padding size determined by
    :func:`scipy.signal.filtfilt`.

    Band-stop filtering (``motion_filter_type = "notch"``) is performed with a notch filter,
    as in :footcite:t:`fair2020correction`.
    This filter uses the mean of the stopband frequencies as the target frequency,
    and the range between the two frequencies as the bandwidth.
    The filter is applied with constant-padding, using the default padding size determined by
    :func:`scipy.signal.filtfilt`.

    References
    ----------
    .. footbibliography::
    """
    if motion_filter_type not in ("lp", "notch"):
        raise ValueError(f"Motion filter type '{motion_filter_type}' not supported.")

    lowpass_hz = band_stop_min / 60

    sampling_frequency = 1 / TR

    if motion_filter_type == "lp":  # low-pass filter
        b, a = butter(
            motion_filter_order / 2,
            lowpass_hz,
            btype="lowpass",
            output="ba",
            fs=sampling_frequency,
        )
        filtered_data = filtfilt(b, a, data, axis=0, padtype="constant", padlen=data.shape[0] - 1)

    else:  # notch filter
        highpass_hz = band_stop_max / 60
        stopband_hz = np.array([lowpass_hz, highpass_hz])
        # Convert stopband to a single notch frequency.
        freq_to_remove = np.mean(stopband_hz)
        bandwidth = np.abs(np.diff(stopband_hz))

        # Create filter coefficients.
        b, a = iirnotch(freq_to_remove, freq_to_remove / bandwidth, fs=sampling_frequency)
        n_filter_applications = int(np.floor(motion_filter_order / 2))

        filtered_data = data.copy()
        for _ in range(n_filter_applications):
            filtered_data = filtfilt(
                b,
                a,
                filtered_data,
                axis=0,
                padtype="constant",
                padlen=data.shape[0] - 1,
            )

    return filtered_data


def _modify_motion_filter(motion_filter_type, band_stop_min, band_stop_max, TR):
    """Modify the motion filter parameters based on the TR.

    Parameters
    ----------
    motion_filter_type : str
        The type of motion filter to apply.
    band_stop_min : float
        The minimum frequency to stop in the filter, in breaths-per-minute.
    band_stop_max : float
        The maximum frequency to stop in the filter, in breaths-per-minute.
    TR : float
        The repetition time of the data.

    Returns
    -------
    band_stop_min_adjusted : float
        The adjusted low-pass filter frequency, in breaths-per-minute.
    band_stop_max_adjusted : float
        The adjusted high-pass filter frequency, in breaths-per-minute.
    is_modified : bool
        Whether the filter parameters were modified.
    """
    sampling_frequency = 1 / TR
    nyquist_frequency = sampling_frequency / 2
    nyquist_bpm = nyquist_frequency * 60

    is_modified = False
    if motion_filter_type == "lp":  # low-pass filter
        # Remove any frequencies above band_stop_min.
        assert band_stop_min is not None
        assert band_stop_min > 0
        if band_stop_max:
            warnings.warn("The parameter 'band_stop_max' will be ignored.")

        lowpass_hz = band_stop_min / 60  # change BPM to right time unit

        # Adjust frequency in case Nyquist is below cutoff.
        # This won't have an effect if the data have a fast enough sampling rate.
        lowpass_hz_adjusted = np.abs(
            lowpass_hz
            - (np.floor((lowpass_hz + nyquist_frequency) / sampling_frequency))
            * sampling_frequency
        )
        band_stop_min_adjusted = lowpass_hz_adjusted * 60  # change Hertz back to BPM
        if band_stop_min_adjusted != band_stop_min:
            warnings.warn(
                f"Low-pass filter frequency is above Nyquist frequency ({nyquist_bpm} BPM), "
                f"so it has been changed ({band_stop_min} --> {band_stop_min_adjusted} BPM)."
            )
            is_modified = True

        band_stop_max_adjusted = None

    elif motion_filter_type == "notch":  # notch filter
        # Retain any frequencies *outside* the band_stop_min-band_stop_max range.
        assert band_stop_max is not None
        assert band_stop_min is not None
        assert band_stop_max > 0
        assert band_stop_min > 0
        assert band_stop_min < band_stop_max, f"{band_stop_min} >= {band_stop_max}"

        stopband = np.array([band_stop_min, band_stop_max])
        stopband_hz = stopband / 60  # change BPM to Hertz

        # Adjust frequencies in case Nyquist is within/below band.
        # This won't have an effect if the data have a fast enough sampling rate.
        stopband_hz_adjusted = np.abs(
            stopband_hz
            - (np.floor((stopband_hz + nyquist_frequency) / sampling_frequency))
            * sampling_frequency
        )
        stopband_adjusted = stopband_hz_adjusted * 60  # change Hertz back to BPM
        if not np.array_equal(stopband_adjusted, stopband):
            warnings.warn(
                f"One or both filter frequencies are above Nyquist frequency ({nyquist_bpm} BPM), "
                "so they have been changed "
                f"({stopband[0]} --> {stopband_adjusted[0]}, "
                f"{stopband[1]} --> {stopband_adjusted[1]} BPM)."
            )
            is_modified = True

        band_stop_min_adjusted, band_stop_max_adjusted = stopband_adjusted
    else:
        band_stop_min_adjusted, band_stop_max_adjusted, is_modified = None, None, False

    return band_stop_min_adjusted, band_stop_max_adjusted, is_modified


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


def censor_basic(array, threshold):
    """Censor basic."""
    array[np.isnan(array)] = 0
    if threshold > 0:
        outlier_mask = array > threshold
    else:
        outlier_mask = np.zeros_like(array, dtype=bool)

    return outlier_mask


def censor_around(outlier_mask, before, after, between):
    """Censor volumes adjacent to outliers.

    Parameters
    ----------
    outlier_mask : array
        Boolean array where True indicates an outlier.
    before : int
        Number of volumes to censor before each outlier.
    after : int
        Number of volumes to censor after each outlier.
    between : int
        Number of volumes to censor between each outlier.

    Returns
    -------
    outlier_mask : array
        Boolean array with additional volumes censored.
    """
    # Censoring before and after outliers
    n_vols = len(outlier_mask)
    outlier_mask = outlier_mask.astype(bool).copy()
    outliers_idx = np.where(outlier_mask)[0]  # index untouched by expansion
    for i in outliers_idx:
        outlier_mask[max(0, i - before) : i] = True
        outlier_mask[i + 1 : min(i + 1 + after, n_vols)] = True

    # Find any contiguous sequences of 0s and censor them if the length of the sequence is
    # less than or equal to the censor_between value.
    # Find all contiguous sequences of Falses
    labeled_array, n_uncensored_groups = ndimage.label(~outlier_mask)

    # Iterate over each contiguous sequence
    for i in range(1, n_uncensored_groups + 1):
        slice_ = ndimage.find_objects(labeled_array == i)[0][0]
        length = slice_.stop - slice_.start

        # If the length of the contiguous sequence is less than censor_between,
        # set the values to True
        if length <= between:
            outlier_mask[slice_] = True

    return outlier_mask


def calculate_outliers(*, confounds, fd_thresh, dvars_thresh, before, after, between):
    """Generate a temporal mask DataFrame.

    Parameters
    ----------
    confounds : pandas.DataFrame
        DataFrame with confounds.
        The two columns that will be used are "framewise_displacement" and "std_dvars".
    fd_thresh : tuple
        Tuple of two floats. The first value is the threshold for censoring during denoising,
        and the second value is the threshold for censoring during interpolation.
    dvars_thresh : tuple
        Tuple of two floats. The first value is the threshold for censoring during denoising,
        and the second value is the threshold for censoring during interpolation.
    before : tuple
        Tuple of two integers. The first value is the number of volumes to censor before each
        FD or DVARS outlier volume during denoising, and the second value is the number of volumes
        to censor before each FD or DVARS outlier volume during interpolation.
    after : tuple
        Tuple of two integers. The first value is the number of volumes to censor after each
        FD or DVARS outlier volume during denoising, and the second value is the number of volumes
        to censor after each FD or DVARS outlier volume during interpolation.
    between : tuple
        Tuple of two integers. The first value is the number of volumes to censor between each
        FD or DVARS outlier volume during denoising, and the second value is the number of volumes
        to censor between each FD or DVARS outlier volume during interpolation.

    Returns
    -------
    outliers_df : pandas.DataFrame
        DataFrame with columns for each type of censoring.
    """
    # Generate temporal mask with all timepoints have FD/DVARS over threshold set to 1.
    n_vols = confounds.shape[0]

    # Grab FD time series
    if (
        fd_thresh[0] > 0 or fd_thresh[1] > 0
    ) and "framewise_displacement" not in confounds.columns:
        raise ValueError(
            "The 'framewise_displacement' column is missing from the fMRIPrep confounds file. "
            "FD-based censoring is not possible."
        )
    elif "framewise_displacement" in confounds.columns:
        fd_timeseries = confounds["framewise_displacement"].to_numpy()
    else:
        # Create a fake array that won't actually be used
        fd_timeseries = np.zeros(n_vols)

    # Grab DVARS time series
    if (dvars_thresh[0] > 0 or dvars_thresh[1] > 0) and "std_dvars" not in confounds.columns:
        raise ValueError(
            "The 'std_dvars' column is missing from the fMRIPrep confounds file. "
            "DVARS-based censoring is not possible."
        )
    elif "std_dvars" in confounds.columns:
        dvars_timeseries = confounds["std_dvars"].to_numpy()
    else:
        # Create a fake array that won't actually be used
        dvars_timeseries = np.zeros(n_vols)

    outliers_df = pd.DataFrame(
        columns=[
            "framewise_displacement",
            "dvars",
            "denoising",
            "framewise_displacement_interpolation",
            "dvars_interpolation",
            "interpolation",
        ],
        data=np.zeros((n_vols, 6), dtype=bool),
    )

    # Censoring for denoising
    outliers_df["framewise_displacement"] = censor_basic(fd_timeseries, fd_thresh[0])
    outliers_df["dvars"] = censor_basic(dvars_timeseries, dvars_thresh[0])

    outlier_mask = np.logical_or(outliers_df["framewise_displacement"], outliers_df["dvars"])
    outliers_df["denoising"] = censor_around(
        outlier_mask,
        before=before[0],
        after=after[0],
        between=between[0],
    )

    # Censoring for interpolated data (to remove interpolated signals that might be spread to
    # other volumes by the temporal filter)
    outliers_df["framewise_displacement_interpolation"] = censor_basic(fd_timeseries, fd_thresh[1])
    outliers_df["dvars_interpolation"] = censor_basic(dvars_timeseries, dvars_thresh[1])

    outlier_mask = np.logical_or(
        outliers_df["framewise_displacement_interpolation"],
        outliers_df["dvars_interpolation"],
    )
    outliers_df["interpolation"] = censor_around(
        outlier_mask,
        before=before[1],
        after=after[1],
        between=between[1],
    )
    # Interpolation outlier index should include all outliers form denoising index,
    # because all volumes in denoising index are interpolated in denoising step.
    outliers_df["interpolation"] = np.logical_or(
        outliers_df["interpolation"],
        outliers_df["denoising"],
    )
    return outliers_df
