# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Confound matrix selection based on Ciric et al. 2007."""
import os
import warnings

import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds
from nipype import logging
from scipy.signal import butter, filtfilt, iirnotch

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


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
    %(TR)s
    %(motion_filter_type)s
        If "lp" or "notch", that filtering will be done in this function.
        Otherwise, no filtering will be applied.
    %(band_stop_min)s
    %(band_stop_max)s
    %(motion_filter_order)s

    Returns
    -------
    motion_confounds_df : pandas.DataFrame
        The six motion regressors.
        The three rotations are listed first, then the three translations.

    References
    ----------
    .. footbibliography::
    """
    if motion_filter_type not in ("lp", "notch", None):
        raise ValueError(f"Motion filter type '{motion_filter_type}' not supported.")

    # Select the motion columns from the overall confounds DataFrame
    motion_confounds_df = confounds_df[
        ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    ]

    # Apply LP or notch filter
    if motion_filter_type in ("lp", "notch"):
        motion_confounds = motion_regression_filter(
            data=motion_confounds_df.to_numpy(),
            TR=TR,
            motion_filter_type=motion_filter_type,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_order=motion_filter_order,
        )
        motion_confounds_df = pd.DataFrame(
            data=motion_confounds,
            columns=motion_confounds_df.columns,
        )

    return motion_confounds_df


@fill_doc
def get_custom_confounds(custom_confounds_folder, fmriprep_confounds_file):
    """Identify a custom confounds file.

    Parameters
    ----------
    %(custom_confounds_folder)s
    %(fmriprep_confounds_file)s
        We expect the custom confounds file to have the same name.

    Returns
    -------
    %(custom_confounds_file)s
    """
    import os

    if custom_confounds_folder is None:
        return None

    if not os.path.isdir(custom_confounds_folder):
        raise ValueError(f"Custom confounds location does not exist: {custom_confounds_folder}")

    custom_confounds_filename = os.path.basename(fmriprep_confounds_file)
    custom_confounds_file = os.path.abspath(
        os.path.join(
            custom_confounds_folder,
            custom_confounds_filename,
        )
    )

    if not os.path.isfile(custom_confounds_file):
        raise FileNotFoundError(f"Custom confounds file not found: {custom_confounds_file}")

    return custom_confounds_file


def consolidate_confounds(
    img_file,
    params,
    custom_confounds_file=None,
):
    """Combine confounds files into a single tsv.

    NOTE: This is a Node function.

    Parameters
    ----------
    img_file : :obj:`str`
        bold file
    params
    custom_confounds_file : :obj:`str` or None
        Path to custom confounds tsv. May be None.

    Returns
    -------
    confounds_file : :obj:`str`
        Path to combined tsv.
    """
    import os

    import numpy as np

    from xcp_d.utils.confounds import load_confound_matrix

    confounds_df = load_confound_matrix(
        img_file=img_file,
        params=params,
        custom_confounds=custom_confounds_file,
    )
    confounds_df["linear_trend"] = np.arange(confounds_df.shape[0])
    confounds_df["intercept"] = np.ones(confounds_df.shape[0])

    confounds_file = os.path.abspath("confounds.tsv")
    confounds_df.to_csv(confounds_file, sep="\t", index=False)

    return confounds_file


@fill_doc
def describe_regression(params, custom_confounds_file):
    """Build a text description of the regression that will be performed.

    Parameters
    ----------
    %(params)s
    %(custom_confounds_file)s

    Returns
    -------
    desc : :obj:`str`
        A text description of the regression.
    """
    import pandas as pd

    use_custom_confounds, orth = False, False
    if custom_confounds_file is not None:
        use_custom_confounds = True
        custom_confounds = pd.read_table(custom_confounds_file)
        orth = any([c.startswith("signal__") for c in custom_confounds.columns])

    BASE_DESCRIPTIONS = {
        "custom": "A custom set of regressors was used, with no other regressors from XCP-D.",
        "24P": (
            "In total, 24 nuisance regressors were selected from the preprocessing confounds, "
            "according to the '24P' strategy. "
            "These nuisance regressors included "
            "six motion parameters with their temporal derivatives, "
            "and their quadratic expansion of those six motion parameters and their "
            "temporal derivatives [@benchmarkp;@satterthwaite_2013]."
        ),
        "27P": (
            "In total, 27 nuisance regressors were selected from the preprocessing confounds, "
            "according to the '27P' strategy. "
            "These nuisance regressors included "
            "six motion parameters with their temporal derivatives, "
            "the quadratic expansion of those six motion parameters and their derivatives, "
            "mean global signal, mean white matter signal, and mean CSF signal "
            "[@benchmarkp;@satterthwaite_2013]."
        ),
        "36P": (
            "In total, 36 nuisance regressors were selected from the preprocessing confounds, "
            "according to the '36P' strategy. "
            "These nuisance regressors included "
            "six motion parameters, mean global signal, mean white matter signal, "
            "mean CSF signal with their temporal derivatives, "
            "and the quadratic expansion of six motion parameters, tissues signals and "
            "their temporal derivatives [@benchmarkp;@satterthwaite_2013]."
        ),
        "acompcor": (
            "Nuisance regressors were selected according to the 'acompcor' strategy. "
            "The top 5 aCompCor principal components from the WM and CSF compartments "
            "were selected as nuisance regressors [@behzadi2007component], "
            "along with the six motion parameters and their temporal derivatives "
            "[@benchmarkp;@satterthwaite_2013]. "
            "As the aCompCor regressors were generated on high-pass filtered data, "
            "the associated cosine basis regressors were included. "
            "This has the effect of high-pass filtering the data as well."
        ),
        "acompcor_gsr": (
            "Nuisance regressors were selected according to the 'acompcor_gsr' strategy. "
            "The top 5 aCompCor principal components from the WM and CSF compartments "
            "were selected as nuisance regressors [@behzadi2007component], "
            "along with the six motion parameters and their temporal derivatives, "
            "mean white matter signal, mean CSF signal, and mean global signal "
            "[@benchmarkp;@satterthwaite_2013]. "
            "As the aCompCor regressors were generated on high-pass filtered data, "
            "the associated cosine basis regressors were included. "
            "This has the effect of high-pass filtering the data as well."
        ),
        "aroma": (
            "Nuisance regressors were selected according to the 'aroma' strategy. "
            "AROMA motion-labeled components [@pruim2015ica], mean white matter signal, "
            "and mean CSF signal were selected as nuisance regressors "
            "[@benchmarkp;@satterthwaite_2013]."
        ),
        "aroma_gsr": (
            "Nuisance regressors were selected according to the 'aroma_gsr' strategy. "
            "AROMA motion-labeled components [@pruim2015ica], mean white matter signal, "
            "mean CSF signal, and mean global signal were selected as nuisance regressors "
            "[@benchmarkp;@satterthwaite_2013]."
        ),
    }

    if params not in BASE_DESCRIPTIONS.keys():
        raise ValueError(f"Unrecognized parameter string '{params}'")

    desc = BASE_DESCRIPTIONS[params]
    if use_custom_confounds and params != "custom":
        desc += " Additionally, custom confounds were also included as nuisance regressors."

    if "aroma" not in params and orth:
        desc += (
            " Custom confounds prefixed with 'signal__' were used to account for variance "
            "explained by known signals. "
            "Prior to denoising the BOLD data, the nuisance confounds were orthogonalized "
            "with respect to the signal regressors."
        )
    elif "aroma" in params and not orth:
        desc += (
            " AROMA non-motion components (i.e., ones assumed to reflect signal) were used to "
            "account for variance by known signals. "
            "Prior to denoising the BOLD data, the nuisance confounds were orthogonalized "
            "with respect to the non-motion components."
        )

    if "aroma" in params or orth:
        desc += (
            " In this way, the confound regressors were orthogonalized to produce regressors "
            "without variance explained by known signals, so that signal would not be removed "
            "from the BOLD data in the later regression."
        )

    desc += (
        " Finally, linear trend and intercept terms were added to the regressors prior to "
        "denoising."
    )

    return desc


@fill_doc
def describe_censoring(
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    head_radius,
    fd_thresh,
):
    """Build a text description of the motion parameter filtering and FD censoring process.

    Parameters
    ----------
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(head_radius)s
    %(fd_thresh)s

    Returns
    -------
    desc : :obj:`str`
        A text description of the censoring procedure.
    """
    from num2words import num2words

    filter_str, filter_post_str = "", ""
    if motion_filter_type:
        if motion_filter_type == "notch":
            filter_sub_str = (
                f"band-stop filtered to remove signals between {band_stop_min} and "
                f"{band_stop_max} breaths-per-minute using a(n) "
                f"{num2words(motion_filter_order, ordinal=True)}-order notch filter, "
                "based on @fair2020correction"
            )
        else:  # lp
            filter_sub_str = (
                f"low-pass filtered below {band_stop_min} breaths-per-minute using a(n) "
                f"{num2words(motion_filter_order, ordinal=True)}-order Butterworth filter, "
                "based on @gratton2020removal"
            )

        filter_str = (
            f"the six translation and rotation head motion traces were {filter_sub_str}. Next, "
        )
        filter_post_str = (
            "The filtered versions of the motion traces and framewise displacement were not used "
            "for denoising."
        )

    return (
        f"In order to identify high-motion outlier volumes, {filter_str}"
        "framewise displacement was calculated using the formula from @power_fd_dvars, "
        f"with a head radius of {head_radius} mm. "
        f"Volumes with {'filtered ' if motion_filter_type else ''}framewise displacement "
        f"greater than {fd_thresh} mm were flagged as high-motion outliers for the sake of later "
        f"censoring [@power_fd_dvars]. {filter_post_str}"
    )


@fill_doc
def load_confound_matrix(params, img_file, custom_confounds=None):
    """Load a subset of the confounds associated with a given file.

    Parameters
    ----------
    %(params)s
    img_file : :obj:`str`
        The path to the bold file.
    custom_confounds : :obj:`str` or None, optional
        Custom confounds TSV if there is one. Default is None.

    Returns
    -------
    confounds_df : pandas.DataFrame
        The loaded and selected confounds.
        If "AROMA" is requested, then this DataFrame will include signal components as well.
        These will be named something like "signal_[XX]".
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
    }

    if params in PARAM_KWARGS.keys():
        kwargs = PARAM_KWARGS[params]
        confounds_df = load_confounds(img_file, **kwargs)[0]

    elif params == "custom":
        # For custom confounds with no other confounds
        confounds_df = pd.read_table(custom_confounds, sep="\t")

    else:
        raise ValueError(f"Unrecognized parameter string '{params}'")

    if "aroma" in params:
        ica_mixing_matrix = _get_mixing_matrix(img_file)
        aroma_noise_comps_idx = _get_aroma_noise_comps(img_file)
        labeled_ica_mixing_matrix = _label_mixing_matrix(ica_mixing_matrix, aroma_noise_comps_idx)
        confounds_df = pd.concat([confounds_df, labeled_ica_mixing_matrix], axis=1)

    if params != "custom" and custom_confounds is not None:
        # For both custom and fMRIPrep confounds
        custom_confounds_df = pd.read_table(custom_confounds, sep="\t")
        confounds_df = pd.concat([custom_confounds_df, confounds_df], axis=1)

    return confounds_df


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

    sampling_frequency = 1 / TR
    nyquist_frequency = sampling_frequency / 2

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
        if lowpass_hz_adjusted != lowpass_hz:
            warnings.warn(
                f"Low-pass filter frequency is above Nyquist frequency ({nyquist_frequency} Hz), "
                f"so it has been changed ({lowpass_hz} --> {lowpass_hz_adjusted} Hz)."
            )

        b, a = butter(
            motion_filter_order / 2,
            lowpass_hz_adjusted,
            btype="lowpass",
            output="ba",
            fs=sampling_frequency,
        )
        filtered_data = filtfilt(b, a, data, axis=0, padtype="constant", padlen=data.shape[0] - 1)

    elif motion_filter_type == "notch":  # notch filter
        # Retain any frequencies *outside* the band_stop_min-band_stop_max range.
        assert band_stop_max is not None
        assert band_stop_min is not None
        assert band_stop_max > 0
        assert band_stop_min > 0
        assert band_stop_min < band_stop_max

        stopband = np.array([band_stop_min, band_stop_max])
        stopband_hz = stopband / 60  # change BPM to Hertz

        # Adjust frequencies in case Nyquist is within/below band.
        # This won't have an effect if the data have a fast enough sampling rate.
        stopband_hz_adjusted = np.abs(
            stopband_hz
            - (np.floor((stopband_hz + nyquist_frequency) / sampling_frequency))
            * sampling_frequency
        )
        if not np.array_equal(stopband_hz, stopband_hz_adjusted):
            warnings.warn(
                "One or both filter frequencies are above Nyquist frequency "
                f"({nyquist_frequency} Hz), "
                "so they have been changed "
                f"({stopband_hz[0]} --> {stopband_hz_adjusted[0]}, "
                f"{stopband_hz[1]} --> {stopband_hz_adjusted[1]} Hz)."
            )

        # Convert stopband to a single notch frequency.
        freq_to_remove = np.mean(stopband_hz_adjusted)
        bandwidth = np.abs(np.diff(stopband_hz_adjusted))

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
