# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Confound matrix selection based on Ciric et al. 2007."""
import os
import warnings

import numpy as np
import pandas as pd
from nilearn.interfaces.fmriprep import load_confounds
from nipype import logging
from scipy.signal import butter, filtfilt

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


def get_customfile(custom_confounds_folder, fmriprep_confounds_file):
    """Identify a custom confounds file.

    Parameters
    ----------
    custom_confounds_folder : str or None
        The path to the custom confounds file.
    fmriprep_confounds_file : str
        Path to the confounds file from the preprocessing pipeline.
        We expect the custom confounds file to have the same name.

    Returns
    -------
    custom_confounds_file : str or None
        The appropriate custom confounds file.
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

    Parameters
    ----------
    img_file : str
        bold file
    params
    custom_confounds_folder : str or None
        Path to custom confounds tsv. May be None.

    Returns
    -------
    out_file : str
        Path to combined tsv.
    """
    import os

    from xcp_d.utils.confounds import load_confound_matrix

    confounds_df = load_confound_matrix(
        img_file=img_file,
        params=params,
        custom_confounds=custom_confounds_file,
    )

    out_file = os.path.abspath("confounds.tsv")
    confounds_df.to_csv(out_file, sep="\t", index=False)

    return out_file


@fill_doc
def describe_regression(params, custom_confounds_file):
    """Build a text description of the regression that will be performed.

    Parameters
    ----------
    %(params)s
    custom_confounds_file : str or None

    Returns
    -------
    desc : str
        A text description of the regression.
    """
    import nilearn
    import pandas as pd

    use_custom_confounds, non_aggro = False, False
    if custom_confounds_file is not None:
        use_custom_confounds = True
        custom_confounds = pd.read_table(custom_confounds_file)
        non_aggro = any([c.startswith("signal__") for c in custom_confounds.columns])

    BASE_DESCRIPTIONS = {
        "custom": "A custom set of regressors was used, with no other regressors from XCP-D. ",
        "24P": (
            "In total, 24 nuisance regressors were selected from the preprocessing confounds. "
            "These nuisance regressors included "
            "six motion parameters with their temporal derivatives, "
            "and their quadratic expansion of those six motion parameters and their "
            "temporal derivatives [@benchmarkp;@satterthwaite_2013]. "
        ),
        "27P": (
            "In total, 27 nuisance regressors were selected from the preprocessing confounds. "
            "These nuisance regressors included "
            "six motion parameters with their temporal derivatives, "
            "the quadratic expansion of those six motion parameters and their derivatives, "
            "mean global signal, mean white matter signal, and mean CSF signal "
            "[@benchmarkp;@satterthwaite_2013]. "
        ),
        "36P": (
            "In total, 36 nuisance regressors were selected from the preprocessing confounds. "
            "These nuisance regressors included "
            "six motion parameters, mean global signal, mean white matter signal, "
            "mean CSF signal with their temporal derivatives, "
            "and the quadratic expansion of six motion parameters, tissues signals and "
            "their temporal derivatives [@benchmarkp;@satterthwaite_2013]. "
        ),
        "acompcor": (
            "The top 5 aCompCor principal components from the WM and CSF compartments "
            "were selected as nuisance regressors [@behzadi2007component], "
            "along with the six motion parameters and their temporal derivatives "
            "[@benchmarkp;@satterthwaite_2013]. "
        ),
        "acompcor_gsr": (
            "The top 5 aCompCor principal components from the WM and CSF compartments "
            "were selected as nuisance regressors [@behzadi2007component], "
            "along with the six motion parameters and their temporal derivatives, "
            "mean white matter signal, mean CSF signal, and mean global signal "
            "[@benchmarkp;@satterthwaite_2013]. "
        ),
        "aroma": (
            "AROMA motion-labeled components [@pruim2015ica], mean white matter signal, "
            "and mean CSF signal were selected as nuisance regressors "
            "[@benchmarkp;@satterthwaite_2013]. "
        ),
        "aroma_gsr": (
            "AROMA motion-labeled components [@pruim2015ica], mean white matter signal, "
            "mean CSF signal, and mean global signal were selected as nuisance regressors "
            "[@benchmarkp;@satterthwaite_2013]. "
        ),
    }

    if params not in BASE_DESCRIPTIONS.keys():
        raise ValueError(f"Unrecognized parameter string '{params}'")

    desc = BASE_DESCRIPTIONS[params]
    if use_custom_confounds and params != "custom":
        desc += "Additionally, custom confounds were also included as nuisance regressors. "

    if "aroma" not in params and non_aggro:
        desc += (
            "Custom confounds prefixed with 'signal__' were used to account for variance "
            "explained by known signals. "
            "These regressors were included in the regression, "
            f"as implemented in nilearn {nilearn.__version__} [@nilearn], "
            "after which the resulting parameter estimates from only the nuisance regressors "
            "were used to denoise the BOLD data. "
        )
    elif "aroma" in params and not non_aggro:
        desc += (
            "AROMA non-motion components (i.e., ones assumed to reflect signal) were also "
            "included in the regression, "
            f"as implemented in nilearn {nilearn.__version__} [@nilearn], "
            "after which the resulting parameter estimates from only the nuisance regressors "
            "were used to denoise the BOLD data. "
        )

    if "aroma" in params or non_aggro:
        desc += (
            "In this way, shared variance between the nuisance regressors "
            "and the signal regressors was separated smartly, "
            "so that signal would not be removed by the regression. "
            "This is colloquially known as 'non-aggressive' denoising, "
            "and is the recommended denoising method when nuisance regressors may share variance "
            "with known signal regressors [@pruim2015ica]."
        )
    else:
        desc += (
            "These nuisance regressors were regressed from the BOLD data using "
            "linear regression, "
            f"as implemented in nilearn {nilearn.__version__} [@nilearn]."
        )

    return desc


@fill_doc
def load_confound_matrix(params, img_file, custom_confounds=None):
    """Load a subset of the confounds associated with a given file.

    Parameters
    ----------
    %(params)s
    img_file : str
        The path to the bold file.
    custom_confounds : str or None, optional
        Custom confounds TSV if there is one. Default is None.

    Returns
    -------
    confound : pandas.DataFrame
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
        confound = load_confounds(img_file, **kwargs)[0]

    elif params == "custom":
        # For custom confounds with no other confounds
        confound = pd.read_table(custom_confounds, sep="\t")

    else:
        raise ValueError(f"Unrecognized parameter string '{params}'")

    if "aroma" in params:
        ica_mixing_matrix = _get_mixing_matrix(img_file)
        aroma_noise_comps_idx = _get_aroma_noise_comps(img_file)
        labeled_ica_mixing_matrix = _label_mixing_matrix(ica_mixing_matrix, aroma_noise_comps_idx)
        confound = pd.concat([confound, labeled_ica_mixing_matrix], axis=1)

    if params != "custom" and custom_confounds is not None:
        # For both custom and fMRIPrep confounds
        custom = pd.read_table(custom_confounds, sep="\t")
        confound = pd.concat([custom, confound], axis=1)

    return confound


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
    data : (V, T) numpy.ndarray
        Data to filter. V = variables, T = time
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

    sampling_frequency = 1 / TR
    nyquist_frequency = sampling_frequency / 2

    if motion_filter_type == "lp":  # low-pass filter
        # Remove any frequencies above band_stop_min.
        assert band_stop_min is not None
        assert band_stop_min > 0
        if band_stop_max:
            warnings.warn("The parameter 'band_stop_max' will be ignored.")

        low_pass_freq_hertz = band_stop_min / 60  # change BPM to right time unit

        highcut = np.float(low_pass_freq_hertz) / nyquist_frequency

        b, a = butter(order / 2, highcut, btype="lowpass", output="ba")  # get filter coeff

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
        bandstop_cuts = bandstop_band_hz / nyquist_frequency

        b, a = butter(order / 2, bandstop_cuts, btype="bandstop", output="ba")  # get filter coeff

    filtered_data = np.zeros(data.shape)  # create something to populate filtered values with

    # apply the filter, loop through columns of regressors
    for i_row in range(filtered_data.shape[0]):
        filtered_data[i_row, :] = filtfilt(b, a, data[i_row, :], padtype="constant")

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
