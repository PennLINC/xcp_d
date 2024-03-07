#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miscellaneous utility functions for xcp_d."""
import gc

import nibabel as nb
import numpy as np
from nipype import logging

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


def get_bold2std_and_t1w_xfms(bold_file, template_to_anat_xfm):
    """Find transform files in reverse order to transform BOLD to MNI152NLin2009cAsym/T1w space.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    NOTE: This is a Node function.

    Parameters
    ----------
    bold_file : :obj:`str`
        The preprocessed BOLD file.
    template_to_anat_xfm
        The ``from`` field is assumed to be the same space as the BOLD file is in.
        The MNI space could be MNI152NLin2009cAsym, MNI152NLin6Asym, or MNIInfant.

    Returns
    -------
    xforms_to_MNI : list of str
        A list of paths to transform files for warping to MNI152NLin2009cAsym space.
    xforms_to_MNI_invert : list of bool
        A list of booleans indicating whether each transform in xforms_to_MNI indicating
        if each should be inverted (True) or not (False).
    xforms_to_T1w : list of str
        A list of paths to transform files for warping to T1w space.
    xforms_to_T1w_invert : list of bool
        A list of booleans indicating whether each transform in xforms_to_T1w indicating
        if each should be inverted (True) or not (False).

    Notes
    -----
    Only used for QCReport in init_postprocess_nifti_wf.
    QCReport wants MNI-space data in MNI152NLin2009cAsym.
    """
    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    from xcp_d.utils.bids import get_entity

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, "space")

    if bold_space in ("native", "T1w"):
        base_std_space = get_entity(template_to_anat_xfm, "from")
        raise ValueError(f"BOLD space '{bold_space}' not supported.")
    elif f"from-{bold_space}" not in template_to_anat_xfm:
        raise ValueError(
            f"Transform does not match BOLD space: {bold_space} != {template_to_anat_xfm}"
        )

    # Pull out the correct transforms based on bold_file name and string them together.
    xforms_to_T1w = [template_to_anat_xfm]  # used for all spaces except T1w and native
    xforms_to_T1w_invert = [False]
    if bold_space == "MNI152NLin2009cAsym":
        # Data already in MNI152NLin2009cAsym space.
        xforms_to_MNI = ["identity"]
        xforms_to_MNI_invert = [False]

    elif bold_space == "MNI152NLin6Asym":
        # MNI152NLin6Asym --> MNI152NLin2009cAsym
        MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
            get_template(
                template="MNI152NLin2009cAsym",
                mode="image",
                suffix="xfm",
                extension=".h5",
                **{"from": "MNI152NLin6Asym"},
            ),
        )
        xforms_to_MNI = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]
        xforms_to_MNI_invert = [False]

    elif bold_space == "MNIInfant":
        # MNIInfant --> MNI152NLin2009cAsym
        MNIInfant_to_MNI152NLin2009cAsym = pkgrf(
            "xcp_d",
            "data/transform/tpl-MNIInfant_from-MNI152NLin2009cAsym_mode-image_xfm.h5",
        )
        xforms_to_MNI = [MNIInfant_to_MNI152NLin2009cAsym]
        xforms_to_MNI_invert = [False]

    elif bold_space == "T1w":
        # T1w --> ?? (extract from template_to_anat_xfm) --> MNI152NLin2009cAsym
        # Should not be reachable, since xcpd doesn't support T1w-space BOLD inputs
        if base_std_space != "MNI152NLin2009cAsym":
            std_to_mni_xfm = str(
                get_template(
                    template="MNI152NLin2009cAsym",
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": base_std_space},
                ),
            )
            xforms_to_MNI = [std_to_mni_xfm, template_to_anat_xfm]
            xforms_to_MNI_invert = [False, True]
        else:
            xforms_to_MNI = [template_to_anat_xfm]
            xforms_to_MNI_invert = [True]

        xforms_to_T1w = ["identity"]
        xforms_to_T1w_invert = [False]

    else:
        raise ValueError(f"Space '{bold_space}' in {bold_file} not supported.")

    return xforms_to_MNI, xforms_to_MNI_invert, xforms_to_T1w, xforms_to_T1w_invert


def get_std2bold_xfms(bold_file):
    """Obtain transforms to warp atlases from MNI152NLin6Asym to the same template as the BOLD.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    NOTE: This is a Node function.

    Parameters
    ----------
    bold_file : :obj:`str`
        The preprocessed BOLD file.

    Returns
    -------
    transform_list : list of str
        A list of paths to transform files.

    Notes
    -----
    Used by:

    - to resample dseg in init_postprocess_nifti_wf for QCReport
    - to warp atlases to the same space as the BOLD data in init_functional_connectivity_nifti_wf
    - to resample dseg to BOLD space for the executive summary plots

    Does not include inversion flag output because there is no need (yet).
    Can easily be added in the future.
    """
    import os

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    from xcp_d.utils.bids import get_entity

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, "space")

    # Load useful inter-template transforms from templateflow
    MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
        get_template(
            template="MNI152NLin2009cAsym",
            mode="image",
            suffix="xfm",
            extension=".h5",
            **{"from": "MNI152NLin6Asym"},
        ),
    )

    # Find the appropriate transform(s)
    if bold_space == "MNI152NLin6Asym":
        # NLin6 --> NLin6 (identity)
        transform_list = ["identity"]

    elif bold_space == "MNI152NLin2009cAsym":
        # NLin6 --> NLin2009c
        transform_list = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]

    elif bold_space == "MNIInfant":
        # NLin6 --> NLin2009c --> MNIInfant
        MNI152NLin2009cAsym_to_MNI152Infant = pkgrf(
            "xcp_d",
            "data/transform/tpl-MNIInfant_from-MNI152NLin2009cAsym_mode-image_xfm.h5",
        )
        transform_list = [
            MNI152NLin2009cAsym_to_MNI152Infant,
            MNI152NLin6Asym_to_MNI152NLin2009cAsym,
        ]

    else:
        file_base = os.path.basename(bold_file)
        raise ValueError(f"Space '{bold_space}' in {file_base} not supported.")

    return transform_list


def fwhm2sigma(fwhm):
    """Convert full width at half maximum to sigma.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum.

    Returns
    -------
    float
        Sigma.
    """
    return fwhm / np.sqrt(8 * np.log(2))


def butter_bandpass(
    data,
    sampling_rate,
    low_pass,
    high_pass,
    padtype="constant",
    padlen=None,
    order=2,
):
    """Apply a Butterworth bandpass filter to data.

    Parameters
    ----------
    data : (T, S) numpy.ndarray
        Time by voxels/vertices array of data.
    sampling_rate : float
        Sampling frequency. 1/TR(s).
    low_pass : float
        frequency, in Hertz
    high_pass : float
        frequency, in Hertz
    padlen
    padtype
    order : int
        The order of the filter.

    Returns
    -------
    filtered_data : (T, S) numpy.ndarray
        The filtered data.
    """
    from scipy.signal import butter, filtfilt

    if low_pass > 0 and high_pass > 0:
        btype = "bandpass"
        filt_input = [high_pass, low_pass]
    elif high_pass > 0:
        btype = "highpass"
        filt_input = high_pass
    elif low_pass > 0:
        btype = "lowpass"
        filt_input = low_pass
    else:
        raise ValueError("Filter parameters are not valid.")

    b, a = butter(
        order,
        filt_input,
        btype=btype,
        output="ba",
        fs=sampling_rate,  # eliminates need to normalize cutoff frequencies
    )

    filtered_data = np.zeros_like(data)  # create something to populate filtered values with

    # apply the filter, loop through columns of regressors
    for i_voxel in range(filtered_data.shape[1]):
        filtered_data[:, i_voxel] = filtfilt(
            b,
            a,
            data[:, i_voxel],
            padtype=padtype,
            padlen=padlen,
        )

    return filtered_data


@fill_doc
def estimate_brain_radius(mask_file, head_radius="auto"):
    """Estimate brain radius from binary brain mask file.

    Parameters
    ----------
    mask_file : :obj:`str`
        Binary brain mask file, in nifti format.
    %(head_radius)s

    Returns
    -------
    brain_radius : float
        Estimated brain radius, in millimeters.

    Notes
    -----
    This function estimates the brain radius based on the brain volume,
    assuming that the brain is a sphere.
    This was Paul Taylor's idea, shared in this NeuroStars post:
    https://neurostars.org/t/estimating-head-brain-radius-automatically/24290/2.
    """
    if head_radius == "auto":
        mask_img = nb.load(mask_file)
        mask_data = mask_img.get_fdata()
        n_voxels = np.sum(mask_data)
        voxel_size = np.prod(mask_img.header.get_zooms())
        volume = n_voxels * voxel_size

        brain_radius = ((3 * volume) / (4 * np.pi)) ** (1 / 3)

        LOGGER.info(f"Brain radius estimated at {brain_radius} mm.")

    else:
        brain_radius = head_radius

    return brain_radius


@fill_doc
def denoise_with_nilearn(
    preprocessed_bold,
    confounds_file,
    temporal_mask,
    low_pass,
    high_pass,
    filter_order,
    TR,
):
    """Denoise an array with Nilearn.

    This step does the following:

        1. Censor the data and associated confounds.
        2. Mean-center the censored and uncensored confounds, based on the censored confounds.
        3. Estimate betas using only the censored data.
        4. Apply the betas to denoise the *full* (uncensored) BOLD data.
        5. Apply the betas to denoise the censored BOLD data.
        6. Interpolate the censored, denoised data.
        7. Bandpass filter the interpolated, denoised data.

    Parameters
    ----------
    preprocessed_bold : :obj:`numpy.ndarray` of shape (T, S)
        Preprocessed BOLD data, after dummy volume removal,
        but without any additional censoring.
    confounds_file : :obj:`str` or None
        Path to TSV file containing selected confounds, after dummy volume removal,
        but without any additional censoring.
        May be None, if no denoising should be performed.
    %(temporal_mask)s
    low_pass, high_pass : float or None
        Lowpass and high_pass thresholds, in Hertz.
    filter_order : int
        Filter order.
    %(TR)s

    Returns
    -------
    %(uncensored_denoised_bold)s
        Returned as a :obj:`numpy.ndarray` of shape (T, S)
    %(interpolated_filtered_bold)s
        Returned as a :obj:`numpy.ndarray` of shape (T, S)
        This is the primary output.
    """
    import pandas as pd
    from nilearn import signal

    n_volumes, n_voxels = preprocessed_bold.shape
    censoring_df = pd.read_table(temporal_mask)
    # Only remove high-motion outliers in this step (not the random volumes for trimming).
    sample_mask = ~censoring_df["framewise_displacement"].to_numpy().astype(bool)
    outlier_idx = list(np.where(~sample_mask)[0])

    denoise = bool(confounds_file)
    censor_and_interpolate = bool(outlier_idx)
    if denoise:
        confounds_df = pd.read_table(confounds_file)

        assert "intercept" in confounds_df.columns
        assert "linear_trend" in confounds_df.columns
        assert confounds_df.columns[-1] == "intercept"

    # Censor the data and confounds
    if censor_and_interpolate:
        preprocessed_bold_censored = preprocessed_bold[sample_mask, :]
    else:
        preprocessed_bold_censored = preprocessed_bold

    if denoise:
        nuisance_arr = confounds_df.to_numpy()
        nuisance_censored = nuisance_arr[sample_mask, :]

        # Mean-center all of the confounds, except the intercept, to be safe
        nuisance_censored_mean = np.mean(nuisance_censored[:, :-1], axis=0)
        nuisance_arr[:, :-1] -= nuisance_censored_mean
        nuisance_censored[:, :-1] -= nuisance_censored_mean  # use censored mean on full regressors

        # Estimate betas using only the censored data
        betas = np.linalg.lstsq(nuisance_censored, preprocessed_bold_censored, rcond=None)[0]

        # Apply the betas to denoise the *full* (uncensored) BOLD data
        uncensored_denoised_bold = preprocessed_bold - np.dot(nuisance_arr, betas)

        # Also denoise the censored BOLD data
        censored_denoised_bold = preprocessed_bold_censored - np.dot(nuisance_censored, betas)

        del preprocessed_bold, preprocessed_bold_censored, betas
        gc.collect()
    else:
        uncensored_denoised_bold = preprocessed_bold
        censored_denoised_bold = preprocessed_bold_censored

    if censor_and_interpolate:
        # Now interpolate the censored, denoised data with cubic spline interpolation
        interpolated_unfiltered_bold = np.zeros(
            (n_volumes, n_voxels),
            dtype=censored_denoised_bold.dtype,
        )
        interpolated_unfiltered_bold[sample_mask, :] = censored_denoised_bold
        del censored_denoised_bold
        gc.collect()

        interpolated_unfiltered_bold = signal._interpolate_volumes(
            interpolated_unfiltered_bold,
            sample_mask=sample_mask,
            t_r=TR,
        )

        # Replace any high-motion volumes at the beginning or end of the run with the closest
        # low-motion volume's data.
        # Use https://stackoverflow.com/a/48106843/2589328 to group consecutive blocks of outliers.
        gaps = [[s, e] for s, e in zip(outlier_idx, outlier_idx[1:]) if s + 1 < e]
        edges = iter(outlier_idx[:1] + sum(gaps, []) + outlier_idx[-1:])
        consecutive_outliers_idx = list(zip(edges, edges))
        first_outliers = consecutive_outliers_idx[0]
        last_outliers = consecutive_outliers_idx[-1]

        # Replace outliers at beginning of run
        if first_outliers[0] == 0:
            LOGGER.warning(
                f"Outlier volumes at beginning of run ({first_outliers[0]}-{first_outliers[1]}) "
                "will be replaced with first non-outlier volume's values."
            )
            interpolated_unfiltered_bold[: first_outliers[1] + 1, :] = (
                interpolated_unfiltered_bold[first_outliers[1] + 1, :]
            )

        # Replace outliers at end of run
        if last_outliers[1] == n_volumes - 1:
            LOGGER.warning(
                f"Outlier volumes at end of run ({last_outliers[0]}-{last_outliers[1]}) "
                "will be replaced with last non-outlier volume's values."
            )
            interpolated_unfiltered_bold[last_outliers[0] :, :] = interpolated_unfiltered_bold[
                last_outliers[0] - 1, :
            ]

    else:
        interpolated_unfiltered_bold = censored_denoised_bold

    # Now apply the bandpass filter to the interpolated, denoised data
    if low_pass is not None and high_pass is not None:
        # TODO: Replace with nilearn.signal.butterworth once 0.10.1 is released.
        interpolated_filtered_bold = butter_bandpass(
            interpolated_unfiltered_bold,
            sampling_rate=1 / TR,
            low_pass=low_pass,
            high_pass=high_pass,
            order=filter_order / 2,
            padtype="constant",
            padlen=n_volumes - 1,
        )
    else:
        interpolated_filtered_bold = interpolated_unfiltered_bold

    return uncensored_denoised_bold, interpolated_filtered_bold


@fill_doc
def denoise_hpmc(
    preprocessed_bold,
    confounds_file,
    temporal_mask,
    low_pass,
    high_pass,
    filter_order,
    TR,
):
    """Denoise an array with Nilearn.

    This step does the following:

        1. Interpolate high-motion volumes in the BOLD data and confounds.
        2. Detrend interpolated BOLD and confounds.
        3. Bandpass filter the interpolated data and confounds.
        4. Censor the data and confounds.
        5. Mean-center the censored and uncensored confounds, based on the censored confounds.
        6. Estimate betas using only the low-motion volumes.
        7. Apply the betas to denoise the *full* (uncensored) BOLD data.
        5. Apply the betas to denoise the censored BOLD data.
        6. Interpolate the censored, denoised data.
        7. Bandpass filter the interpolated, denoised data.

    Parameters
    ----------
    preprocessed_bold : :obj:`numpy.ndarray` of shape (T, S)
        Preprocessed BOLD data, after dummy volume removal,
        but without any additional censoring.
    confounds_file : :obj:`str` or None
        Path to TSV file containing selected confounds, after dummy volume removal,
        but without any additional censoring.
        May be None, if no denoising should be performed.
    %(temporal_mask)s
    low_pass, high_pass : float or None
        Lowpass and high_pass thresholds, in Hertz.
    filter_order : int
        Filter order.
    %(TR)s

    Returns
    -------
    %(uncensored_denoised_bold)s
        Returned as a :obj:`numpy.ndarray` of shape (T, S)
    %(interpolated_filtered_bold)s
        Returned as a :obj:`numpy.ndarray` of shape (T, S)
        This is the primary output.
    """
    import pandas as pd

    n_volumes = preprocessed_bold.shape[0]
    censoring_df = pd.read_table(temporal_mask)
    # Only remove high-motion outliers in this step (not the random volumes for trimming).
    sample_mask = ~censoring_df["framewise_displacement"].to_numpy().astype(bool)
    outlier_idx = list(np.where(~sample_mask)[0])

    denoise = bool(confounds_file)
    censor_and_interpolate = bool(outlier_idx)
    if denoise:
        confounds_df = pd.read_table(confounds_file)

        assert "intercept" in confounds_df.columns
        assert "linear_trend" in confounds_df.columns
        assert confounds_df.columns[-1] == "intercept"

    if censor_and_interpolate:
        interpolated_bold = _interpolate(arr=preprocessed_bold, sample_mask=sample_mask, TR=TR)
        if denoise:
            interpolated_confounds = _interpolate(
                arr=confounds_df.to_numpy(), sample_mask=sample_mask, TR=TR
            )

    else:
        interpolated_bold = preprocessed_bold
        if denoise:
            interpolated_confounds = confounds_df.to_numpy()

    # Now apply the bandpass filter to the interpolated data and confounds
    if (low_pass is not None) or (high_pass is not None):
        # TODO: Replace with nilearn.signal.butterworth once 0.10.1 is released.
        filtered_interpolated_bold = butter_bandpass(
            interpolated_bold,
            sampling_rate=1 / TR,
            low_pass=low_pass,
            high_pass=high_pass,
            order=filter_order / 2,
            padtype="constant",
            padlen=n_volumes - 1,
        )
        censored_filtered_bold = filtered_interpolated_bold[sample_mask, :]

        filtered_preprocessed_bold = butter_bandpass(
            preprocessed_bold,
            sampling_rate=1 / TR,
            low_pass=low_pass,
            high_pass=high_pass,
            order=filter_order / 2,
            padtype="constant",
            padlen=n_volumes - 1,
        )
        if denoise:
            filtered_interpolated_confounds = butter_bandpass(
                interpolated_confounds,
                sampling_rate=1 / TR,
                low_pass=low_pass,
                high_pass=high_pass,
                order=filter_order / 2,
                padtype="constant",
                padlen=n_volumes - 1,
            )
            filtered_confounds = butter_bandpass(
                confounds_df.to_numpy(),
                sampling_rate=1 / TR,
                low_pass=low_pass,
                high_pass=high_pass,
                order=filter_order / 2,
                padtype="constant",
                padlen=n_volumes - 1,
            )
            censored_filtered_confounds = filtered_interpolated_confounds[sample_mask, :]

    else:
        filtered_interpolated_bold = interpolated_bold
        censored_filtered_bold = interpolated_bold
        if denoise:
            filtered_interpolated_confounds = interpolated_confounds.copy()
            censored_filtered_confounds = interpolated_confounds.copy()
            filtered_confounds = confounds_df.to_numpy()

    # Denoise the data using the censored data and confounds
    if denoise:
        # Mean-center all of the confounds, except the intercept, to be safe
        nuisance_censored_mean = np.mean(censored_filtered_confounds[:, :-1], axis=0)
        censored_filtered_confounds[:, :-1] -= nuisance_censored_mean
        # use censored mean on full regressors
        filtered_confounds[:, :-1] -= nuisance_censored_mean

        # Estimate betas using only the censored data
        betas = np.linalg.lstsq(
            censored_filtered_confounds,
            censored_filtered_bold,
            rcond=None,
        )[0]

        # Denoise the interpolated data.
        # The low-motion volumes of the denoised, interpolated data will be the same as the full
        # censored, denoised data.
        denoised_interpolated_bold = filtered_interpolated_bold - np.dot(
            filtered_interpolated_confounds, betas
        )

        # Apply the betas to denoise the preprocessed BOLD data without interpolation or filtering
        denoised_filtered_bold = filtered_preprocessed_bold - np.dot(filtered_confounds, betas)

        del preprocessed_bold, betas
        gc.collect()
    else:
        denoised_filtered_bold = filtered_preprocessed_bold
        denoised_interpolated_bold = filtered_interpolated_bold

    return denoised_filtered_bold, denoised_interpolated_bold


def _interpolate(*, arr, sample_mask, TR):
    from nilearn import signal

    outlier_idx = list(np.where(~sample_mask)[0])
    n_volumes = arr.shape[0]

    interpolated_arr = signal._interpolate_volumes(
        arr,
        sample_mask=sample_mask,
        t_r=TR,
    )
    # Replace any high-motion volumes at the beginning or end of the run with the closest
    # low-motion volume's data.
    # Use https://stackoverflow.com/a/48106843/2589328 to group consecutive blocks of outliers.
    gaps = [[s, e] for s, e in zip(outlier_idx, outlier_idx[1:]) if s + 1 < e]
    edges = iter(outlier_idx[:1] + sum(gaps, []) + outlier_idx[-1:])
    consecutive_outliers_idx = list(zip(edges, edges))
    first_outliers = consecutive_outliers_idx[0]
    last_outliers = consecutive_outliers_idx[-1]

    # Replace outliers at beginning of run
    if first_outliers[0] == 0:
        LOGGER.warning(
            f"Outlier volumes at beginning of run ({first_outliers[0]}-{first_outliers[1]}) "
            "will be replaced with first non-outlier volume's values."
        )
        interpolated_arr[: first_outliers[1] + 1, :] = interpolated_arr[first_outliers[1] + 1, :]

    # Replace outliers at end of run
    if last_outliers[1] == n_volumes - 1:
        LOGGER.warning(
            f"Outlier volumes at end of run ({last_outliers[0]}-{last_outliers[1]}) "
            "will be replaced with last non-outlier volume's values."
        )
        interpolated_arr[last_outliers[0] :, :] = interpolated_arr[last_outliers[0] - 1, :]

    return interpolated_arr


def _select_first(lst):
    """Select the first element in a list."""
    return lst[0]


def list_to_str(lst):
    """Convert a list to a pretty string."""
    if not lst:
        raise ValueError("Zero-length list provided.")

    lst_str = [str(item) for item in lst]
    if len(lst_str) == 1:
        return lst_str[0]
    elif len(lst_str) == 2:
        return " and ".join(lst_str)
    else:
        return f"{', '.join(lst_str[:-1])}, and {lst_str[-1]}"


def _listify(obj):
    """Wrap all non-list or tuple objects in a list.

    This provides a simple way to accept flexible arguments.
    """
    return obj if isinstance(obj, (list, tuple, type(None), np.ndarray)) else [obj]


def _make_dictionary(metadata=None, **kwargs):
    """Create or modify a dictionary.

    This will add kwargs to a metadata dictionary if the dictionary is provided,
    or create a dictionary from scratch if not.
    """
    from copy import deepcopy

    from xcp_d.utils.utils import _listify

    if metadata:
        out_metadata = deepcopy(metadata)
        for key, value in kwargs.items():
            if key not in metadata.keys():
                out_metadata[key] = value
            elif isinstance(value, list) or isinstance(out_metadata[key], list):
                # Append the values if they're a list
                out_metadata[key] = _listify(out_metadata[key]) + _listify(value)
            else:
                # Overwrite the old value
                out_metadata[key] = value

        return out_metadata
    else:
        return dict(kwargs)


def _transpose_lol(lol):
    """Transpose list of lists."""
    return list(map(list, zip(*lol)))
