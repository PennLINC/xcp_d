#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miscellaneous utility functions for xcp_d."""
import warnings

import nibabel as nb
import numpy as np
from nipype import logging

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


def get_bold2std_and_t1w_xfms(bold_file, template_to_anat_xfm, anat_to_native_xfm):
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
    anat_to_native_xfm

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

    elif bold_space == "native":
        # native (BOLD) --> T1w --> ?? (extract from template_to_anat_xfm) --> MNI152NLin2009cAsym
        # Should not be reachable, since xcpd doesn't support native-space BOLD inputs
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
            xforms_to_MNI = [std_to_mni_xfm, template_to_anat_xfm, anat_to_native_xfm]
            xforms_to_MNI_invert = [False, True, True]
        else:
            xforms_to_MNI = [template_to_anat_xfm, anat_to_native_xfm]
            xforms_to_MNI_invert = [True, True]

        xforms_to_T1w = [anat_to_native_xfm]
        xforms_to_T1w_invert = [True]

    else:
        raise ValueError(f"Space '{bold_space}' in {bold_file} not supported.")

    return xforms_to_MNI, xforms_to_MNI_invert, xforms_to_T1w, xforms_to_T1w_invert


def get_std2bold_xfms(bold_file, template_to_anat_xfm, anat_to_native_xfm):
    """Obtain transforms to warp atlases from MNI152NLin6Asym to the same space as the BOLD.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    NOTE: This is a Node function.

    Parameters
    ----------
    bold_file : :obj:`str`
        The preprocessed BOLD file.
    %(template_to_anat_xfm)s
        The ``from`` field is assumed to be the same space as the BOLD file is in.
    %(anat_to_native_xfm)s

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

    # Check that the MNI-to-T1w xform is from the right space
    if bold_space in ("native", "T1w"):
        base_std_space = get_entity(template_to_anat_xfm, "from")
    elif f"from-{bold_space}" not in template_to_anat_xfm:
        raise ValueError(
            f"Transform does not match BOLD space: {bold_space} != {template_to_anat_xfm}"
        )

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

    elif bold_space == "T1w":
        # NLin6 --> ?? (extract from template_to_anat_xfm) --> T1w (BOLD)
        if base_std_space != "MNI152NLin6Asym":
            mni_to_std_xfm = str(
                get_template(
                    template=base_std_space,
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": "MNI152NLin6Asym"},
                ),
            )
            transform_list = [template_to_anat_xfm, mni_to_std_xfm]
        else:
            transform_list = [template_to_anat_xfm]

    elif bold_space == "native":
        # The BOLD data are in native space
        # NLin6 --> ?? (extract from template_to_anat_xfm) --> T1w --> native (BOLD)
        if base_std_space != "MNI152NLin6Asym":
            mni_to_std_xfm = str(
                get_template(
                    template=base_std_space,
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": "MNI152NLin6Asym"},
                ),
            )
            transform_list = [anat_to_native_xfm, template_to_anat_xfm, mni_to_std_xfm]
        else:
            transform_list = [anat_to_native_xfm, template_to_anat_xfm]

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

    b, a = butter(
        order,
        [high_pass, low_pass],
        btype="bandpass",
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

        1. Orthogonalize nuisance regressors w.r.t. any signal regressors.
        2. Censor the data and associated confounds.
        3. Mean-center the censored and uncensored confounds, based on the censored confounds.
        4. Estimate betas using only the censored data.
        5. Apply the betas to denoise the *full* (uncensored) BOLD data.
        6. Apply the betas to denoise the censored BOLD data.
        7. Interpolate the censored, denoised data.
        8. Bandpass filter the interpolated, denoised data.

    Parameters
    ----------
    preprocessed_bold : :obj:`numpy.ndarray` of shape (T, S)
        Preprocessed BOLD data, after dummy volume removal,
        but without any additional censoring.
    confounds_file : :obj:`str`
        Path to TSV file containing selected confounds, after dummy volume removal,
        but without any additional censoring.
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
    confounds_df = pd.read_table(confounds_file)

    assert "intercept" in confounds_df.columns
    assert "linear_trend" in confounds_df.columns
    assert confounds_df.columns[-1] == "intercept"

    censoring_df = pd.read_table(temporal_mask)
    sample_mask = ~censoring_df["framewise_displacement"].to_numpy().astype(bool)

    # Orthogonalize full nuisance regressors w.r.t. any signal regressors
    signal_columns = [c for c in confounds_df.columns if c.startswith("signal__")]
    if signal_columns:
        warnings.warn(
            "Signal columns detected. "
            "Orthogonalizing nuisance columns w.r.t. the following signal columns: "
            f"{', '.join(signal_columns)}"
        )
        noise_columns = [c for c in confounds_df.columns if not c.startswith("signal__")]
        # Don't orthogonalize the intercept or linear trend regressors
        columns_to_denoise = [c for c in noise_columns if c not in ["linear_trend", "intercept"]]
        temp_confounds_df = confounds_df[noise_columns].copy()

        signal_regressors = confounds_df[signal_columns].to_numpy()
        noise_regressors = confounds_df[columns_to_denoise].to_numpy()
        signal_betas = np.linalg.lstsq(signal_regressors, noise_regressors, rcond=None)[0]
        pred_noise_regressors = np.dot(signal_regressors, signal_betas)
        orth_noise_regressors = noise_regressors - pred_noise_regressors
        temp_confounds_df.loc[:, columns_to_denoise] = orth_noise_regressors
        confounds_df = temp_confounds_df

    # Censor the data and confounds
    preprocessed_bold_censored = preprocessed_bold[sample_mask, :]
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

    # Now interpolate the censored, denoised data with cubic spline interpolation
    interpolated_unfiltered_bold = np.zeros(
        (n_volumes, n_voxels),
        dtype=censored_denoised_bold.dtype,
    )
    interpolated_unfiltered_bold[sample_mask, :] = censored_denoised_bold
    interpolated_unfiltered_bold = signal._interpolate_volumes(
        interpolated_unfiltered_bold,
        sample_mask=sample_mask,
        t_r=TR,
    )

    # Now apply the bandpass filter to the interpolated, denoised data
    if low_pass is not None and high_pass is not None:
        # TODO: Replace with nilearn.signal.butterworth once 0.10.1 is released.
        interpolated_filtered_bold = butter_bandpass(
            interpolated_unfiltered_bold.copy(),
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


def _select_first(lst):
    """Select the first element in a list."""
    return lst[0]
