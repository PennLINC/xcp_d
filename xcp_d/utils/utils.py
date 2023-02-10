#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miscellaneous utility functions for xcp_d."""
import glob
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from templateflow.api import get as get_template


def _t12native(fname):
    """Select T1w-to-scanner transform associated with a given BOLD file.

    TODO: Update names and refactor

    Parameters
    ----------
    fname : str
        The BOLD file from which to identify the transform.

    Returns
    -------
    t1w_to_native_xform : str
        Path to the T1w-to-scanner transform.

    Notes
    -----
    Only used in get_segfile, which should be removed ASAP.
    """
    import os

    pth, fname = os.path.split(fname)
    file_prefix = fname.split("space-")[0]
    t1w_to_native_xform = os.path.join(pth, f"{file_prefix}from-T1w_to-scanner_mode-image_xfm.txt")

    if not os.path.isfile(t1w_to_native_xform):
        raise FileNotFoundError(f"File not found: {t1w_to_native_xform}")

    return t1w_to_native_xform


def get_segfile(bold_file):
    """Select the segmentation file associated with a given BOLD file.

    This function identifies the appropriate MNI-space discrete segmentation file for carpet
    plots, then applies the necessary transforms to warp the file into BOLD reference space.
    The warped segmentation file will be written to a temporary file and its path returned.

    Parameters
    ----------
    bold_file : str
        Path to the BOLD file.

    Returns
    -------
    segfile : str
        The associated segmentation file.

    Notes
    -----
    Only used in concatenation code and should be dropped in favor of BIDSLayout methods ASAP.
    """
    from xcp_d.interfaces.ants import ApplyTransforms

    # get transform files
    dd = Path(os.path.dirname(bold_file))
    anatdir = str(dd.parent) + "/anat"

    if Path(anatdir).is_dir():
        mni_to_t1 = glob.glob(anatdir + "/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5")[0]
    else:
        anatdir = str(dd.parent.parent) + "/anat"
        mni_to_t1 = glob.glob(anatdir + "/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5")[0]

    transformfilex = get_std2bold_xforms(
        bold_file=bold_file,
        template_to_t1w=mni_to_t1,
        t1w_to_native=_t12native(bold_file),
    )

    boldref = bold_file.split("desc-preproc_bold.nii.gz")[0] + "boldref.nii.gz"

    segfile = tempfile.mkdtemp() + "segfile.nii.gz"
    carpet = str(
        get_template(
            "MNI152NLin2009cAsym",
            resolution=1,
            desc="carpet",
            suffix="dseg",
            extension=[".nii", ".nii.gz"],
        ),
    )

    # seg_data file to bold space
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = carpet
    at.inputs.reference_image = boldref
    at.inputs.output_image = segfile
    at.inputs.interpolation = "MultiLabel"
    at.inputs.transforms = transformfilex
    os.system(at.cmdline)

    return segfile


def get_bold2std_and_t1w_xforms(bold_file, template_to_t1w, t1w_to_native):
    """Find transform files in reverse order to transform BOLD to MNI152NLin2009cAsym/T1w space.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    template_to_t1w : str
        The MNI-to-T1w transform file.
        The ``from`` field is assumed to be the same space as the BOLD file is in.
        The MNI space could be MNI152NLin2009cAsym, MNI152NLin6Asym, or MNIInfant.
    t1w_to_native : str
        The T1w-to-native space transform file.

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
    Only used for QCReport in init_boldpostprocess_wf.
    QCReport wants MNI-space data in MNI152NLin2009cAsym.
    """
    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    from xcp_d.utils.bids import get_entity

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, "space")

    if bold_space in ("native", "T1w"):
        base_std_space = get_entity(template_to_t1w, "from")
    elif f"from-{bold_space}" not in template_to_t1w:
        raise ValueError(f"Transform does not match BOLD space: {bold_space} != {template_to_t1w}")

    # Pull out the correct transforms based on bold_file name and string them together.
    xforms_to_T1w = [template_to_t1w]  # used for all spaces except T1w and native
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
        # T1w --> ?? (extract from template_to_t1w) --> MNI152NLin2009cAsym
        # Should not be reachable, since xcpd doesn't support T1w-space BOLD inputs
        if base_std_space != "MNI152NLin2009cAsym":
            std_to_mni_xform = str(
                get_template(
                    template="MNI152NLin2009cAsym",
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": base_std_space},
                ),
            )
            xforms_to_MNI = [std_to_mni_xform, template_to_t1w]
            xforms_to_MNI_invert = [False, True]
        else:
            xforms_to_MNI = [template_to_t1w]
            xforms_to_MNI_invert = [True]

        xforms_to_T1w = ["identity"]
        xforms_to_T1w_invert = [False]

    elif bold_space == "native":
        # native (BOLD) --> T1w --> ?? (extract from template_to_t1w) --> MNI152NLin2009cAsym
        # Should not be reachable, since xcpd doesn't support native-space BOLD inputs
        if base_std_space != "MNI152NLin2009cAsym":
            std_to_mni_xform = str(
                get_template(
                    template="MNI152NLin2009cAsym",
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": base_std_space},
                ),
            )
            xforms_to_MNI = [std_to_mni_xform, template_to_t1w, t1w_to_native]
            xforms_to_MNI_invert = [False, True, True]
        else:
            xforms_to_MNI = [template_to_t1w, t1w_to_native]
            xforms_to_MNI_invert = [True, True]

        xforms_to_T1w = [t1w_to_native]
        xforms_to_T1w_invert = [True]

    else:
        raise ValueError(f"Space '{bold_space}' in {bold_file} not supported.")

    return xforms_to_MNI, xforms_to_MNI_invert, xforms_to_T1w, xforms_to_T1w_invert


def get_std2bold_xforms(bold_file, template_to_t1w, t1w_to_native):
    """Obtain transforms to warp atlases from MNI152NLin6Asym to the same space as the BOLD.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    template_to_t1w : str
        The MNI-to-T1w transform file.
        The ``from`` field is assumed to be the same space as the BOLD file is in.
    t1w_to_native : str
        The T1w-to-native space transform file.

    Returns
    -------
    transform_list : list of str
        A list of paths to transform files.

    Notes
    -----
    Used by:

    - get_segfile (to be removed)
    - to resample dseg in init_boldpostprocess_wf for QCReport
    - to warp atlases to the same space as the BOLD data in init_nifti_functional_connectivity_wf
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
        base_std_space = get_entity(template_to_t1w, "from")
    elif f"from-{bold_space}" not in template_to_t1w:
        raise ValueError(f"Transform does not match BOLD space: {bold_space} != {template_to_t1w}")

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
        # NLin6 --> ?? (extract from template_to_t1w) --> T1w (BOLD)
        if base_std_space != "MNI152NLin6Asym":
            mni_to_std_xform = str(
                get_template(
                    template=base_std_space,
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": "MNI152NLin6Asym"},
                ),
            )
            transform_list = [template_to_t1w, mni_to_std_xform]
        else:
            transform_list = [template_to_t1w]

    elif bold_space == "native":
        # The BOLD data are in native space
        # NLin6 --> ?? (extract from template_to_t1w) --> T1w --> native (BOLD)
        if base_std_space != "MNI152NLin6Asym":
            mni_to_std_xform = str(
                get_template(
                    template=base_std_space,
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": "MNI152NLin6Asym"},
                ),
            )
            transform_list = [t1w_to_native, template_to_t1w, mni_to_std_xform]
        else:
            transform_list = [t1w_to_native, template_to_t1w]

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


def estimate_brain_radius(mask_file, head_radius="auto"):
    """Estimate brain radius from binary brain mask file.

    Parameters
    ----------
    mask_file : str
        Binary brain mask file, in nifti format.
    head_radius : float or "auto", optional
        Head radius to use. Either a number, in millimeters, or "auto".
        If set to "auto", the brain radius will be estimated from the mask file.
        Default is "auto".

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
    import nibabel as nb
    import numpy as np
    from nipype import logging

    LOGGER = logging.getLogger("nipype.utils")

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


def denoise_nifti_with_nilearn(
    bold_file,
    mask_file,
    confounds_file,
    censoring_file,
    low_pass,
    high_pass,
    TR,
):
    """Denoise fMRI data with Nilearn.

    Parameters
    ----------
    bold_file : str or niimg
    mask_file : str
    confounds : pandas.DataFrame
    low_pass : float
    high_pass : float
    TR : float
    tmask : str
    """
    import os

    from nilearn import maskers

    from xcp_d.utils.utils import _denoise_with_nilearn

    out_file = os.path.abspath("desc-denoised_bold.nii.gz")

    # Use a NiftiMasker instead of apply_mask to retain TR in the image header.
    # Note that this doesn't use any of the masker's denoising capabilities.
    masker = maskers.NiftiMasker(
        mask_img=mask_file,
        runs=None,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=False,  # non-default
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
    )
    raw_data = masker.fit_transform(bold_file)

    clean_data = _denoise_with_nilearn(
        raw_data=raw_data,
        confounds_file=confounds_file,
        censoring_file=censoring_file,
        low_pass=low_pass,
        high_pass=high_pass,
        TR=TR,
    )

    clean_img = masker.inverse_transform(clean_data)

    clean_img.to_filename(out_file)
    return out_file


def denoise_cifti_with_nilearn(
    bold_file,
    confounds_file,
    censoring_file,
    low_pass,
    high_pass,
    TR,
):
    """Denoise a CIFTI file with Nilearn.

    The CIFTI file must be read into an array before Nilearn can be called.
    """
    import os

    from xcp_d.utils.utils import _denoise_with_nilearn
    from xcp_d.utils.write_save import read_ndata, write_ndata

    out_file = os.path.abspath("desc-denoised_bold.dtseries.nii")

    raw_data = read_ndata(bold_file)

    # Transpose from SxT (xcpd order) to TxS (nilearn order)
    raw_data = raw_data.T

    clean_data = _denoise_with_nilearn(
        raw_data=raw_data,
        confounds_file=confounds_file,
        censoring_file=censoring_file,
        low_pass=low_pass,
        high_pass=high_pass,
        TR=TR,
    )

    # Transpose from TxS (nilearn order) to SxT (xcpd order)
    clean_data = clean_data.T

    write_ndata(clean_data, template=bold_file, filename=out_file, TR=TR)

    return out_file


def _denoise_with_nilearn(
    raw_data,
    confounds_file,
    censoring_file,
    lowpass,
    highpass,
    filter_order,
    TR,
):
    """Denoise an array with Nilearn.

    This step does the following.
    Linearly detrend, but don't mean-center, the BOLD data.
    Regress out confounds from BOLD data.
    Use list of outliers to censor BOLD data during regression.
    Temporally filter BOLD data.
    """
    import pandas as pd
    from nilearn import signal

    n_volumes, n_voxels = raw_data.shape
    confounds_df = pd.read_table(confounds_file)

    signal_columns = [c for c in confounds_df.columns if c.startswith("signal__")]
    if signal_columns:
        warnings.warn(
            "Signal columns detected. "
            "Orthogonalizing nuisance columns w.r.t. the following signal columns: "
            f"{', '.join(signal_columns)}"
        )
        noise_columns = [c for c in confounds_df.columns if not c.startswith("signal__")]
        temp_confounds_df = confounds_df[noise_columns].copy()
        signal_regressors = confounds_df[signal_columns].to_numpy()
        noise_regressors = temp_confounds_df.to_numpy()
        betas = np.linalg.lstsq(signal_regressors, noise_regressors, rcond=None)[0]
        pred_noise_regressors = np.dot(signal_regressors, betas)
        orth_noise_regressors = noise_regressors - pred_noise_regressors
        temp_confounds_df.loc[:, :] = orth_noise_regressors
        confounds_df = temp_confounds_df

    confounds = confounds_df.to_numpy()
    censoring_df = pd.read_table(censoring_file)
    sample_mask = ~censoring_df["framewise_displacement"].to_numpy().astype(bool)

    # Per xcp_d's style, censor the data first
    raw_data_censored = raw_data[sample_mask, :]
    confounds_censored = confounds[sample_mask, :]

    # Then detrend and regress
    clean_data_censored = signal.clean(
        signals=raw_data_censored,
        detrend=True,
        standardize=False,
        sample_mask=sample_mask,
        confounds=confounds_censored,
        standardize_confounds=True,
        filter=None,
        t_r=TR,
        ensure_finite=True,
    )

    # Now interpolate with cubic spline interpolation
    clean_data_interp = np.zeros((n_volumes, n_voxels), dtype=clean_data_censored.dtype)
    clean_data_interp[sample_mask, :] = clean_data_censored
    clean_data_interp = signal._interpolate_volumes(
        clean_data_interp,
        sample_mask=sample_mask,
        t_r=TR,
    )

    # Now filter
    if lowpass is not None and highpass is not None:
        # TODO: Replace with nilearn.signal.butterworth once 0.10.1 is released.
        butter_bandpass(
            clean_data_interp,
            sampling_rate=1 / TR,
            low_pass=lowpass,
            high_pass=highpass,
            order=filter_order // 2,
            padtype="constant",  # constant is similar to zero-padding
            padlen=n_volumes - 1,  # maximum allowed pad length
        )
    else:
        clean_data = clean_data_interp

    return clean_data
