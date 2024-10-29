"""Miscellaneous utility functions for xcp_d."""

import nibabel as nb
import numpy as np
from nipype import logging

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


def check_deps(workflow):
    """Make sure dependencies are present in this system."""
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, "_cmd") and which(node.interface._cmd.split()[0]) is None)
    )


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
    from templateflow.api import get as get_template

    from xcp_d.data import load as load_data
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
        MNIInfant_to_MNI152NLin2009cAsym = str(
            load_data(
                "transform/tpl-MNI152NLin2009cAsym_from-MNIInfant_mode-image_xfm.h5",
            )
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


def get_std2bold_xfms(bold_file, source_file, source_space=None):
    """Obtain transforms to warp atlases from a source space to the same template as the BOLD.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    NOTE: This is a Node function.

    Parameters
    ----------
    bold_file : :obj:`str`
        The preprocessed BOLD file.
    source_file : :obj:`str`
        The source file to warp to the BOLD space.
    source_space : :obj:`str`, optional
        The space of the source file. If None, the space of the source file is inferred and used.

    Returns
    -------
    transforms : list of str
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
    from templateflow.api import get as get_template

    from xcp_d.data import load as load_data
    from xcp_d.utils.bids import get_entity

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, "space")

    if source_space is None:
        # If a source space is not provided, extract the space of the source file
        # First try tpl because that won't raise an error
        source_space = get_entity(source_file, "tpl")
        if source_space is None:
            # If tpl isn't available, try space.
            # get_entity will raise an error if space isn't there.
            source_space = get_entity(source_file, "space")

    if source_space not in ("MNI152NLin6Asym", "MNI152NLin2009cAsym", "MNIInfant"):
        raise ValueError(f"Source space '{source_space}' not supported.")

    if bold_space not in ("MNI152NLin6Asym", "MNI152NLin2009cAsym", "MNIInfant"):
        raise ValueError(f"BOLD space '{bold_space}' not supported.")

    # Load useful inter-template transforms from templateflow and package data
    MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
        get_template(
            template="MNI152NLin2009cAsym",
            mode="image",
            suffix="xfm",
            extension=".h5",
            **{"from": "MNI152NLin6Asym"},
        ),
    )
    MNI152NLin2009cAsym_to_MNI152NLin6Asym = str(
        get_template(
            template="MNI152NLin6Asym",
            mode="image",
            suffix="xfm",
            extension=".h5",
            **{"from": "MNI152NLin2009cAsym"},
        ),
    )
    MNIInfant_to_MNI152NLin2009cAsym = str(
        load_data(
            "transform/tpl-MNIInfant_from-MNI152NLin2009cAsym_mode-image_xfm.h5",
        )
    )
    MNI152NLin2009cAsym_to_MNIInfant = str(
        load_data(
            "transform/tpl-MNI152NLin2009cAsym_from-MNIInfant_mode-image_xfm.h5",
        )
    )

    if bold_space == source_space:
        transforms = ["identity"]

    elif bold_space == "MNI152NLin6Asym":
        if source_space == "MNI152NLin2009cAsym":
            transforms = [MNI152NLin2009cAsym_to_MNI152NLin6Asym]
        elif source_space == "MNIInfant":
            transforms = [
                MNI152NLin2009cAsym_to_MNI152NLin6Asym,
                MNIInfant_to_MNI152NLin2009cAsym,
            ]

    elif bold_space == "MNI152NLin2009cAsym":
        if source_space == "MNI152NLin6Asym":
            transforms = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]
        elif source_space == "MNIInfant":
            transforms = [MNIInfant_to_MNI152NLin2009cAsym]

    elif bold_space == "MNIInfant":
        if source_space == "MNI152NLin6Asym":
            transforms = [
                MNI152NLin2009cAsym_to_MNIInfant,
                MNI152NLin6Asym_to_MNI152NLin2009cAsym,
            ]
        elif source_space == "MNI152NLin2009cAsym":
            transforms = [MNI152NLin2009cAsym_to_MNIInfant]

    return transforms


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
    confounds,
    voxelwise_confounds,
    sample_mask,
    low_pass,
    high_pass,
    filter_order,
    TR,
):
    """Denoise an array with Nilearn.

    This function does the following:

    1.  Interpolate high-motion volumes in the BOLD data and confounds.
    2.  Detrend interpolated BOLD and confounds.
        -   Only done if denoising is requested.
        -   This also mean-centers the data.
    3.  Bandpass filter the interpolated data and confounds.
    4.  Censor the data and confounds.
    5.  Estimate betas using only the low-motion volumes.
    6.  Apply the betas to denoise the interpolated BOLD data. This is re-censored in a later step.

    Parameters
    ----------
    preprocessed_bold : :obj:`numpy.ndarray` of shape (T, S)
        Preprocessed BOLD data, after dummy volume removal,
        but without any additional censoring.
    confounds : :obj:`pandas.DataFrame` of shape (T, C1) or None
        DataFrame containing selected confounds, after dummy volume removal,
        but without any additional censoring.
        May be None, if no denoising should be performed.
    voxelwise_confounds : :obj:`list` with of :obj:`numpy.ndarray` of shape (T, S) or None
        Voxelwise confounds after dummy volume removal, but without any additional censoring.
        Will typically be None, as voxelwise regressors are rare.
        A list of C2 arrays, where C2 is the number of voxelwise regressors.
    sample_mask : :obj:`numpy.ndarray` of shape (T,)
        Low-motion volumes are True and high-motion volumes are False.
    low_pass, high_pass : :obj:`float`
        Low-pass and high-pass thresholds, in Hertz.
        If 0, that bound will be skipped
        (e.g., if low-pass is 0 and high-pass isn't,
        then high-pass filtering will be performed instead of bnadpass filtering).
    filter_order : :obj:`int`
        Filter order.
    %(TR)s

    Returns
    -------
    denoised_interpolated_bold : :obj:`numpy.ndarray` of shape (T, S)
        The denoised, interpolated data.

    Notes
    -----
    This step only removes high-motion outliers (not the random volumes for trimming).

    The denoising method is designed to follow recommendations from
    :footcite:t:`lindquist2019modular`.
    The method is largely equivalent to Lindquist et al.'s HPMC with orthogonalization.

    This function is a modified version of Nilearn's :func:`~nilearn.signal.clean` function,
    with the following changes:

    1.  Use :func:`numpy.linalg.lstsq` to estimate betas, instead of QR decomposition,
        in order to denoise the interpolated data as well.
    2.  Set any leading or trailing high-motion volumes to the closest low-motion volume's values
        instead of disabling extrapolation.
    3.  Return denoised, interpolated data.

    References
    ----------
    .. footbibliography::
    """
    from nilearn.signal import butterworth, standardize_signal

    # Don't want to modify the input arrays
    preprocessed_bold = preprocessed_bold.copy()

    n_volumes = preprocessed_bold.shape[0]
    n_voxels = preprocessed_bold.shape[1]

    # Coerce 0 filter values to None
    low_pass = low_pass if low_pass != 0 else None
    high_pass = high_pass if high_pass != 0 else None

    outlier_idx = list(np.where(~sample_mask)[0])

    # Determine which steps to apply
    have_confounds = confounds is not None
    have_voxelwise_confounds = voxelwise_confounds is not None
    detrend_and_denoise = have_confounds or have_voxelwise_confounds
    censor_and_interpolate = bool(outlier_idx)

    if detrend_and_denoise:
        if have_confounds:
            confounds_arr = confounds.to_numpy().copy()

        if have_voxelwise_confounds:
            voxelwise_confounds = [arr.copy() for arr in voxelwise_confounds]

    if censor_and_interpolate:
        # Replace high-motion volumes in the BOLD data and confounds with interpolated values.
        preprocessed_bold = _interpolate(arr=preprocessed_bold, sample_mask=sample_mask, TR=TR)
        if detrend_and_denoise:
            if have_confounds:
                confounds_arr = _interpolate(arr=confounds_arr, sample_mask=sample_mask, TR=TR)

            if have_voxelwise_confounds:
                voxelwise_confounds = [
                    _interpolate(arr=arr, sample_mask=sample_mask, TR=TR)
                    for arr in voxelwise_confounds
                ]

    if detrend_and_denoise:
        # Detrend the interpolated data and confounds.
        # This also mean-centers the data and confounds.
        preprocessed_bold = standardize_signal(preprocessed_bold, detrend=True, standardize=False)
        if have_confounds:
            confounds_arr = standardize_signal(confounds_arr, detrend=True, standardize=False)

        if have_voxelwise_confounds:
            voxelwise_confounds = [
                standardize_signal(arr, detrend=True, standardize=False)
                for arr in voxelwise_confounds
            ]

    if low_pass or high_pass:
        # Now apply the bandpass filter to the interpolated data and confounds
        butterworth_kwargs = {
            "sampling_rate": 1.0 / TR,
            "low_pass": low_pass,
            "high_pass": high_pass,
            "order": filter_order,
            "padtype": "constant",
            "padlen": n_volumes - 1,  # maximum possible padding
        }
        preprocessed_bold = butterworth(signals=preprocessed_bold, **butterworth_kwargs)
        if detrend_and_denoise:
            if have_confounds:
                confounds_arr = butterworth(signals=confounds_arr, **butterworth_kwargs)

            if have_voxelwise_confounds:
                voxelwise_confounds = [
                    butterworth(signals=arr, **butterworth_kwargs) for arr in voxelwise_confounds
                ]

    if detrend_and_denoise:
        # Censor the data and confounds
        censored_bold = preprocessed_bold[sample_mask, :]

        if have_confounds and not voxelwise_confounds:
            # Estimate betas using only the censored data
            censored_confounds = confounds_arr[sample_mask, :]
            betas = np.linalg.lstsq(censored_confounds, censored_bold, rcond=None)[0]

            # Denoise the interpolated data.
            # The low-motion volumes of the denoised, interpolated data will be the same as the
            # denoised, censored data.
            preprocessed_bold = preprocessed_bold - np.dot(confounds_arr, betas)
        else:
            # Loop over voxels
            for i_voxel in range(n_voxels):
                design_matrix = []
                if have_confounds:
                    design_matrix.append(confounds_arr.copy())

                for voxelwise_arr in voxelwise_confounds:
                    temp_voxelwise = voxelwise_arr[:, i_voxel]
                    design_matrix.append(temp_voxelwise[:, None])

                # Estimate betas using only the censored data
                design_matrix = np.hstack(design_matrix)
                censored_design_matrix = design_matrix[sample_mask, :]
                betas = np.linalg.lstsq(
                    censored_design_matrix,
                    censored_bold[:, i_voxel],
                    rcond=None,
                )[0]

                # Denoise the interpolated data.
                # The low-motion volumes of the denoised, interpolated data will be the same as the
                # denoised, censored data.
                preprocessed_bold[:, i_voxel] = preprocessed_bold[:, i_voxel] - np.dot(
                    design_matrix, betas
                )

    return preprocessed_bold


def _interpolate(*, arr, sample_mask, TR):
    """Replace high-motion volumes with a Lomb-Scargle periodogram-based interpolation method.

    This function applies Nilearn's :func:`~nilearn.signal._interpolate_volumes` function,
    followed by an extra step that replaces extrapolated, high-motion values at the beginning and
    end of the run with the closest low-motion volume's data.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray` of shape (T, S)
        The data to interpolate.
    sample_mask : :obj:`numpy.ndarray` of shape (T,)
        The sample mask. True for low-motion volumes, False for high-motion volumes.
    TR : float
        The repetition time.

    Returns
    -------
    interpolated_arr : :obj:`numpy.ndarray` of shape (T, S)
        The interpolated data.

    Notes
    -----
    This function won't work if sample_mask is all zeros, but that should never happen.

    The function uses the least squares spectral analysis method described in
    :footcite:t:`power_fd_dvars`, which in turn is based on the method described in
    :footcite:t:`mathias2004algorithms`.

    References
    ----------
    .. footbibliography::
    """
    from tqdm import trange

    from xcp_d.utils.utils import get_transform

    n_volumes = arr.shape[0]
    time = np.arange(0, n_volumes * TR, TR)
    censored_time = time[sample_mask]
    n_voxels = arr.shape[1]

    interpolated_arr = arr.copy()
    for i_voxel in trange(n_voxels, desc="Interpolating high-motion volumes"):
        voxel_data = arr[:, i_voxel]
        interpolated_voxel_data = get_transform(
            censored_time=censored_time,
            arr=voxel_data[sample_mask],
            uncensored_time=time,
            oversampling_factor=4,
            TR=TR,
        )

        # Replace high-motion volumes in interpolated array with the modified data
        interpolated_arr[~sample_mask, i_voxel] = interpolated_voxel_data[~sample_mask, 0]

    return interpolated_arr


def get_transform(*, censored_time, arr, uncensored_time, oversampling_factor, TR):
    """Interpolate high-motion volumes in a time series using least squares spectral analysis.

    Parameters
    ----------
    censored_time : ndarray of shape (C,)
        Time points for which observations are present.
        C = number of low-motion time points
    arr : ndarray of shape (C, S)
        Observations in columns. The number of rows equals the number of time points.
        C = number of low-motion time points
        S = number of voxels
    uncensored_time : ndarray of shape (T,)
        Time points for which to reconstruct the original time series.
        T = total number of time points
    oversampling_factor : int
        Oversampling frequency, generally >= 4.

    Returns
    -------
    reconstructed_arr : ndarray of shape (T, S)
        The reconstructed time series.
        T = number of time points in uncensored_time
        S = number of voxels in arr

    Notes
    -----
    This function is translated from Anish Mitra's MATLAB function ``getTransform``,
    available at https://www.jonathanpower.net/2014-ni-motion-2.html.

    The function implements the least squares spectral analysis method described in
    :footcite:t:`power_fd_dvars`, which in turn is based on the method described in
    :footcite:t:`mathias2004algorithms`.

    References
    ----------
    .. footbibliography::
    """
    import warnings

    import numpy as np

    assert arr.ndim == 1
    assert censored_time.ndim == 1
    assert uncensored_time.ndim == 1
    assert arr.shape[0] == censored_time.shape[0]
    assert uncensored_time.shape[0] > censored_time.shape[0]

    arr = arr[:, None]
    fs = 1 / TR
    n_volumes = arr.shape[0]  # Number of time points in censored array
    n_voxels = arr.shape[1]  # Number of voxels
    time_span = np.max(censored_time) - np.min(censored_time)  # Total time span
    n_oversampled_timepoints = int((time_span / TR) * oversampling_factor)

    # calculate sampling frequencies
    max_freq = 0.5 * fs
    frequencies_hz = np.linspace(
        1 / (time_span * oversampling_factor),
        max_freq,
        n_oversampled_timepoints,
    )

    # angular frequencies and constant offsets
    frequencies_angular = 2 * np.pi * frequencies_hz
    offsets = np.arctan2(
        np.sum(np.sin(2 * np.dot(frequencies_angular[:, None], censored_time[None, :])), axis=1),
        np.sum(np.cos(2 * np.dot(frequencies_angular[:, None], censored_time[None, :])), axis=1),
    ) / (2 * frequencies_angular)

    # spectral power sin and cosine terms
    spectral_power = np.dot(frequencies_angular[:, None], censored_time[None, :]) - (
        frequencies_angular[:, None] * offsets[:, None]
    )
    cterm = np.cos(spectral_power)
    sterm = np.sin(spectral_power)

    D = arr.copy()
    D = D.reshape((1, n_volumes, n_voxels))

    # This calculation is done by separately for the numerator, denominator, and the division
    cos_mult = cterm[:, :, None] * D
    numerator = np.sum(cos_mult, axis=1)
    denominator = np.sum(cterm**2, axis=1)[:, None]
    power_cos = numerator / denominator

    # Repeat the above for Sine term
    sin_mult = sterm[:, :, None] * D
    numerator = np.sum(sin_mult, axis=1)
    denominator = np.sum(sterm**2, axis=1)[:, None]
    power_sin = numerator / denominator

    # The inverse function to re-construct the original time series
    T_rep = np.repeat(uncensored_time[None, :], repeats=len(frequencies_hz), axis=0)[:, :, None]
    prod = T_rep * frequencies_angular[:, None, None]
    sin_t = np.sin(prod)
    cos_t = np.cos(prod)
    sw_p = sin_t * power_sin[:, None, :]
    cw_p = cos_t * power_cos[:, None, :]
    S = np.sum(sw_p, axis=0)
    C = np.sum(cw_p, axis=0)
    reconstructed_arr = C + S
    reconstructed_arr = reconstructed_arr.reshape((len(uncensored_time), arr.shape[1]))

    # Normalize the reconstructed spectrum, needed when oversampling_factor > 1
    Std_H = np.std(reconstructed_arr, axis=0)
    Std_h = np.std(arr, axis=0)
    with warnings.filterwarnings("error") as w:
        try:
            norm_fac = Std_H / Std_h
        except RuntimeWarning:
            raise ValueError(arr)

    reconstructed_arr = reconstructed_arr / norm_fac[None, :]

    return reconstructed_arr


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


def _transpose_lol(lol):
    """Transpose list of lists."""
    return list(map(list, zip(*lol)))


def _create_mem_gb(bold_fname):
    import os

    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gbz = {
        "derivative": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "timeseries": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    if mem_gbz["timeseries"] < 4.0:
        mem_gbz["timeseries"] = 6.0
        mem_gbz["resampled"] = 2
    elif mem_gbz["timeseries"] > 8.0:
        mem_gbz["timeseries"] = 8.0
        mem_gbz["resampled"] = 3

    return mem_gbz
