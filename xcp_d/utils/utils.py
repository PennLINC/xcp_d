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

    This function is a modified version of Nilearn's signal.clean function, with the following
    changes:

    1.  Use :func:`numpy.linalg.lstsq` to estimate betas, instead of QR decomposition,
        in order to denoise the interpolated data as well.
    2.  Set any leading or trailing high-motion volumes to the closest low-motion volume's values
        instead of disabling extrapolation.
    3.  Return denoised, interpolated data.

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
    confounds_file : :obj:`str` or None
        Path to TSV file containing selected confounds, after dummy volume removal,
        but without any additional censoring.
        May be None, if no denoising should be performed.
    %(temporal_mask)s
    low_pass, high_pass : float
        Low-pass and high-pass thresholds, in Hertz.
        If 0, that bound will be skipped
        (e.g., if low-pass is 0 and high-pass isn't,
        then high-pass filtering will be performed instead of bnadpass filtering).
    filter_order : int
        Filter order.
    %(TR)s

    Returns
    -------
    denoised_interpolated_bold
        Returned as a :obj:`numpy.ndarray` of shape (T, S)

    Notes
    -----
    This step only removes high-motion outliers in this step (not the random volumes for trimming).

    The denoising method is designed to follow recommendations from
    :footcite:t:`lindquist2019modular`.
    The method is largely equivalent to Lindquist et al.'s HPMC with orthogonalization.

    References
    ----------
    .. footbibliography::
    """
    import pandas as pd
    from nilearn.signal import butterworth, standardize_signal

    n_volumes = preprocessed_bold.shape[0]

    # Coerce 0 filter values to None
    low_pass = low_pass if low_pass != 0 else None
    high_pass = high_pass if high_pass != 0 else None

    censoring_df = pd.read_table(temporal_mask)
    if censoring_df.shape[0] != n_volumes:
        raise ValueError(
            f"Temporal mask file has {censoring_df.shape[0]} rows, "
            f"but BOLD data has {n_volumes} volumes."
        )

    # Invert temporal mask, so low-motion volumes are True and high-motion volumes are False.
    sample_mask = ~censoring_df["framewise_displacement"].to_numpy().astype(bool)
    outlier_idx = list(np.where(~sample_mask)[0])

    # Determine which steps to apply
    detrend_and_denoise = bool(confounds_file)
    censor_and_interpolate = bool(outlier_idx)

    if detrend_and_denoise:
        confounds_df = pd.read_table(confounds_file)
        confounds_arr = confounds_df.to_numpy()
        if confounds_arr.shape[0] != n_volumes:
            raise ValueError(
                f"Confounds file has {confounds_arr.shape[0]} rows, "
                f"but BOLD data has {n_volumes} volumes."
            )

    if censor_and_interpolate:
        # Replace high-motion volumes in the BOLD data and confounds with cubic-spline interpolated
        # values.
        preprocessed_bold = _interpolate(arr=preprocessed_bold, sample_mask=sample_mask, TR=TR)
        if detrend_and_denoise:
            confounds_arr = _interpolate(arr=confounds_arr, sample_mask=sample_mask, TR=TR)

    if detrend_and_denoise:
        # Detrend the interpolated data and confounds.
        # This also mean-centers the data and confounds.
        preprocessed_bold = standardize_signal(preprocessed_bold, detrend=True, standardize=False)
        confounds_arr = standardize_signal(confounds_arr, detrend=True, standardize=False)

    if low_pass or high_pass:
        # Now apply the bandpass filter to the interpolated data and confounds
        preprocessed_bold = butterworth(
            signals=preprocessed_bold,
            sampling_rate=1.0 / TR,
            low_pass=low_pass,
            high_pass=high_pass,
            order=filter_order,
            padtype="constant",
            padlen=n_volumes - 1,  # maximum possible padding
        )
        if detrend_and_denoise:
            confounds_arr = butterworth(
                signals=confounds_arr,
                sampling_rate=1.0 / TR,
                low_pass=low_pass,
                high_pass=high_pass,
                order=filter_order,
                padtype="constant",
                padlen=n_volumes - 1,  # maximum possible padding
            )

    if detrend_and_denoise:
        # Censor the data and confounds
        censored_bold = preprocessed_bold[sample_mask, :]
        censored_confounds = confounds_arr[sample_mask, :]

        # Estimate betas using only the censored data
        betas = np.linalg.lstsq(censored_confounds, censored_bold, rcond=None)[0]

        # Denoise the interpolated data.
        # The low-motion volumes of the denoised, interpolated data will be the same as the
        # denoised, censored data.
        preprocessed_bold = preprocessed_bold - np.dot(confounds_arr, betas)

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
    from xcp_d.utils.utils import get_transform

    n_volumes = arr.shape[0]
    time = np.arange(0, n_volumes * TR, TR)
    censored_time = time[sample_mask]
    n_voxels = arr.shape[1]

    interpolated_arr = arr.copy()
    for i_voxel in range(n_voxels):
        voxel_data = arr[:, i_voxel]
        interpolated_voxel_data = get_transform(
            censored_time=censored_time,
            arr=voxel_data[sample_mask],
            uncensored_time=time,
            oversampling_factor=4,
            TR=TR,
        )
        LOGGER.info(f"Interpolated voxel {i_voxel + 1}: {interpolated_voxel_data.shape}.")

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
    norm_fac = Std_H / Std_h
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
