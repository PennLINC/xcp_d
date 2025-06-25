"""Miscellaneous utility functions for xcp_d."""

import multiprocessing

import nibabel as nb
import numpy as np
from nipype import logging

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger('nipype.utils')


def check_deps(workflow):
    """Make sure dependencies are present in this system."""
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (hasattr(node.interface, '_cmd') and which(node.interface._cmd.split()[0]) is None)
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
    bold_space = get_entity(bold_file, 'space')

    if bold_space in ('native', 'T1w'):
        base_std_space = get_entity(template_to_anat_xfm, 'from')
        raise ValueError(f"BOLD space '{bold_space}' not supported.")
    elif f'from-{bold_space}' not in template_to_anat_xfm:
        raise ValueError(
            f'Transform does not match BOLD space: {bold_space} != {template_to_anat_xfm}'
        )

    # Pull out the correct transforms based on bold_file name and string them together.
    xforms_to_T1w = [template_to_anat_xfm]  # used for all spaces except T1w and native
    xforms_to_T1w_invert = [False]
    if bold_space == 'MNI152NLin2009cAsym':
        # Data already in MNI152NLin2009cAsym space.
        xforms_to_MNI = ['identity']
        xforms_to_MNI_invert = [False]

    elif bold_space == 'MNI152NLin6Asym':
        # MNI152NLin6Asym --> MNI152NLin2009cAsym
        MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
            get_template(
                template='MNI152NLin2009cAsym',
                mode='image',
                suffix='xfm',
                extension='.h5',
                **{'from': 'MNI152NLin6Asym'},
            ),
        )
        xforms_to_MNI = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]
        xforms_to_MNI_invert = [False]

    elif bold_space == 'MNIInfant':
        # MNIInfant --> MNI152NLin2009cAsym
        MNIInfant_to_MNI152NLin2009cAsym = str(
            load_data(
                'transform/tpl-MNI152NLin2009cAsym_from-MNIInfant_mode-image_xfm.h5',
            )
        )
        xforms_to_MNI = [MNIInfant_to_MNI152NLin2009cAsym]
        xforms_to_MNI_invert = [False]

    elif bold_space == 'T1w':
        # T1w --> ?? (extract from template_to_anat_xfm) --> MNI152NLin2009cAsym
        # Should not be reachable, since xcpd doesn't support T1w-space BOLD inputs
        if base_std_space != 'MNI152NLin2009cAsym':
            std_to_mni_xfm = str(
                get_template(
                    template='MNI152NLin2009cAsym',
                    mode='image',
                    suffix='xfm',
                    extension='.h5',
                    **{'from': base_std_space},
                ),
            )
            xforms_to_MNI = [std_to_mni_xfm, template_to_anat_xfm]
            xforms_to_MNI_invert = [False, True]
        else:
            xforms_to_MNI = [template_to_anat_xfm]
            xforms_to_MNI_invert = [True]

        xforms_to_T1w = ['identity']
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
    bold_space = get_entity(bold_file, 'space')

    if source_space is None:
        # If a source space is not provided, extract the space of the source file
        # First try tpl because that won't raise an error
        source_space = get_entity(source_file, 'tpl')
        if source_space is None:
            # If tpl isn't available, try space.
            # get_entity will raise an error if space isn't there.
            source_space = get_entity(source_file, 'space')

    if source_space not in ('MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'MNIInfant'):
        raise ValueError(f"Source space '{source_space}' not supported.")

    if bold_space not in ('MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'MNIInfant'):
        raise ValueError(f"BOLD space '{bold_space}' not supported.")

    # Load useful inter-template transforms from templateflow and package data
    MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
        get_template(
            template='MNI152NLin2009cAsym',
            mode='image',
            suffix='xfm',
            extension='.h5',
            **{'from': 'MNI152NLin6Asym'},
        ),
    )
    MNI152NLin2009cAsym_to_MNI152NLin6Asym = str(
        get_template(
            template='MNI152NLin6Asym',
            mode='image',
            suffix='xfm',
            extension='.h5',
            **{'from': 'MNI152NLin2009cAsym'},
        ),
    )
    MNIInfant_to_MNI152NLin2009cAsym = str(
        load_data(
            'transform/tpl-MNIInfant_from-MNI152NLin2009cAsym_mode-image_xfm.h5',
        )
    )
    MNI152NLin2009cAsym_to_MNIInfant = str(
        load_data(
            'transform/tpl-MNI152NLin2009cAsym_from-MNIInfant_mode-image_xfm.h5',
        )
    )

    if bold_space == source_space:
        transforms = ['identity']

    elif bold_space == 'MNI152NLin6Asym':
        if source_space == 'MNI152NLin2009cAsym':
            transforms = [MNI152NLin2009cAsym_to_MNI152NLin6Asym]
        elif source_space == 'MNIInfant':
            transforms = [
                MNI152NLin2009cAsym_to_MNI152NLin6Asym,
                MNIInfant_to_MNI152NLin2009cAsym,
            ]

    elif bold_space == 'MNI152NLin2009cAsym':
        if source_space == 'MNI152NLin6Asym':
            transforms = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]
        elif source_space == 'MNIInfant':
            transforms = [MNIInfant_to_MNI152NLin2009cAsym]

    elif bold_space == 'MNIInfant':
        if source_space == 'MNI152NLin6Asym':
            transforms = [
                MNI152NLin2009cAsym_to_MNIInfant,
                MNI152NLin6Asym_to_MNI152NLin2009cAsym,
            ]
        elif source_space == 'MNI152NLin2009cAsym':
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
def estimate_brain_radius(mask_file, head_radius='auto'):
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
    if head_radius == 'auto':
        mask_img = nb.load(mask_file)
        mask_data = mask_img.get_fdata()
        n_voxels = np.sum(mask_data)
        voxel_size = np.prod(mask_img.header.get_zooms())
        volume = n_voxels * voxel_size

        brain_radius = ((3 * volume) / (4 * np.pi)) ** (1 / 3)

        LOGGER.info(f'Brain radius estimated at {brain_radius} mm.')

    else:
        brain_radius = head_radius

    return brain_radius


def denoise_with_nilearn(
    preprocessed_bold,
    confounds,
    voxelwise_confounds,
    sample_mask,
    low_pass,
    high_pass,
    filter_order,
    TR,
    num_threads,
):
    """A wrapper to call _denoise_with_nilearn using multiprocessing"""
    if num_threads < 1:
        raise Exception('num_threads must be a positive integer')
    elif num_threads == 1:
        return _denoise_with_nilearn(
            preprocessed_bold.copy(),
            confounds,
            voxelwise_confounds,
            sample_mask,
            low_pass,
            high_pass,
            filter_order,
            TR,
        )

    # Split the bold data into chunks to br processed in parallel
    preprocessed_bold_chunks = np.array_split(preprocessed_bold, num_threads, axis=1)
    # This np.array_split works on lists too - we just don't use an axis arg
    if voxelwise_confounds is None:
        voxelwise_confounds_chunks = [None] * num_threads
    else:
        voxelwise_confounds_chunks = np.array_split(voxelwise_confounds, num_threads)

    arg_chunks = [
        (
            preprocessed_bold_chunk,
            confounds,
            voxelwise_confounds_chunk,
            sample_mask,
            low_pass,
            high_pass,
            filter_order,
            TR,
        )
        for preprocessed_bold_chunk, voxelwise_confounds_chunk in zip(
            preprocessed_bold_chunks,
            voxelwise_confounds_chunks,
            strict=True,
        )
    ]

    with multiprocessing.Pool(processes=num_threads) as pool:
        results = pool.starmap(_denoise_with_nilearn, arg_chunks)

    return np.column_stack(results)


@fill_doc
def _denoise_with_nilearn(
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
            'sampling_rate': 1.0 / TR,
            'low_pass': low_pass,
            'high_pass': high_pass,
            'order': filter_order,
            'padtype': 'constant',
            'padlen': n_volumes - 1,  # maximum possible padding
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
    """Replace high-motion volumes with cubic-spline interpolated values.

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
    """
    from nilearn import signal

    outlier_idx = list(np.where(~sample_mask)[0])
    n_volumes = arr.shape[0]

    interpolated_arr = signal._interpolate_volumes(
        arr,
        sample_mask=sample_mask,
        t_r=TR,
        extrapolate=True,
    )
    # Replace any high-motion volumes at the beginning or end of the run with the closest
    # low-motion volume's data.
    # Use https://stackoverflow.com/a/48106843/2589328 to group consecutive blocks of outliers.
    gaps = [
        [start, end]
        for start, end in zip(outlier_idx, outlier_idx[1:], strict=False)
        if start + 1 < end
    ]
    edges = iter(outlier_idx[:1] + sum(gaps, []) + outlier_idx[-1:])
    consecutive_outliers_idx = list(zip(edges, edges, strict=False))
    first_outliers = consecutive_outliers_idx[0]
    last_outliers = consecutive_outliers_idx[-1]

    # Replace outliers at beginning of run
    if first_outliers[0] == 0:
        LOGGER.warning(
            f'Outlier volumes at beginning of run ({first_outliers[0]}-{first_outliers[1]}) '
            "will be replaced with first non-outlier volume's values."
        )
        interpolated_arr[: first_outliers[1] + 1, :] = interpolated_arr[first_outliers[1] + 1, :]

    # Replace outliers at end of run
    if last_outliers[1] == n_volumes - 1:
        LOGGER.warning(
            f'Outlier volumes at end of run ({last_outliers[0]}-{last_outliers[1]}) '
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
        raise ValueError('Zero-length list provided.')

    lst_str = [str(item) for item in lst]
    if len(lst_str) == 1:
        return lst_str[0]
    elif len(lst_str) == 2:
        return ' and '.join(lst_str)
    else:
        return f'{", ".join(lst_str[:-1])}, and {lst_str[-1]}'


def _transpose_lol(lol):
    """Transpose list of lists."""
    return list(map(list, zip(*lol, strict=False)))


def _create_mem_gb(bold_fname):
    import os

    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gbz = {
        'derivative': bold_size_gb,
        'resampled': bold_size_gb * 4,
        'timeseries': bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    if mem_gbz['timeseries'] < 4.0:
        mem_gbz['timeseries'] = 6.0
        mem_gbz['resampled'] = 2
    elif mem_gbz['timeseries'] > 8.0:
        mem_gbz['timeseries'] = 8.0
        mem_gbz['resampled'] = 3

    return mem_gbz


def is_number(s):
    """Check if a string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def correlate_timeseries(timeseries, temporal_mask):
    """Correlate timeseries stored in a TSV file."""
    import pandas as pd

    timeseries_df = pd.read_table(timeseries)
    node_names = timeseries_df.columns.values
    correlations_dict = correlate_timeseries_xdf(
        arr=timeseries_df.values.T,
        temporal_mask=temporal_mask,
    )
    output_to_check = ['r', 'z', 'var_r', 'var_z']
    for output in output_to_check:
        if output in correlations_dict.keys():
            correlations_dict[output] = pd.DataFrame(
                correlations_dict[output],
                index=node_names,
                columns=node_names,
            )

    return correlations_dict


def correlate_timeseries_xdf(
    arr,
    temporal_mask,
    method='truncate',
    methodparam='adaptive',
    limit_variance=True,
):
    """Correlate timeseries and calculate variance and dof measures using the xDF method.

    Parameters
    ----------
    arr : :obj:`numpy.ndarray` of shape (V, T)
        Time series array to correlate with xDF.
        V = number of features/regions/voxels
        T = number of samples/data points/volumes
    temporal_mask : :obj:`numpy.ndarray` of shape (T,)
        Temporal mask to apply to the time series.
        Outliers are set to 1, non-outliers are set to 0.
        T = number of samples/data points/volumes
    method : {"tukey", "truncate"}, optional
        The method for estimating autocorrelation.
        Default = "truncate".
    methodparam : :obj:`str`, :obj:`int`, or :obj:`float`, optional
        If ``method`` is "truncate", ``methodparam`` must be "adaptive" or an integer.
        If ``method`` is "tukey", ``methodparam`` must be an empty string ("") or a number.
        Default = "adaptive".
    limit_variance : :obj:`bool`, optional
        If an estimate is lower than the theoretical variance of a white noise then it increases
        the estimate up to ``(1-rho^2)^2/n_cols``.
        To disable this "curbing", set limit_variance to False.
        Default = True.

    Returns
    -------
    out : :obj:`dict`
        A dictionary containing the following keys:
        -   "r": IxI array of correlation coefficients.
        -   "z": IxI array of z-transformed correlation coefficients.
        -   "r_var": IxI array of variance of correlation coefficient between corresponding
            elements, with the diagonal set to 0.
        -   "z_var": IxI array of variance of z-transformed correlation coefficient between
            corresponding elements, with the diagonal set to 0.
        -   "varlimit": Theoretical variance under x & y are i.i.d; (1-rho^2)^2.
        -   "varlimit_idx": Index of (i,j) edges of which their variance exceeded the theoretical
            variance.

    Notes
    -----
    Per :footcite:t:`afyouni2019effective`, method="truncate" + methodparam="adaptive" works best.
    """
    from itertools import combinations

    import scipy.signal as ss
    from scipy.linalg import toeplitz

    n_features, n_samples = arr.shape
    time_idx = np.arange(temporal_mask.size)
    scrubbed_frames = time_idx[temporal_mask == 0]
    n_retained_samples = n_samples - scrubbed_frames.size

    arr = arr - np.nanmean(arr, axis=1)[:, None]
    norms = np.nanmean(arr**2, axis=1)[:, None]

    # Mask NaNs with 0s for summation
    if n_retained_samples < n_samples:
        arr[:, scrubbed_frames] = 0

    # Calculate denominator according to acf in R
    list_pairs = list(combinations(np.setdiff1d(np.arange(n_samples), scrubbed_frames), 2))
    n_pairs = [n_retained_samples] + list(
        np.bincount([y - x for (x, y) in list_pairs], minlength=n_samples)[1:]
    )
    n_denom = n_pairs + np.arange(n_samples)

    # Calculate autocorrelation
    ac = (
        np.concatenate(
            [
                ss.correlate(arr[i, :], arr[i, :], method='fft')[None, -n_samples:] / n_denom
                for i in np.arange(n_features)
            ],
            axis=0,
        )
        / norms
    )

    # Calculate cross-correlation
    xc = np.zeros((n_features, n_features, 2 * n_samples - 1))
    triu_idx_x = np.triu_indices(n_features, 1)[0]
    triu_idx_y = np.triu_indices(n_features, 1)[1]

    for i, j in zip(triu_idx_x, triu_idx_y, strict=False):
        xc[i, j, :] = ss.correlate(arr[i, :], arr[j, :], method='fft') / (
            np.sqrt(norms[i] * norms[j])
            * np.concatenate((np.flip(np.delete(n_denom, 0)), n_denom))
        )

    xc = xc + np.transpose(xc, (1, 0, 2))

    # Extract positive and negative cross-correlations
    xc_p = xc[:, :, : (n_samples - 1)]
    xc_p = np.flip(xc_p, axis=2)
    xc_n = xc[:, :, -(n_samples - 1) :]

    # Extract lag-0 correlations
    rho = np.eye(n_features) + xc[:, :, n_samples - 1]

    # Regularize!
    if method.lower() == 'tukey':
        if methodparam == '':
            M = np.sqrt(n_samples)
        else:
            M = methodparam

        LOGGER.info(f'AC regularization: Tukey tapering of M = {int(np.round(M))}')

        ac = tukeytaperme(ac, n_samples - 1, M)
        xc_p = tukeytaperme(xc_p, n_samples - 1, M)
        xc_n = tukeytaperme(xc_n, n_samples - 1, M)

    elif method.lower() == 'truncate':
        if isinstance(methodparam, str):  # Adaptive truncation
            if methodparam.lower() != 'adaptive':
                raise ValueError('methodparam for truncation must be "adaptive" or an integer')

            LOGGER.info('AC regularization: adaptive truncation')

            ac, bp = shrinkme(ac, n_samples)

            for i in np.arange(n_features):
                for j in np.arange(n_features):
                    maxBP = np.max([bp[i], bp[j]])
                    xc_p[i, j, :] = curbtaperme(ac=xc_p[i, j, :], M=maxBP)
                    xc_n[i, j, :] = curbtaperme(ac=xc_n[i, j, :], M=maxBP)

        elif isinstance(methodparam, int):  # Non-adaptive truncation
            LOGGER.info(f'AC regularization: non-adaptive truncation on M = {methodparam}')
            ac = curbtaperme(ac=ac, M=methodparam)
            xc_p = curbtaperme(ac=xc_p, M=methodparam)
            xc_n = curbtaperme(ac=xc_n, M=methodparam)

        else:
            raise ValueError('methodparam for truncation method should be either str or int')
    else:
        raise ValueError('Method parameter must be either "tukey" or "truncate".')

    # Estimate variance (big formula)
    var_hat_rho = np.zeros((n_features, n_features))

    for i, j in zip(triu_idx_x, triu_idx_y, strict=False):
        r = rho[i, j]

        Sigx = toeplitz(ac[i, :])
        Sigy = toeplitz(ac[j, :])

        Sigx = np.delete(Sigx, scrubbed_frames, axis=0)
        Sigx = np.delete(Sigx, scrubbed_frames, axis=1)
        Sigy = np.delete(Sigy, scrubbed_frames, axis=0)
        Sigy = np.delete(Sigy, scrubbed_frames, axis=1)

        Sigxy = np.triu(toeplitz(np.insert(xc_p[i, j, :], 0, r)), k=1) + np.tril(
            toeplitz(np.insert(xc_n[i, j, :], 0, r))
        )
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=0)
        Sigxy = np.delete(Sigxy, scrubbed_frames, axis=1)
        Sigyx = np.transpose(Sigxy)

        var_hat_rho[i, j] = (
            (r**2 / 2) * np.trace(Sigx @ Sigx)
            + (r**2 / 2) * np.trace(Sigy @ Sigy)
            + r**2 * np.trace(Sigyx @ Sigxy)
            + np.trace(Sigxy @ Sigxy)
            + np.trace(Sigx @ Sigy)
            - 2 * r * np.trace(Sigx @ Sigxy)
            - 2 * r * np.trace(Sigy @ Sigyx)
        ) / n_retained_samples**2

    var_hat_rho = var_hat_rho + np.transpose(var_hat_rho)

    # Truncate to theoretical variance
    varlimit = (1 - rho**2) ** 2 / n_retained_samples

    varlimit_idx = np.where(var_hat_rho < varlimit)
    n_var_outliers = varlimit_idx[1].size / 2
    if n_var_outliers > 0 and limit_variance:
        LOGGER.info('Variance truncation is ON.')

        # Assuming that the variance can *only* get larger in presence of autocorrelation.
        var_hat_rho[varlimit_idx] = varlimit[varlimit_idx]

        FGE = (n_features * (n_features - 1)) / 2
        LOGGER.info(
            f'{n_var_outliers} ({str(round((n_var_outliers / FGE) * 100, 3))}%) '
            'edges had variance smaller than the textbook variance!'
        )
    else:
        LOGGER.info('NO truncation to the theoretical variance.')

    # Our turf--------------------------------
    rf = np.arctanh(rho)
    var_z = var_hat_rho / ((1 - rho**2) ** 2)

    # Set diagonal to 0. I guess variance is not meaningful here?
    np.fill_diagonal(var_hat_rho, 0)
    np.fill_diagonal(var_z, 0)

    out = {
        'r': rho,
        'z': rf,
        'var_r': var_hat_rho,
        'var_z': var_z,
        'varlimit': varlimit,
        'varlimit_idx': varlimit_idx,
    }

    return out


def tukeytaperme(ac, T, M):
    """Perform single Tukey tapering for given length of window, M, and initial value, intv.

    Parameters
    ----------
    ac
    T
    M

    Returns
    -------
    tt_ts

    Notes
    -----
    intv should only be used on crosscorrelation matrices.

    SA, Ox, 2018
    """
    if T not in ac.shape:
        raise ValueError('There is something wrong, mate!')

    ac = ac.copy()

    M = int(np.round(M))

    tukeymultiplier = (1 + np.cos(np.arange(1, M) * np.pi / M)) / 2
    tt_ts = np.zeros(ac.shape)

    if ac.ndim == 2:
        LOGGER.debug('The input is 2D.')
        if ac.shape[1] != T:
            ac = ac.T

        n_rows = ac.shape[0]
        tt_ts[:, : M - 1] = np.tile(tukeymultiplier, [n_rows, 1]) * ac[:, : M - 1]

    elif ac.ndim == 3:
        LOGGER.debug('The input is 3D.')

        n_rows = ac.shape[0]
        tt_ts[:, :, : M - 1] = (
            np.tile(
                tukeymultiplier,
                [n_rows, n_rows, 1],
            )
            * ac[:, :, : M - 1]
        )

    elif ac.ndim == 1:
        LOGGER.debug('The input is 1D.')

        tt_ts[: M - 1] = tukeymultiplier * ac[: M - 1]

    return tt_ts


def curbtaperme(ac, M):
    """Curb the autocorrelations, according to Anderson 1984.

    Parameters
    ----------
    ac
    M

    Returns
    -------
    ct_ts

    Notes
    -----
    multi-dimensional, and therefore is fine!
    SA, Ox, 2018
    """
    ac = ac.copy()
    M = int(round(M))
    msk = np.zeros(np.shape(ac))
    if ac.ndim == 2:
        LOGGER.debug('The input is 2D.')
        msk[:, :M] = 1

    elif ac.ndim == 3:
        LOGGER.debug('The input is 3D.')
        msk[:, :, :M] = 1

    elif ac.ndim == 1:
        LOGGER.debug('The input is 1D.')
        msk[:M] = 1

    ct_ts = msk * ac

    return ct_ts


def shrinkme(ac, T):
    """Shrink the *early* bunches of autocorr coefficients beyond the CI.

    Parameters
    ----------
    ac
    T

    Returns
    -------
    masked_ac
    BreakPoint

    Notes
    -----
    Yo! this should be transformed to the matrix form, those fors at the top are bleak!

    SA, Ox, 2018
    """
    ac = ac.copy()

    if np.shape(ac)[1] != T:
        ac = ac.T

    bnd = (np.sqrt(2) * 1.3859) / np.sqrt(T)
    # assumes normality for AC

    n_rows = np.shape(ac)[0]
    msk = np.zeros(np.shape(ac))
    BreakPoint = np.zeros(n_rows)
    for i in np.arange(n_rows):
        # finds the break point -- intercept
        TheFirstFalse = np.where(np.abs(ac[i, :]) < bnd)

        # if you couldn't find a break point, then continue = the row will remain zero
        if np.size(TheFirstFalse) == 0:
            continue
        else:
            BreakPoint_tmp = TheFirstFalse[0][0]

        msk[i, :BreakPoint_tmp] = 1
        BreakPoint[i] = BreakPoint_tmp

    return ac * msk, BreakPoint
