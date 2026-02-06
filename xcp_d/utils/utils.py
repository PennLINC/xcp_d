"""Miscellaneous utility functions for xcp_d."""

import multiprocessing

import nibabel as nb
import numpy as np
from nipype import logging

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger('nipype.utils')


def _get_template_transform(template, from_space, cohort=None):
    """Get a TemplateFlow transform path as a string.

    Parameters
    ----------
    template : :obj:`str`
        Template name to query from TemplateFlow.
    from_space : :obj:`str`
        Source space for the transform, passed via the ``from`` entity.
    cohort : :obj:`str` or None
        Cohort label to include when the target template requires it.

    Returns
    -------
    transform_path : :obj:`str`
        Path to the transform file.

    Raises
    ------
    ValueError
        If the TemplateFlow query cannot be resolved.
    """
    from templateflow.api import get as get_template

    template_kwargs = {'from': from_space}
    if cohort is not None:
        template_kwargs['cohort'] = cohort

    return str(
        get_template(
            template=template,
            mode='image',
            suffix='xfm',
            extension='.h5',
            raise_empty=True,
            **template_kwargs,
        )
    )


def _build_mni_chain(source_space, target_space, source_cohort=None, target_cohort=None):
    """Build a transform chain between supported MNI template spaces.

    Parameters
    ----------
    source_space : :obj:`str`
        The source space for the transform chain.
    target_space : :obj:`str`
        The target space for the transform chain.
    source_cohort : :obj:`str` or None
        Cohort label for MNIInfant source spaces.
    target_cohort : :obj:`str` or None
        Cohort label for MNIInfant target spaces.

    Returns
    -------
    transforms : :obj:`list` of :obj:`str`
        Transform chain from ``source_space`` to ``target_space``.

    Raises
    ------
    ValueError
        If the spaces are not supported or cohort is missing for MNIInfant sources.
    """
    if source_space == target_space:
        return ['identity']

    if target_space == 'MNI152NLin6Asym':
        if source_space == 'MNI152NLin2009cAsym':
            return [_get_template_transform('MNI152NLin6Asym', 'MNI152NLin2009cAsym')]
        if source_space == 'MNIInfant':
            if source_cohort is None:
                raise ValueError('Source cohort is required for MNIInfant transforms.')
            return [_get_template_transform('MNI152NLin6Asym', f'MNIInfant+{source_cohort}')]

    if target_space == 'MNI152NLin2009cAsym':
        if source_space == 'MNI152NLin6Asym':
            return [_get_template_transform('MNI152NLin2009cAsym', 'MNI152NLin6Asym')]
        if source_space == 'MNIInfant':
            if source_cohort is None:
                raise ValueError('Source cohort is required for MNIInfant transforms.')
            return [
                _get_template_transform('MNI152NLin2009cAsym', 'MNI152NLin6Asym'),
                _get_template_transform('MNI152NLin6Asym', f'MNIInfant+{source_cohort}'),
            ]

    if target_space == 'MNIInfant':
        target_to_infant = _get_template_transform(
            'MNIInfant',
            'MNI152NLin6Asym',
            cohort=target_cohort,
        )
        if source_space == 'MNI152NLin6Asym':
            return [target_to_infant]
        if source_space == 'MNI152NLin2009cAsym':
            return [
                target_to_infant,
                _get_template_transform('MNI152NLin6Asym', 'MNI152NLin2009cAsym'),
            ]

    raise ValueError(f'Transform chain not supported: {source_space} -> {target_space}')


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
    from xcp_d.utils.bids import get_entity

    def _format_space_error(space, context=None, needs_cohort=False):
        context_str = f' for {context}' if context else ''
        if needs_cohort:
            return (
                f'Space "{space}" requires a cohort{context_str}. '
                'Please specify the cohort using the "cohort" entity.'
            )

        context_str = f' in {context}' if context else ''
        return f'Space "{space}"{context_str} not supported.'

    def _validate_bold_space(space, cohort, supported_spaces, context=None):
        if space not in supported_spaces:
            raise ValueError(_format_space_error(space, context=context))

        if space == 'MNIInfant' and cohort is None:
            raise ValueError(_format_space_error(space, context=context, needs_cohort=True))

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, 'space')
    bold_cohort = get_entity(bold_file, 'cohort')

    if bold_space in ('native', 'T1w'):
        raise ValueError(f'BOLD space "{bold_space}" not supported.')
    elif f'from-{bold_space}' not in template_to_anat_xfm:
        raise ValueError(
            f'Transform does not match BOLD space: {bold_space} != {template_to_anat_xfm}'
        )

    _validate_bold_space(
        space=bold_space,
        cohort=bold_cohort,
        supported_spaces=('MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'MNIInfant'),
        context=bold_file,
    )

    # Pull out the correct transforms based on bold_file name and string them together.
    xforms_to_T1w = [template_to_anat_xfm]  # used for all spaces except T1w and native
    xforms_to_T1w_invert = [False]
    if bold_space == 'MNI152NLin2009cAsym':
        # Data already in MNI152NLin2009cAsym space.
        xforms_to_MNI = _build_mni_chain('MNI152NLin2009cAsym', 'MNI152NLin2009cAsym')
    elif bold_space == 'MNI152NLin6Asym':
        # MNI152NLin6Asym --> MNI152NLin2009cAsym
        xforms_to_MNI = _build_mni_chain('MNI152NLin6Asym', 'MNI152NLin2009cAsym')
    elif bold_space == 'MNIInfant':
        # MNIInfant --> MNI152NLin2009cAsym
        xforms_to_MNI = _build_mni_chain(
            'MNIInfant',
            'MNI152NLin2009cAsym',
            source_cohort=bold_cohort,
        )
    else:
        raise ValueError(f'Space "{bold_space}" in {bold_file} not supported.')

    xforms_to_MNI_invert = [False] * len(xforms_to_MNI)

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
    from xcp_d.utils.bids import get_entity

    def _format_space_error(space, context=None, needs_cohort=False):
        context_str = f' for {context}' if context else ''
        if needs_cohort:
            return (
                f'Space "{space}" requires a cohort{context_str}. '
                'Please specify the cohort using the "cohort" entity.'
            )

        context_str = f' in {context}' if context else ''
        return f'Space "{space}"{context_str} not supported.'

    def _validate_bold_space(space, cohort, supported_spaces, context=None):
        if space not in supported_spaces:
            raise ValueError(_format_space_error(space, context=context))

        if space == 'MNIInfant' and cohort is None:
            raise ValueError(_format_space_error(space, context=context, needs_cohort=True))

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, 'space')
    bold_cohort = get_entity(bold_file, 'cohort')

    source_cohort = None
    if source_space is None:
        # If a source space is not provided, extract the space of the source file
        # First try tpl because that won't raise an error
        source_space = get_entity(source_file, 'tpl')
        source_cohort = get_entity(source_file, 'cohort')
        if source_space is None:
            # If tpl isn't available, try space.
            # get_entity will raise an error if space isn't there.
            source_space = get_entity(source_file, 'space')
    elif '+' in source_space:
        source_space, source_cohort = source_space.split('+')

    _validate_bold_space(
        space=source_space,
        cohort=source_cohort,
        supported_spaces=('MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'MNIInfant'),
        context=source_file,
    )
    _validate_bold_space(
        space=bold_space,
        cohort=bold_cohort,
        supported_spaces=('MNI152NLin6Asym', 'MNI152NLin2009cAsym', 'MNIInfant'),
        context=bold_file,
    )

    transforms = _build_mni_chain(
        source_space,
        bold_space,
        source_cohort=source_cohort,
        target_cohort=bold_cohort,
    )

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
        preprocessed_bold = standardize_signal(preprocessed_bold, detrend=True, standardize=None)
        if have_confounds:
            confounds_arr = standardize_signal(confounds_arr, detrend=True, standardize=None)

        if have_voxelwise_confounds:
            voxelwise_confounds = [
                standardize_signal(arr, detrend=True, standardize=None)
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
    # Load only the header to get shape and data type information
    bold_img = nb.load(bold_fname)

    # Calculate uncompressed memory size from header information
    # This accounts for compression (e.g., .nii.gz files)
    n_voxels = np.prod(bold_img.shape)
    dtype_size = bold_img.get_data_dtype().itemsize
    uncompressed_size_gb = (n_voxels * dtype_size) / (1024**3)

    n_volumes = bold_img.shape[-1]

    # Estimate memory usage based on uncompressed size
    mem_gbz = {
        'bold': uncompressed_size_gb,
        'volume': uncompressed_size_gb / n_volumes,
    }

    return mem_gbz


def is_number(s):
    """Check if a string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_col(df, col):
    """Get a column from a dataframe, with support for `_hash-<hash>` suffixes.

    Parameters
    ----------
    df : :obj:`pandas.DataFrame`
        The dataframe to get the column from.
    col : :obj:`str`
        The column name to get. The actual column name in the DataFrame may have a `_hash-<hash>`
        suffix.

    Returns
    -------
    :obj:`pandas.Series`
        The column from the dataframe.
    """
    import re

    # Pattern to match the base name with optional hash suffix
    # The hash can contain alphanumeric characters and plus signs
    pattern = f'^{re.escape(col)}(_hash-[0-9a-zA-Z+]+)?$'

    # Find columns that match the pattern
    matching_cols = [c for c in df.columns if re.match(pattern, c)]

    if not matching_cols:
        raise ValueError(
            f'No column found matching pattern "{pattern}" in DataFrame. '
            f'Available columns: {list(df.columns)}'
        )

    # If multiple columns match, prefer the one without hash suffix
    # (exact match first, then hash variants)
    exact_match = [c for c in matching_cols if c == col]
    if exact_match:
        return df[exact_match[0]]
    else:
        # If no exact match, use the first hash variant
        return df[matching_cols[0]]
