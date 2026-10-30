"""Test functions in xcp_d.utils.utils."""

import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats

from xcp_d.utils import utils


def test_estimate_brain_radius(ds001419_data):
    """Ensure that the brain radius estimation function returns the right value."""
    bold_mask = ds001419_data['brain_mask_file']

    radius = utils.estimate_brain_radius(bold_mask, head_radius='auto')
    assert radius == 77.39268749395897

    radius = utils.estimate_brain_radius(bold_mask, head_radius=50)
    assert radius == 50


def test_denoise_with_nilearn():
    """Test xcp_d.utils.utils.denoise_with_nilearn."""
    high_pass, low_pass, filter_order, TR = 0.01, 0.08, 2, 2

    n_voxels, n_volumes, n_signals, n_confounds = 100, 10000, 2, 5

    data_arr = np.zeros((n_volumes, n_voxels))

    # Create signals and add them to the data
    rng = np.random.default_rng(0)
    signal_timeseries = rng.standard_normal(size=(n_volumes, n_signals))
    signal_timeseries = signal.detrend(signal_timeseries, axis=0)
    signal_timeseries = stats.zscore(signal_timeseries, axis=0)
    signal_weights = rng.standard_normal(size=(n_signals, n_voxels))

    for i_signal in range(n_signals):
        # The first n_signals voxels are only affected by the corresponding signal
        signal_weights[:, i_signal] = 0
        signal_weights[i_signal, i_signal] = 1

    signal_arr = np.dot(signal_timeseries, signal_weights)
    data_arr += signal_arr

    # Check that signals are present in the "raw" data
    sample_mask = np.ones(n_volumes, dtype=bool)
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # Create confounds and add them to the data
    rng = np.random.default_rng(1)
    confound_timeseries = rng.standard_normal(size=(n_volumes, n_confounds))
    confound_timeseries = signal.detrend(confound_timeseries, axis=0)

    # Orthogonalize the confound_timeseries w.r.t. the signal_timeseries
    signal_betas = np.linalg.lstsq(signal_timeseries, confound_timeseries, rcond=None)[0]
    pred_confound_timeseries = np.dot(signal_timeseries, signal_betas)
    confound_timeseries = confound_timeseries - pred_confound_timeseries

    confound_timeseries = stats.zscore(confound_timeseries, axis=0)
    confound_weights = rng.standard_normal(size=(n_confounds, n_voxels))
    for i_confound in range(n_confounds):
        # The first n_confounds + n_signals voxels are only affected by the corresponding confound
        confound_weights[:, i_confound + n_signals] = 0
        confound_weights[i_confound, i_confound + n_signals] = 1

    confound_arr = np.dot(confound_timeseries, confound_weights)
    data_arr += confound_arr
    confounds_df = pd.DataFrame(
        confound_timeseries,
        columns=[f'confound_{i}' for i in range(n_confounds)],
    )

    # Check that signals are present in the "raw" data at this point
    sample_mask = np.ones(n_volumes, dtype=bool)
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # Now add trends
    rng = np.random.default_rng(2)
    trend = np.arange(n_volumes).astype(np.float32)
    trend -= trend.mean()
    trend_weights = rng.standard_normal(size=(1, n_voxels))
    data_arr += np.dot(trend[:, np.newaxis], trend_weights)
    orig_data_arr = data_arr.copy()
    orig_signal_timeseries = signal_timeseries.copy()

    # Check that signals are still present
    sample_mask = np.ones(n_volumes, dtype=bool)
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # Check that censoring doesn't cause any obvious problems
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    _check_signal(data_arr, signal_timeseries, sample_mask, atol=0.01)

    # First, try out filtering without censoring or denoising
    params = {
        'confounds': None,
        'voxelwise_confounds': None,
        'sample_mask': np.ones(n_volumes, dtype=bool),
        'low_pass': low_pass,
        'high_pass': high_pass,
        'filter_order': filter_order,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified

    # Now, no filtering (censoring + denoising + interpolation)
    # With 10000 volumes, censoring shouldn't have much impact on the betas
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    params = {
        'confounds': confounds_df,
        'voxelwise_confounds': None,
        'sample_mask': sample_mask,
        'low_pass': None,
        'high_pass': None,
        'filter_order': 0,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    assert np.array_equal(data_arr, orig_data_arr)
    assert np.array_equal(signal_timeseries, orig_signal_timeseries)
    _check_trend(out_arr, trend, sample_mask)
    _check_confounds(out_arr, confound_timeseries, sample_mask)
    _check_signal(data_arr, signal_timeseries, sample_mask)
    _check_signal(out_arr, signal_timeseries, sample_mask)

    # Denoise without censoring or filtering
    params = {
        'confounds': confounds_df,
        'voxelwise_confounds': None,
        'sample_mask': np.ones(n_volumes, dtype=bool),
        'low_pass': None,
        'high_pass': None,
        'filter_order': 0,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified
    _check_trend(out_arr, trend, sample_mask)
    _check_confounds(out_arr, confound_timeseries, sample_mask)
    _check_signal(data_arr, signal_timeseries, sample_mask)
    _check_signal(out_arr, signal_timeseries, sample_mask)

    # Run without denoising (censoring + interpolation + filtering)
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    params = {
        'confounds': None,
        'voxelwise_confounds': None,
        'sample_mask': sample_mask,
        'low_pass': low_pass,
        'high_pass': high_pass,
        'filter_order': filter_order,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified

    # Run without denoising (censoring + interpolation + filtering): 2 cpus
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[10:20] = False
    sample_mask[150:160] = False
    params = {
        'confounds': None,
        'voxelwise_confounds': None,
        'sample_mask': sample_mask,
        'low_pass': low_pass,
        'high_pass': high_pass,
        'filter_order': filter_order,
        'TR': TR,
        'num_threads': 2,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified

    # signals were retained
    _check_signal(data_arr, signal_timeseries, sample_mask)
    # XXX: I don't like how high the atol is here, but it's not terrible
    filtered_signals = utils.denoise_with_nilearn(preprocessed_bold=signal_timeseries, **params)
    _check_signal(out_arr, filtered_signals, sample_mask, atol=0.06)

    # Ensure that interpolation + filtering doesn't cause problems at beginning/end of scan
    # Create an updated censoring file with outliers at first and last two volumes
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[:2] = False
    sample_mask[-2:] = False
    n_censored_volumes = np.sum(~sample_mask)
    assert n_censored_volumes == 4

    # Run without denoising or filtering (censoring + interpolation only)
    params = {
        'confounds': None,
        'voxelwise_confounds': None,
        'sample_mask': sample_mask,
        'low_pass': None,
        'high_pass': None,
        'filter_order': 0,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)
    assert not np.allclose(out_arr, data_arr)  # data aren't modified
    # The first two volumes should be the same as the third (first non-outlier) volume
    assert np.allclose(out_arr[0, :], out_arr[2, :])
    assert np.allclose(out_arr[1, :], out_arr[2, :])
    assert not np.allclose(out_arr[2, :], out_arr[3, :])
    # The last volume should be the same as the third-to-last (last non-outlier) volume
    assert np.allclose(out_arr[-1, :], out_arr[-3, :])
    assert np.allclose(out_arr[-2, :], out_arr[-3, :])
    assert not np.allclose(out_arr[-3, :], out_arr[-4, :])

    # signals were retained
    filtered_signals = utils.denoise_with_nilearn(preprocessed_bold=signal_timeseries, **params)

    _check_signal(data_arr, signal_timeseries, sample_mask)
    _check_signal(out_arr, filtered_signals, sample_mask)


def test_denoise_with_nilearn_voxelwise():
    """Test xcp_d.utils.utils.denoise_with_nilearn with voxel-wise regressors.

    Just a smoke test.
    """
    high_pass, low_pass, filter_order, TR = 0.01, 0.08, 2, 2
    n_voxels, n_volumes, n_confounds, n_voxelwise_confounds = 1000, 300, 5, 3
    data_arr = np.random.random((n_volumes, n_voxels))
    confounds = np.random.random((n_volumes, n_confounds))
    confounds_df = pd.DataFrame(
        confounds,
        columns=[f'confound_{i}' for i in range(n_confounds)],
    )
    voxelwise_confounds = [
        np.random.random((n_volumes, n_voxels)) for _ in range(n_voxelwise_confounds)
    ]
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[40:60] = False

    # Denoising with bandpass filtering and censoring
    params = {
        'confounds': confounds_df,
        'voxelwise_confounds': voxelwise_confounds,
        'sample_mask': sample_mask,
        'low_pass': low_pass,
        'high_pass': high_pass,
        'filter_order': filter_order,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    # Denoising without bandpass filtering
    params = {
        'confounds': confounds_df,
        'voxelwise_confounds': voxelwise_confounds,
        'sample_mask': sample_mask,
        'low_pass': None,
        'high_pass': None,
        'filter_order': None,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    # Denoising with bandpass filtering but no general confounds
    params = {
        'confounds': None,
        'voxelwise_confounds': voxelwise_confounds,
        'sample_mask': sample_mask,
        'low_pass': low_pass,
        'high_pass': high_pass,
        'filter_order': filter_order,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)

    # Denoising without bandpass filtering or general confounds
    params = {
        'confounds': None,
        'voxelwise_confounds': voxelwise_confounds,
        'sample_mask': sample_mask,
        'low_pass': None,
        'high_pass': None,
        'filter_order': None,
        'TR': TR,
        'num_threads': 1,
    }
    out_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, **params)
    assert out_arr.shape == (n_volumes, n_voxels)


@pytest.mark.parametrize('n_voxelwise_confounds', [1, 3])
@pytest.mark.parametrize('have_confounds', [True, False])
def test_denoise_with_nilearn_voxelwise_parallel(n_voxelwise_confounds, have_confounds):
    """Test that denoising with voxel-wise regressors gives the same result in parallel.

    Each chunk of voxels must be denoised with the voxel-wise regressors for those voxels.
    """
    high_pass, low_pass, filter_order, TR = 0.01, 0.08, 2, 2
    n_voxels, n_volumes, n_confounds = 100, 120, 5
    rng = np.random.default_rng(0)
    voxelwise_confounds = [
        rng.standard_normal((n_volumes, n_voxels)) for _ in range(n_voxelwise_confounds)
    ]
    # Give every voxel a strong voxel-wise confound signal, so skipping or misassigning
    # the voxel-wise regressors changes the result.
    data_arr = rng.standard_normal((n_volumes, n_voxels)) + 10 * sum(voxelwise_confounds)
    confounds_df = None
    if have_confounds:
        confounds_df = pd.DataFrame(
            rng.standard_normal((n_volumes, n_confounds)),
            columns=[f'confound_{i}' for i in range(n_confounds)],
        )

    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[40:50] = False

    params = {
        'confounds': confounds_df,
        'voxelwise_confounds': voxelwise_confounds,
        'sample_mask': sample_mask,
        'low_pass': low_pass,
        'high_pass': high_pass,
        'filter_order': filter_order,
        'TR': TR,
    }
    serial_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, num_threads=1, **params)
    parallel_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        num_threads=3,
        **params,
    )
    assert np.allclose(parallel_arr, serial_arr)

    # The voxel-wise regressors should be removed from every voxel.
    for voxelwise_arr in voxelwise_confounds:
        corrs = [
            np.corrcoef(voxelwise_arr[sample_mask, i], parallel_arr[sample_mask, i])[0, 1]
            for i in range(n_voxels)
        ]
        assert np.max(np.abs(corrs)) < 0.5


@pytest.mark.parametrize('n_confounds', [6, 36])
@pytest.mark.parametrize('bandpass_filter', [True, False])
def test_denoise_with_nilearn_parallel(n_confounds, bandpass_filter):
    """Test that denoising with standard (global) confounds gives the same result in parallel."""
    TR = 2
    n_voxels, n_volumes = 100, 120
    rng = np.random.default_rng(0)
    confounds_df = pd.DataFrame(
        rng.standard_normal((n_volumes, n_confounds)),
        columns=[f'confound_{i}' for i in range(n_confounds)],
    )
    data_arr = rng.standard_normal((n_volumes, n_voxels)) + 100
    data_arr += confounds_df.to_numpy() @ rng.standard_normal((n_confounds, n_voxels))
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[[0, 1, 40, 41, 42, 118, 119]] = False

    params = {
        'confounds': confounds_df,
        'voxelwise_confounds': None,
        'sample_mask': sample_mask,
        'low_pass': 0.08 if bandpass_filter else None,
        'high_pass': 0.01 if bandpass_filter else None,
        'filter_order': 2,
        'TR': TR,
    }
    serial_arr = utils.denoise_with_nilearn(preprocessed_bold=data_arr, num_threads=1, **params)
    parallel_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        num_threads=3,
        **params,
    )
    assert np.allclose(parallel_arr, serial_arr)


def test_denoise_with_nilearn_voxelwise_shape_mismatch():
    """Test that voxel-wise regressors with the wrong shape are rejected."""
    n_voxels, n_volumes = 50, 60
    rng = np.random.default_rng(0)
    data_arr = rng.standard_normal((n_volumes, n_voxels))
    voxelwise_confounds = [rng.standard_normal((n_voxels, n_volumes))]  # S x T, not T x S

    with pytest.raises(ValueError, match='Voxelwise confound 0 has shape'):
        utils.denoise_with_nilearn(
            preprocessed_bold=data_arr,
            confounds=None,
            voxelwise_confounds=voxelwise_confounds,
            sample_mask=np.ones(n_volumes, dtype=bool),
            low_pass=None,
            high_pass=None,
            filter_order=None,
            TR=2,
            num_threads=1,
        )


@pytest.mark.parametrize('num_threads', [1, 3])
def test_denoise_with_nilearn_voxelwise_empty_list(num_threads):
    """Test that an empty list of voxel-wise regressors is treated like no voxel-wise regressors."""
    n_voxels, n_volumes, n_confounds = 30, 60, 3
    rng = np.random.default_rng(0)
    data_arr = rng.standard_normal((n_volumes, n_voxels))
    confounds_df = pd.DataFrame(rng.standard_normal((n_volumes, n_confounds)))
    params = {
        'confounds': confounds_df,
        'sample_mask': np.ones(n_volumes, dtype=bool),
        'low_pass': None,
        'high_pass': None,
        'filter_order': None,
        'TR': 2,
        'num_threads': num_threads,
    }
    empty_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        voxelwise_confounds=[],
        **params,
    )
    none_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        voxelwise_confounds=None,
        **params,
    )
    assert np.allclose(empty_arr, none_arr)


@pytest.mark.parametrize('num_threads', [1, 3])
def test_denoise_with_nilearn_voxelwise_matches_joint_regression(num_threads):
    """Test voxel-wise denoising against an independent least-squares reference.

    Each voxel's design combines the global confounds and that voxel's voxel-wise regressors.
    The global and voxel-wise regressors are correlated and both contribute to the data,
    so leaving either out of the design, or giving a voxel another voxel's regressors,
    changes the result.
    """
    n_voxels, n_volumes, n_confounds, n_voxelwise_confounds = 40, 150, 4, 2
    rng = np.random.default_rng(1)
    global_arr = rng.standard_normal((n_volumes, n_confounds))
    voxelwise_confounds = [
        global_arr @ rng.standard_normal((n_confounds, n_voxels))
        + rng.standard_normal((n_volumes, n_voxels))
        for _ in range(n_voxelwise_confounds)
    ]
    data_arr = (
        100
        + global_arr @ rng.standard_normal((n_confounds, n_voxels))
        + sum(rng.uniform(1, 3, n_voxels) * arr for arr in voxelwise_confounds)
        + rng.standard_normal((n_volumes, n_voxels))
    )

    # No censoring or filtering, so the only preprocessing is linear detrending.
    denoised_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        confounds=pd.DataFrame(global_arr),
        voxelwise_confounds=voxelwise_confounds,
        sample_mask=np.ones(n_volumes, dtype=bool),
        low_pass=None,
        high_pass=None,
        filter_order=None,
        TR=2,
        num_threads=num_threads,
    )

    detrended_data = signal.detrend(data_arr, axis=0)
    detrended_global = signal.detrend(global_arr, axis=0)
    detrended_voxelwise = [signal.detrend(arr, axis=0) for arr in voxelwise_confounds]
    expected_arr = np.empty_like(detrended_data)
    for i_voxel in range(n_voxels):
        design = np.column_stack(
            [detrended_global] + [arr[:, i_voxel] for arr in detrended_voxelwise],
        )
        betas = np.linalg.lstsq(design, detrended_data[:, i_voxel], rcond=None)[0]
        expected_arr[:, i_voxel] = detrended_data[:, i_voxel] - design @ betas

    assert np.allclose(denoised_arr, expected_arr)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_interpolate_chunks_match_one_spline(dtype):
    """Test that interpolating in chunks of voxels matches one spline over all voxels."""
    from scipy.interpolate import CubicSpline

    n_volumes, TR = 100, 2
    # Enough voxels for several chunks, with a partial chunk at the end
    n_voxels = 3 * ((2**20) // n_volumes) + 7
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n_volumes, n_voxels)).astype(dtype)
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[[0, 1, 30, 31, 32, 60, 98, 99]] = False

    frame_times = np.arange(n_volumes) * TR
    expected = arr.copy()
    spline = CubicSpline(frame_times[sample_mask], arr[sample_mask, :], extrapolate=True)
    expected[~sample_mask, :] = spline(frame_times)[~sample_mask, :]
    # Leading and trailing outliers take the closest low-motion volume's values
    expected[:2, :] = expected[2, :]
    expected[98:, :] = expected[97, :]

    interpolated = utils._interpolate(arr=arr, sample_mask=sample_mask, TR=TR)
    assert interpolated is arr
    assert interpolated.dtype == dtype
    assert np.allclose(interpolated, expected, rtol=1e-6 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize('n_voxelwise_confounds', [0, 1])
def test_denoise_with_nilearn_float32(n_voxelwise_confounds):
    """Test that float32 data are denoised in float64 and returned as float32.

    The result must be exactly the result for the same values in float64, rounded to float32.
    There are enough voxels for several chunks, with a partial chunk at the end.
    """
    n_volumes = 100
    n_voxels = 2 * ((2**20) // n_volumes) + 7
    rng = np.random.default_rng(0)
    voxelwise_confounds = [
        rng.standard_normal((n_volumes, n_voxels)).astype(np.float32)
        for _ in range(n_voxelwise_confounds)
    ]
    data_arr = (
        1000 + 10 * rng.standard_normal((n_volumes, n_voxels)) + 10 * sum(voxelwise_confounds)
    ).astype(np.float32)
    original_arr = data_arr.copy()
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[[0, 40, 41, 42, 99]] = False
    params = {
        'confounds': pd.DataFrame(rng.standard_normal((n_volumes, 3))),
        'sample_mask': sample_mask,
        'low_pass': 0.08,
        'high_pass': 0.01,
        'filter_order': 2,
        'TR': 2,
        'num_threads': 1,
    }

    denoised_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr,
        voxelwise_confounds=voxelwise_confounds or None,
        **params,
    )
    assert denoised_arr.dtype == np.float32
    assert np.array_equal(data_arr, original_arr)

    expected_arr = utils.denoise_with_nilearn(
        preprocessed_bold=data_arr.astype(np.float64),
        voxelwise_confounds=[arr.astype(np.float64) for arr in voxelwise_confounds] or None,
        **params,
    )
    assert np.array_equal(denoised_arr, expected_arr)


def test_create_mem_gb_float32(tmp_path):
    """Test that _create_mem_gb sizes the data as float32, whatever the data type on disk."""
    import nibabel as nb

    shape = (10, 20, 30, 40)
    bold_file = str(tmp_path / 'bold.nii.gz')
    nb.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4)).to_filename(bold_file)
    assert nb.load(bold_file).get_data_dtype() == np.int16

    mem_gb = utils._create_mem_gb(bold_file)
    assert mem_gb['bold'] == np.prod(shape) * 4 / (1024**3)
    assert mem_gb['volume'] == mem_gb['bold'] / shape[3]


def test_estimate_denoise_mem_gb(tmp_path):
    """Check the denoising memory estimate against its phases for a small NIfTI run."""
    import nibabel as nb

    shape, n_volumes = (100, 100, 100), 100
    mask_arr = np.zeros(shape, dtype=np.uint8)
    mask_arr[40:60, 40:60, 40:60] = 1
    mask_file = str(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(mask_arr, np.eye(4)).to_filename(mask_file)

    # Only the header of the BOLD file is read, so write the header alone
    header = nb.Nifti1Header()
    header.set_data_shape((*shape, n_volumes))
    header.set_data_dtype(np.float32)
    bold_file = str(tmp_path / 'bold.nii')
    with open(bold_file, 'wb') as fobj:
        header.write_to(fobj)

    gib = 1024**3
    data_gib = 20**3 * n_volumes * 4 / gib
    grid_gib = 100**3 * n_volumes * 4 / gib

    # With only global confounds in one process, loading the BOLD image is the peak
    estimate = utils._estimate_denoise_mem_gb(bold_file=bold_file, mask_file=mask_file)
    assert np.isclose(estimate, 0.25 + 2.3 * grid_gib + data_gib)

    # Three voxel-wise confounds in one process: loading the four images is the peak
    estimate = utils._estimate_denoise_mem_gb(
        bold_file=bold_file,
        mask_file=mask_file,
        n_voxelwise_confounds=3,
    )
    assert np.isclose(estimate, 0.25 + 2.3 * grid_gib + 4 * data_gib)

    # With eight processes, denoising is the peak
    estimate = utils._estimate_denoise_mem_gb(
        bold_file=bold_file,
        mask_file=mask_file,
        n_voxelwise_confounds=3,
        n_procs=8,
    )
    assert np.isclose(estimate, 0.25 + 7 * data_gib + 0.4 * 8)


def test_estimate_denoise_mem_gb_cifti(tmp_path):
    """Check the denoising memory estimate for a small CIFTI run, which is read as float32."""
    import nibabel as nb

    n_volumes = 50
    brain_models = nb.cifti2.BrainModelAxis.from_mask(
        np.ones((4, 4, 4), dtype=bool),
        affine=np.eye(4),
        name='thalamus_left',
    )
    series = nb.cifti2.SeriesAxis(start=0, step=2, size=n_volumes)
    bold_file = str(tmp_path / 'bold.dtseries.nii')
    bold_arr = np.zeros((n_volumes, len(brain_models)), dtype=np.float32)
    nb.Cifti2Image(bold_arr, header=(series, brain_models)).to_filename(bold_file)

    data_gib = n_volumes * len(brain_models) * 4 / (1024**3)

    # The data are tiny, so the fixed cost of writing is the peak in one process
    estimate = utils._estimate_denoise_mem_gb(bold_file=bold_file)
    assert np.isclose(estimate, 0.25 + data_gib + 0.25)

    # With four processes and only global confounds, denoising is the peak
    estimate = utils._estimate_denoise_mem_gb(bold_file=bold_file, n_procs=4)
    assert np.isclose(estimate, 0.25 + 4 * data_gib + 0.4 * 4)


@pytest.mark.parametrize('have_confounds', [True, False])
def test_regress_voxelwise_confounds_rank_deficient(have_confounds):
    """Test the two-stage regression against the combined design with degenerate confounds.

    Voxel-wise confounds that are numerically zero after the global confounds are regressed out
    must be dropped, as lstsq on the combined design would, rather than fit to rounding noise.
    """
    n_volumes, n_voxels, n_confounds = 120, 6, 3
    rng = np.random.default_rng(2)
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[[10, 11, 12, 60, 61]] = False

    confounds_arr = rng.standard_normal((n_volumes, n_confounds)) if have_confounds else None
    voxelwise_confounds = [rng.standard_normal((n_volumes, n_voxels)) for _ in range(2)]
    if have_confounds:
        # Voxel 0: both voxel-wise confounds lie in the span of the global confounds
        voxelwise_confounds[0][:, 0] = 2.5 * confounds_arr[:, 0]
        voxelwise_confounds[1][:, 0] = confounds_arr @ np.array([1.0, -0.5, 2.0])
        # Voxel 1: only the first one does
        voxelwise_confounds[0][:, 1] = confounds_arr @ np.array([0.3, 1.0, 1.5])
    else:
        # Voxel 0: both voxel-wise confounds are zero
        voxelwise_confounds[0][:, 0] = 0
        voxelwise_confounds[1][:, 0] = 0
        # Voxel 1: the second one duplicates the first one
        voxelwise_confounds[1][:, 1] = voxelwise_confounds[0][:, 1]

    bold_arr = rng.standard_normal((n_volumes, n_voxels)) + sum(voxelwise_confounds)
    if have_confounds:
        bold_arr += confounds_arr @ rng.standard_normal((n_confounds, n_voxels))

    expected_arr = np.empty_like(bold_arr)
    for i_voxel in range(n_voxels):
        columns = [arr[:, i_voxel] for arr in voxelwise_confounds]
        if have_confounds:
            columns = [confounds_arr, *columns]

        design = np.column_stack(columns)
        betas = np.linalg.lstsq(
            design[sample_mask, :],
            bold_arr[sample_mask, i_voxel],
            rcond=None,
        )[0]
        expected_arr[:, i_voxel] = bold_arr[:, i_voxel] - design @ betas

    denoised_arr = utils._regress_voxelwise_confounds(
        preprocessed_bold=bold_arr.copy(),
        confounds_arr=confounds_arr,
        voxelwise_confounds=[arr.copy() for arr in voxelwise_confounds],
        sample_mask=sample_mask,
    )
    assert np.allclose(denoised_arr, expected_arr, rtol=1e-10, atol=1e-10)


def test_limit_blas_threads():
    """Test that the denoising workers' initializer limits BLAS to one thread."""
    import multiprocessing

    from threadpoolctl import threadpool_info

    with multiprocessing.Pool(
        processes=2,
        initializer=utils._limit_blas_threads,
        initargs=(1,),
    ) as pool:
        worker_info = pool.apply(threadpool_info)

    blas_threads = [info['num_threads'] for info in worker_info if info['user_api'] == 'blas']
    assert blas_threads
    assert all(n_threads == 1 for n_threads in blas_threads)


def test_limit_blas_threads_in_process():
    """Test the BLAS-limiting initializer in the current process.

    Coverage is not collected in pool workers, so this also runs the initializer here.
    The surrounding context restores the test process's BLAS limit afterwards.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2, user_api='blas'):
        utils._limit_blas_threads(1)
        threads_during = _blas_threads()

    assert threads_during
    assert all(n_threads == 1 for n_threads in threads_during)


def _blas_threads():
    from threadpoolctl import threadpool_info

    return [info['num_threads'] for info in threadpool_info() if info['user_api'] == 'blas']


def _denoise_params(n_volumes=40, n_voxels=12):
    rng = np.random.default_rng(0)
    return {
        'preprocessed_bold': rng.standard_normal((n_volumes, n_voxels)),
        'confounds': pd.DataFrame(rng.standard_normal((n_volumes, 2))),
        'voxelwise_confounds': None,
        'sample_mask': np.ones(n_volumes, dtype=bool),
        'low_pass': None,
        'high_pass': None,
        'filter_order': None,
        'TR': 2,
    }


def test_denoise_with_nilearn_serial_limits_blas_threads(monkeypatch):
    """Test that serial denoising runs with one BLAS thread and then restores the limit."""
    from threadpoolctl import threadpool_limits

    threads_during = []

    def _record_threads(preprocessed_bold, *args):
        threads_during.extend(_blas_threads())
        return preprocessed_bold

    monkeypatch.setattr(utils, '_denoise_with_nilearn', _record_threads)
    with threadpool_limits(limits=2, user_api='blas'):
        threads_before = _blas_threads()
        utils.denoise_with_nilearn(num_threads=1, **_denoise_params())
        threads_after = _blas_threads()

    assert threads_during
    assert all(n_threads == 1 for n_threads in threads_during)
    assert threads_after == threads_before


def test_denoise_with_nilearn_parallel_limits_blas_threads(monkeypatch):
    """Test that the denoising pool limits each worker to one BLAS thread.

    The pool is replaced with an in-process stand-in that records how it was created,
    and test_limit_blas_threads checks that the initializer itself works.
    """
    pool_kwargs = {}

    class _InProcessPool:
        def __init__(self, **kwargs):
            pool_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def starmap(self, func, iterable):
            return [func(*args) for args in iterable]

    monkeypatch.setattr(utils.multiprocessing, 'Pool', _InProcessPool)
    params = _denoise_params()
    parallel_arr = utils.denoise_with_nilearn(num_threads=3, **params)

    assert pool_kwargs['processes'] == 3
    assert pool_kwargs['initializer'] is utils._limit_blas_threads
    assert pool_kwargs['initargs'] == (1,)
    assert np.allclose(parallel_arr, utils.denoise_with_nilearn(num_threads=1, **params))


def _check_trend(data, trend, sample_mask, atol=0.01):
    """Ensure that the trend was removed by the denoising process."""
    trend_corr = np.corrcoef(trend[sample_mask], data[sample_mask, :].T)[0, 1:]
    assert all(np.abs(trend_corr) < atol)


def _check_confounds(data, confounds, sample_mask, atol=0.01):
    """Ensure that confounds were removed by the denoising process."""
    for i_confound in range(confounds.shape[1]):
        confound_corr = np.corrcoef(
            confounds[sample_mask, i_confound],
            data[sample_mask, :].T,
        )[0, 1:]
        assert all(np.abs(confound_corr) < atol)


def _check_signal(data, signals, sample_mask, atol=0.001):
    """Ensure that signals were retained by the denoising process."""
    signal_betas = []

    # Add constant and linear trend
    dm = np.hstack(
        (
            signals,
            np.ones((signals.shape[0], 1)),
            np.arange(signals.shape[0])[:, np.newaxis],
        ),
    )
    dm = dm[sample_mask, :]
    for i_signal in range(signals.shape[1]):
        res = np.linalg.lstsq(
            dm,
            data[sample_mask, i_signal][:, np.newaxis],
            rcond=None,
        )
        signal_beta = res[0][i_signal]
        signal_betas.append(signal_beta)

    assert np.allclose(signal_betas, 1.0, atol=atol)


def test_list_to_str():
    """Test the list_to_str function."""
    string = utils.list_to_str(['a'])
    assert string == 'a'

    string = utils.list_to_str(['a', 'b'])
    assert string == 'a and b'

    string = utils.list_to_str(['a', 'b', 'c'])
    assert string == 'a, b, and c'

    with pytest.raises(ValueError, match=r'Zero-length list provided\.'):
        utils.list_to_str([])


def test_get_bold2std_and_t1w_xfms(ds001419_data):
    """Test get_bold2std_and_t1w_xfms."""
    bold_file_nlin6asym = ds001419_data['nifti_file']
    nlin6asym_to_anat_xfm = ds001419_data['template_to_anat_xfm']

    # MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    (
        xforms_to_mni,
        xforms_to_mni_invert,
        xforms_to_t1w,
        xforms_to_t1w_invert,
    ) = utils.get_bold2std_and_t1w_xfms(
        bold_file_nlin6asym,
        nlin6asym_to_anat_xfm,
    )
    assert len(xforms_to_mni) == 1
    assert len(xforms_to_mni_invert) == 1
    assert len(xforms_to_t1w) == 1
    assert len(xforms_to_t1w_invert) == 1

    # MNI152NLin2009cAsym --> MNI152NLin2009cAsym/T1w
    bold_file_nlin2009c = bold_file_nlin6asym.replace(
        'space-MNI152NLin6Asym_',
        'space-MNI152NLin2009cAsym_',
    )
    nlin2009c_to_anat_xfm = nlin6asym_to_anat_xfm.replace(
        'from-MNI152NLin6Asym_',
        'from-MNI152NLin2009cAsym_',
    )
    (
        xforms_to_mni,
        xforms_to_mni_invert,
        xforms_to_t1w,
        xforms_to_t1w_invert,
    ) = utils.get_bold2std_and_t1w_xfms(
        bold_file_nlin2009c,
        nlin2009c_to_anat_xfm,
    )
    assert len(xforms_to_mni) == 1
    assert len(xforms_to_mni_invert) == 1
    assert len(xforms_to_t1w) == 1
    assert len(xforms_to_t1w_invert) == 1

    # MNIInfant cohort-1 --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_infant = bold_file_nlin6asym.replace(
        'space-MNI152NLin6Asym_',
        'space-MNIInfant_cohort-1_',
    )
    infant_to_anat_xfm = nlin6asym_to_anat_xfm.replace(
        'from-MNI152NLin6Asym_',
        'from-MNIInfant+1_',
    )
    (
        xforms_to_mni,
        xforms_to_mni_invert,
        xforms_to_t1w,
        xforms_to_t1w_invert,
    ) = utils.get_bold2std_and_t1w_xfms(
        bold_file_infant,
        infant_to_anat_xfm,
    )
    assert len(xforms_to_mni) == 2
    assert len(xforms_to_mni_invert) == 2
    assert len(xforms_to_t1w) == 1
    assert len(xforms_to_t1w_invert) == 1

    # T1w --> MNI152NLin2009cAsym/T1w
    bold_file_t1w = bold_file_nlin6asym.replace('space-MNI152NLin6Asym_', 'space-T1w_')
    with pytest.raises(ValueError, match=r'BOLD space "T1w" not supported\.'):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_t1w,
            nlin6asym_to_anat_xfm,
        )

    # T1w --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_t1w = bold_file_nlin6asym.replace('space-MNI152NLin6Asym_', 'space-T1w_')
    with pytest.raises(ValueError, match=r'BOLD space "T1w" not supported\.'):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_t1w,
            nlin6asym_to_anat_xfm,
        )

    # native --> MNI152NLin2009cAsym/T1w
    bold_file_native = bold_file_nlin6asym.replace('space-MNI152NLin6Asym_', '')
    with pytest.raises(ValueError, match=r'BOLD space "native" not supported\.'):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_native,
            nlin6asym_to_anat_xfm,
        )

    # native --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w
    bold_file_native = bold_file_nlin6asym.replace('space-MNI152NLin6Asym_', '')
    with pytest.raises(ValueError, match=r'BOLD space "native" not supported\.'):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_native,
            nlin6asym_to_anat_xfm,
        )

    # tofail --> MNI152NLin2009cAsym/T1w
    bold_file_tofail = bold_file_nlin6asym.replace('space-MNI152NLin6Asym_', 'space-tofail_')
    with pytest.raises(ValueError, match='Transform does not match BOLD space'):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_tofail,
            nlin6asym_to_anat_xfm,
        )

    tofail_to_anat_xfm = nlin6asym_to_anat_xfm.replace('from-MNI152NLin6Asym_', 'from-tofail_')
    with pytest.raises(ValueError, match='BOLD space "tofail" not supported'):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_tofail,
            tofail_to_anat_xfm,
        )

    # MNIInfant without cohort --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w (fails)
    bold_file_infant = bold_file_nlin6asym.replace(
        'space-MNI152NLin6Asym_',
        'space-MNIInfant_',
    )
    infant_to_anat_xfm = nlin6asym_to_anat_xfm.replace(
        'from-MNI152NLin6Asym_',
        'from-MNIInfant+1_',
    )
    with pytest.raises(ValueError, match='BOLD cohort is not specified'):
        utils.get_bold2std_and_t1w_xfms(
            bold_file_infant,
            infant_to_anat_xfm,
        )


def test_get_std2bold_xfms(ds001419_data):
    """Test get_std2bold_xfms.

    get_std2bold_xfms finds transforms to go from a source file's space to the BOLD file's space.
    """
    bold_file_nlin6asym = ds001419_data['nifti_file']

    # MNI152NLin6Asym --> MNI152NLin6Asym with source file containing tpl entity
    xforms_to_mni = utils.get_std2bold_xfms(
        bold_file_nlin6asym,
        source_file='tpl-MNI152NLin6Asym_T1w.nii.gz',
        source_space=None,
    )
    assert len(xforms_to_mni) == 1

    # MNI152NLin6Asym --> MNI152NLin6Asym with source file containing space entity
    xforms_to_mni = utils.get_std2bold_xfms(
        bold_file_nlin6asym,
        source_file='space-MNI152NLin6Asym_T1w.nii.gz',
        source_space=None,
    )
    assert len(xforms_to_mni) == 1

    SPACES = [
        ('MNI152NLin6Asym', 'MNI152NLin6Asym', 1),
        ('MNI152NLin6Asym', 'MNI152NLin2009cAsym', 1),
        ('MNI152NLin6Asym', 'MNIInfant+1', 1),
        ('MNI152NLin2009cAsym', 'MNI152NLin2009cAsym', 1),
        ('MNI152NLin2009cAsym', 'MNI152NLin6Asym', 1),
        ('MNI152NLin2009cAsym', 'MNIInfant+1', 2),
        ('MNIInfant_cohort-1', 'MNIInfant+1', 1),
        ('MNIInfant_cohort-1', 'MNI152NLin2009cAsym', 2),
        ('MNIInfant_cohort-1', 'MNI152NLin6Asym', 1),
    ]
    for space_check in SPACES:
        target_space, source_space, n_xforms = space_check
        bold_file_target_space = bold_file_nlin6asym.replace(
            'space-MNI152NLin6Asym_',
            f'space-{target_space}_',
        )
        xforms_to_mni = utils.get_std2bold_xfms(
            bold_file_target_space,
            source_file=None,
            source_space=source_space,
        )
        assert len(xforms_to_mni) == n_xforms

    # Outside of the supported spaces, we expect an error
    # No space or tpl entity in source file
    with pytest.raises(ValueError, match='Unknown space'):
        utils.get_std2bold_xfms(bold_file_nlin6asym, source_file='T1w.nii.gz', source_space=None)

    # MNI152NLin6Asym --> tofail
    bold_file_tofail = bold_file_nlin6asym.replace('space-MNI152NLin6Asym_', 'space-tofail_')
    with pytest.raises(ValueError, match='BOLD space "tofail" not supported'):
        utils.get_std2bold_xfms(bold_file_tofail, source_file=None, source_space='MNI152NLin6Asym')

    # tofail --> MNI152NLin6Asym
    with pytest.raises(ValueError, match='Source space "tofail" not supported'):
        utils.get_std2bold_xfms(bold_file_nlin6asym, source_file=None, source_space='tofail')

    # MNIInfant without cohort --> MNI152NLin6Asym/T1w (fails)
    with pytest.raises(ValueError, match='Source cohort is not specified'):
        utils.get_std2bold_xfms(
            bold_file_nlin6asym,
            source_file=None,
            source_space='MNIInfant',
        )

    # MNIInfant without cohort --> MNI152NLin6Asym --> MNI152NLin2009cAsym/T1w (fails)
    bold_file_target_space = bold_file_nlin6asym.replace(
        'space-MNI152NLin6Asym_',
        'space-MNI152NLin2009cAsym_',
    )
    with pytest.raises(ValueError, match='Source cohort is not specified'):
        utils.get_std2bold_xfms(
            bold_file_target_space,
            source_file=None,
            source_space='MNIInfant',
        )


def test_fwhm2sigma():
    """Test fwhm2sigma."""
    fwhm = 8
    sigma = utils.fwhm2sigma(fwhm)
    assert np.allclose(sigma, 3.39728)


def test_select_first():
    """Test _select_first."""
    lst = ['a', 'b', 'c']
    assert utils._select_first(lst) == 'a'

    lst = 'abc'
    assert utils._select_first(lst) == 'a'


def test_transpose_lol():
    """Test _transpose_lol."""
    inputs = [
        [
            ['a', 'b', 'c'],
            [1, 2, 3],
        ],
        [
            ['a', 'b', 'c', 'd'],
            [1, 2, 3],
        ],
    ]
    outputs = [
        [
            ['a', 1],
            ['b', 2],
            ['c', 3],
        ],
        [
            ['a', 1],
            ['b', 2],
            ['c', 3],
        ],
    ]
    for i, input_ in enumerate(inputs):
        expected_output = outputs[i]
        assert utils._transpose_lol(input_) == expected_output


def test_get_col():
    """Test get_col function with hash suffix support."""
    # Test dataframe with columns that may have hash suffixes
    test_df = pd.DataFrame(
        {
            'motion': [1, 2, 3],
            'motion_hash-abc123': [4, 5, 6],
            'outliers_hash-def456': [7, 8, 9],
            'other_col': [10, 11, 12],
        }
    )

    # Test exact match (should prefer non-hash version)
    result = utils.get_col(test_df, 'motion')
    assert result.tolist() == [1, 2, 3]

    # Test hash-only match
    result = utils.get_col(test_df, 'outliers')
    assert result.tolist() == [7, 8, 9]

    # Test regular column without hash
    result = utils.get_col(test_df, 'other_col')
    assert result.tolist() == [10, 11, 12]

    # Test no match (should raise error)
    with pytest.raises(ValueError, match='No column found matching pattern'):
        utils.get_col(test_df, 'nonexistent')

    # Test with different hash formats
    hash_df = pd.DataFrame(
        {
            'signal_hash-123abc': [1, 2, 3],
            'signal_hash-456def+': [4, 5, 6],
            'signal_hash-789ghi': [7, 8, 9],
        }
    )

    # Should return the first hash variant when no exact match
    result = utils.get_col(hash_df, 'signal')
    assert result.tolist() == [1, 2, 3]
