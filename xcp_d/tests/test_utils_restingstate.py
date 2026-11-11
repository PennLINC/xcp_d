"""Tests for xcp_d.utils.restingstate."""

import warnings

import numpy as np
import pytest
from nilearn import masking
from scipy import signal

from xcp_d.utils import restingstate


def test_compute_alff(ds001419_data):
    """Test ALFF calculated from an array.

    Get the FFT of a Nifti, add to the amplitude of its lower frequencies
    and confirm the mean ALFF after addition to lower frequencies
    has increased.
    """

    # Get the file names
    bold_file = ds001419_data['nifti_file']
    bold_mask = ds001419_data['brain_mask_file']

    # Let's initialize the ALFF node
    TR = 3
    bold_data = masking.apply_mask(bold_file, bold_mask).T

    # Now let's do an FFT
    # Let's work with a single voxel
    voxel_data = bold_data[100, :]
    fft_data = np.fft.fft(voxel_data)
    mean = fft_data.mean()

    # Let's increase the values of the first few frequencies' amplitudes to create fake data
    fft_data[:11] += 300 * mean

    # Let's convert this back into time domain
    changed_voxel_data = np.fft.ifft(fft_data)
    # Let's replace the original value with the fake data
    bold_data[101, :] = changed_voxel_data

    alff = restingstate.compute_alff(
        data_matrix=bold_data,
        low_pass=0.1,
        high_pass=0.01,
        TR=TR,
        sample_mask=None,
    )

    # Now let's make sure ALFF has increased ...
    assert alff[101] > alff[100]

    # Now try with a sample mask
    sample_mask = np.ones(bold_data.shape[1], dtype=bool)
    sample_mask[20:30] = False

    alff2 = restingstate.compute_alff(
        data_matrix=bold_data,
        low_pass=0.1,
        high_pass=0.01,
        TR=TR,
        sample_mask=sample_mask,
    )

    # Now let's make sure ALFF has increased ...
    assert alff2[101] > alff2[100]


def _reference_compute_alff(*, data_matrix, low_pass, high_pass, TR, sample_mask):
    """The original per-voxel ALFF implementation, frozen as a reference."""
    fs = 1 / TR  # sampling frequency
    n_voxels, n_volumes = data_matrix.shape

    alff = np.zeros(n_voxels)
    for i_voxel in range(n_voxels):
        voxel_data = data_matrix[i_voxel, :]
        if np.nanstd(voxel_data) == 0:
            alff[i_voxel] = 0
            continue

        sd_scale = np.nanstd(voxel_data)

        if sample_mask is not None:
            voxel_data_censored = voxel_data[sample_mask]
            voxel_data_censored -= np.nanmean(voxel_data_censored)
            voxel_data_censored /= np.nanstd(voxel_data_censored)

            time_arr = np.arange(n_volumes) * TR
            time_arr = time_arr[sample_mask]
            frequencies_hz = np.linspace(0, 0.5 * fs, (n_volumes // 2) + 1)[1:]
            angular_frequencies = 2 * np.pi * frequencies_hz
            power_spectrum = signal.lombscargle(
                time_arr,
                voxel_data_censored,
                angular_frequencies,
                normalize=True,
            )
        else:
            voxel_data -= np.nanmean(voxel_data)
            voxel_data /= np.nanstd(voxel_data)
            frequencies_hz, power_spectrum = signal.periodogram(
                voxel_data,
                fs,
                scaling='spectrum',
            )

        power_spectrum_sqrt = np.sqrt(power_spectrum)
        if high_pass == 0:
            high_pass = frequencies_hz[0]

        if low_pass == 0:
            low_pass = frequencies_hz[-1]

        ff_alff = [
            np.argmin(np.abs(frequencies_hz - high_pass)),
            np.argmin(np.abs(frequencies_hz - low_pass)),
        ]
        alff[i_voxel] = len(ff_alff) * np.nanmean(power_spectrum_sqrt[ff_alff[0] : ff_alff[1]])
        alff[i_voxel] *= sd_scale

    return alff


def _make_alff_data(n_voxels, n_volumes, sample_mask, rng):
    """Simulate AR(1) time series at varied scales, with edge-case voxels in the first rows."""
    data = np.zeros((n_voxels, n_volumes))
    noise = rng.standard_normal((n_voxels, n_volumes))
    phi = rng.uniform(0.1, 0.95, n_voxels)
    for t in range(1, n_volumes):
        data[:, t] = phi * data[:, t - 1] + noise[:, t]

    data = data * rng.uniform(1, 200, (n_voxels, 1)) + rng.uniform(-500, 20000, (n_voxels, 1))

    retained = np.flatnonzero(sample_mask) if sample_mask is not None else np.arange(n_volumes)
    data[0] = 0  # all zeros
    data[1] = 7.25  # constant
    data[2, retained[len(retained) // 2]] = np.nan  # NaN in a retained volume
    data[3] = np.nan  # all NaN
    data[4, 0] = np.inf
    data[5, retained] = 5.0  # constant across retained volumes only
    if sample_mask is not None:
        data[6, np.flatnonzero(~sample_mask)[0]] = np.nan  # NaN in a censored volume

    return data


@pytest.mark.parametrize('n_volumes', [11, 120, 401])
@pytest.mark.parametrize('censoring', ['none', 'random', 'edges'])
@pytest.mark.parametrize('dtype', ['float64', 'float32'])
def test_compute_alff_matches_reference(n_volumes, censoring, dtype):
    """Check that compute_alff matches the original per-voxel implementation bit for bit.

    Data are read as float32 and transposed (Fortran-ordered) by read_ndata,
    and compute_alff processes them in float64.
    """
    rng = np.random.default_rng(n_volumes)
    sample_mask = None
    if censoring == 'random':
        sample_mask = np.ones(n_volumes, dtype=bool)
        sample_mask[rng.choice(n_volumes, max(1, n_volumes // 5), replace=False)] = False
    elif censoring == 'edges':
        sample_mask = np.ones(n_volumes, dtype=bool)
        sample_mask[[0, 1, -1]] = False

    data = _make_alff_data(40, n_volumes, sample_mask, rng)
    data = np.ascontiguousarray(data.T.astype(dtype)).T
    original = data.copy(order='A')

    for low_pass, high_pass, TR in ((0.08, 0.01, 0.8), (0, 0.01, 2.0), (0.1, 0, 3), (0, 0, 0.72)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expected = _reference_compute_alff(
                data_matrix=data.astype(np.float64),
                low_pass=low_pass,
                high_pass=high_pass,
                TR=TR,
                sample_mask=sample_mask,
            )
            alff = restingstate.compute_alff(
                data_matrix=data,
                low_pass=low_pass,
                high_pass=high_pass,
                TR=TR,
                sample_mask=sample_mask,
            )

        assert alff.dtype == expected.dtype
        assert alff.tobytes() == expected.tobytes()
        # The input array must not be modified.
        assert data.tobytes() == original.tobytes()


def test_compute_alff_matches_reference_across_blocks():
    """Check that voxels in later blocks match the reference too."""
    rng = np.random.default_rng(0)
    n_volumes = 1000
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[rng.choice(n_volumes, 200, replace=False)] = False
    # 2**21 // 1000 = 2097 voxels per block, so this spans two blocks.
    data = rng.standard_normal((2200, n_volumes)).astype(np.float32) * 40 + 10000
    data[[2096, 2097, 2150]] = 0

    expected = _reference_compute_alff(
        data_matrix=data.astype(np.float64),
        low_pass=0.08,
        high_pass=0.01,
        TR=0.8,
        sample_mask=sample_mask,
    )
    alff = restingstate.compute_alff(
        data_matrix=data,
        low_pass=0.08,
        high_pass=0.01,
        TR=0.8,
        sample_mask=sample_mask,
    )
    assert alff.tobytes() == expected.tobytes()


@pytest.mark.parametrize('n_volumes', [2000, 2400])
def test_compute_alff_matches_reference_long_uncensored(n_volumes):
    """Check long uncensored scans, for which a batched FFT would not match the per-voxel FFT."""
    rng = np.random.default_rng(n_volumes)
    data = _make_alff_data(30, n_volumes, None, rng)
    data = np.ascontiguousarray(data.T).T

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = _reference_compute_alff(
            data_matrix=data.copy(order='A'),
            low_pass=0.08,
            high_pass=0.01,
            TR=0.8,
            sample_mask=None,
        )
        alff = restingstate.compute_alff(
            data_matrix=data,
            low_pass=0.08,
            high_pass=0.01,
            TR=0.8,
            sample_mask=None,
        )

    assert alff.tobytes() == expected.tobytes()


def test_lombscargle_normalized_matches_scipy():
    """Check the reimplementation of SciPy's Lomb-Scargle periodogram against SciPy itself."""
    rng = np.random.default_rng(1)
    n_volumes = 300
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[rng.choice(n_volumes, 60, replace=False)] = False
    time_arr = (np.arange(n_volumes) * 0.8)[sample_mask]
    frequencies_hz = np.linspace(0, 0.5 / 0.8, (n_volumes // 2) + 1)[1:]
    angular_frequencies = 2 * np.pi * frequencies_hz
    data = rng.standard_normal((25, time_arr.size))

    terms = restingstate._lombscargle_terms(time_arr, angular_frequencies)
    pgram = restingstate._lombscargle_normalized(data, *terms)
    for i_row in range(data.shape[0]):
        expected = signal.lombscargle(time_arr, data[i_row], angular_frequencies, normalize=True)
        assert pgram[i_row].tobytes() == expected.tobytes()


def test_compute_2d_reho_float32():
    """Check that ReHo is the same for float32 data as for the same data in float64."""
    rng = np.random.default_rng(0)
    n_vertices, n_volumes = 30, 300
    data = (rng.standard_normal((n_vertices, n_volumes)) * 50 + 10000).astype(np.float32)
    adjacency_matrix = rng.random((n_vertices, n_vertices)) < 0.2
    adjacency_matrix = np.logical_or(adjacency_matrix, adjacency_matrix.T)
    np.fill_diagonal(adjacency_matrix, False)

    reho = restingstate.compute_2d_reho(datat=data, adjacency_matrix=adjacency_matrix)
    expected = restingstate.compute_2d_reho(
        datat=data.astype(np.float64),
        adjacency_matrix=adjacency_matrix,
    )
    assert reho.tobytes() == expected.tobytes()
