"""Tests for xcp_d.utils.restingstate."""

import numpy as np
from nilearn import masking

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
    bold_data = bold_data[:, sample_mask]

    alff2 = restingstate.compute_alff_chunk((bold_data, 0.1, 0.01, TR, sample_mask))

    # Now let's make sure ALFF has increased ...
    assert alff2[101] > alff2[100]
