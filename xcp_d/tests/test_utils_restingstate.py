"""Tests for xcp_d.utils.restingstate."""

import numpy as np
from nilearn import masking


def test_compute_alff(ds001419_data):
    """Test ALFF calculated from an array.

    Get the FFT of a Nifti, add to the amplitude of its lower frequencies
    and confirm the mean ALFF after addition to lower frequencies
    has increased.
    """
    from xcp_d.utils.restingstate import compute_alff

    # Get the file names
    bold_file = ds001419_data["nifti_file"]
    bold_mask = ds001419_data["brain_mask_file"]

    # Let's initialize the ALFF node
    TR = 3
    bold_data = masking.apply_mask(bold_file, bold_mask).T

    original_alff = compute_alff(
        data_matrix=bold_data,
        low_pass=0.1,
        high_pass=0.01,
        TR=TR,
        sample_mask=None,
    )

    # Now let's do an FFT
    # Let's work with a single voxel
    voxel_data = bold_data[2, :]
    fft_data = np.fft.fft(voxel_data)
    mean = fft_data.mean()

    # Let's increase the values of the first few frequencies' amplitudes to create fake data
    fft_data[:11] += 300 * mean

    # Let's convert this back into time domain
    changed_voxel_data = np.fft.ifft(fft_data)
    # Let's replace the original value with the fake data
    bold_data[2, :] = changed_voxel_data

    # Now let's compute ALFF for the new file and see how it compares
    # to the original ALFF - it should increase since we increased
    # the amplitude in low frequencies for a voxel
    new_alff = compute_alff(
        data_matrix=bold_data,
        low_pass=0.1,
        high_pass=0.01,
        TR=TR,
        sample_mask=None,
    )

    # Now let's make sure ALFF has increased ...
    assert new_alff[2] > original_alff[2]
