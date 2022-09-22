"""Tests for interpolation methods.

source: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
"""
import os
import tempfile

import numpy as np
from scipy.fftpack import fft

from xcp_d.interfaces.prepostcleaning import Interpolate
from xcp_d.utils import read_ndata, write_ndata


def test_interpolate_cifti(data_dir, tmp_path_factory):
    """Test CIFTI interpolation."""
    # edit this if you want to see the edited confounds
    tmpdir = tmp_path_factory.mktemp("test_interpolate_cifti")

    # CIFTI - ORIGINAL SIGNAL
    # Feed in inputs
    boldfile = os.path.join(
        data_dir,
        "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii",
    )
    TR = 2.5

    # Let's replace some voxels in the original bold_file
    file_data = read_ndata(boldfile, maskfile=None)

    # Let's make a basic signal to replace the data
    ts = file_data.shape[1]
    t = np.linspace(0, 1, ts)
    freq = 1.0
    x = 3 * np.sin(2 * np.pi * freq * t)
    freq = 4
    x += np.sin(2 * np.pi * freq * t)
    freq = 7
    x += 0.5 * np.sin(2 * np.pi * freq * t)

    # Let's get the fft of x and plot it
    X = fft(x)
    fft_original = X

    # Create some spikes we can add later
    stdev = np.std(x)
    spike = 2 * stdev + np.mean(x)

    # Let's replace voxel 3
    file_data[3, :] = x

    # Let's replace some timepoints with a spike
    # in each of these columns
    file_data[3, 3] = spike
    file_data[3, 88] = spike
    file_data[3, 48] = spike
    file_data[3, 20] = spike
    file_data[3, 100] = spike

    # Let's get the fft of a noisy timepoint and plot it:
    X = fft(file_data[3, :])
    fft_spike = X
    # sourced from python notebook - FFT
    N = len(X)
    n = np.arange(N)
    freq = n / (N * TR)

    # let's save out this noisy file
    noise_file = os.path.join(tmpdir, "noisy_file.dtseries.nii")
    write_ndata(
        data_matrix=file_data,
        template=boldfile,
        mask=None,
        TR=TR,
        filename=noise_file,
    )
    assert os.path.isfile(noise_file)

    # Start testing interpolation - feed in input
    # Let's create a fake tmask...
    tmask_file = os.path.join(tmpdir, "tmask.tsv")
    tmask = np.zeros(N)
    tmask[3] = 1
    tmask[88] = 1
    tmask[48] = 1
    tmask[20] = 1
    tmask[100] = 1
    np.savetxt(tmask_file, tmask)

    interpolation = Interpolate()
    interpolation.inputs.tmask = tmask_file
    interpolation.inputs.in_file = noise_file
    interpolation.inputs.bold_file = boldfile
    interpolation.inputs.TR = TR
    results = interpolation.run()

    assert os.path.isfile(results.outputs.bold_interpolated)

    # FFT for interpolated bold_file
    # Read in file
    file_data = read_ndata(results.outputs.bold_interpolated, maskfile=None)
    voxel_data = file_data[3, :]  # the previously noisy voxel

    # Let's get the FFT of this voxel and plot it
    # from python notebook
    fft_interpolated = fft(voxel_data)  # i.e: fft_interpolated
    N = len(fft_interpolated)
    n = np.arange(N)
    freq = n / (N * TR)

    # The differences between the interpolated and original signal should be less
    # than the differences between the interpolated and spiky signal.
    diff1 = sum(abs(fft_interpolated - fft_original))
    diff2 = sum(abs(fft_interpolated - fft_spike))

    assert diff1 < diff2


def test_interpolate_nifti(data_dir):
    """Check results - first must censor file."""
    boldfile = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    TR = 0.5
    mask = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )
    # Let's replace some voxels in the original bold_file
    file_data = read_ndata(boldfile, mask)

    # Let's make a basic signal to replace the data

    ts = file_data.shape[1]
    t = np.linspace(0, 1, ts)
    freq = 1.0
    x = 3 * np.sin(2 * np.pi * freq * t)
    freq = 4
    x += np.sin(2 * np.pi * freq * t)
    freq = 7
    x += 0.5 * np.sin(2 * np.pi * freq * t)

    # Let's get the fft of x and plot it
    X = fft(x)
    fft_original = X
    # sourced from python notebook - FFT
    N = len(X)
    n = np.arange(N)
    freq = n / (N * TR)

    # Create some spikes we can add later
    stdev = np.std(x)
    spike = 2 * stdev + np.mean(x)

    # Let's replace voxel 3
    file_data[3, :] = x

    # Let's replace some timepoints with a spike
    # in each of these columns
    file_data[3, 3] = spike
    file_data[3, 12] = spike
    file_data[3, 10] = spike
    file_data[3, 6] = spike
    file_data[3, 8] = spike

    # Let's get the fft of a noisy timepoint and plot it:
    X = fft(file_data[3, :])
    fft_spike = X
    # sourced from python notebook - FFT
    N = len(X)
    n = np.arange(N)
    freq = n / (N * TR)

    # Since this looks good, let's save out this noisy file
    tmpdir = tempfile.mkdtemp()  # edit this if you want to see the edited confounds
    # on your Desktop, etc.
    os.chdir(tmpdir)
    write_ndata(
        data_matrix=file_data,
        template=boldfile,
        mask=mask,
        TR=TR,
        filename="noisy_file.nii.gz",
    )

    # Start testing interpolation - feed in input
    # Let's create a fake tmask...
    tmask = np.zeros(N)
    tmask[3] = 1
    tmask[12] = 1
    tmask[10] = 1
    tmask[6] = 1
    tmask[8] = 1
    np.savetxt("tmask.tsv", tmask)

    interpolation = Interpolate()
    interpolation.inputs.tmask = "tmask.tsv"
    interpolation.inputs.in_file = "noisy_file.nii.gz"
    interpolation.inputs.bold_file = boldfile
    interpolation.inputs.TR = TR
    interpolation.inputs.mask_file = mask
    results = interpolation.run()

    # FFT for interpolated bold_file
    # Read in file
    file_data = read_ndata(results.outputs.bold_interpolated, mask)
    voxel_data = file_data[3, :]  # the previously noisy voxel

    # Let's get the FFT of this voxel and plot it
    # from python notebook
    X = fft(voxel_data)  # i.e: fft_interpolated
    fft_interpolated = X
    N = len(X)
    n = np.arange(N)
    freq = n / (N * TR)

    # Are the differences between the interpolated and original signal less
    # than the differences between the interpolated and spiky signal?
    diff1 = sum(abs(fft_interpolated - fft_original))
    diff2 = sum(abs(fft_interpolated - fft_spike))
    assert diff1 < diff2
