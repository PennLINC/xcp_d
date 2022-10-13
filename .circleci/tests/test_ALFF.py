"""Test for ALFF."""

# Necessary imports

import os
import nibabel as nb
import numpy as np

from numpy.fft import fft, ifft

from xcp_d.utils.plot import _get_tr
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflow.restingstate import init_compute_alff_wf


def test_nifti_alff(data_dir, tmp_path_factory):
    """
    Test ALFF computations as done for Niftis.

    Get the FFT of a Nifti, add to the amplitude of its lower frequencies
    and confirm the mean ALFF after addition to lower frequencies
    has increased.
    """
    # Get the file names
    bold_file = os.path.join(
        data_dir, "fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym"
        "_desc-preproc_bold.nii.gz"
    )
    bold_mask = os.path.join(
        data_dir, "fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym"
        "_desc-brain_mask.nii.gz"
    )
    # Let's initialize the ALFF node
    TR = _get_tr(nb.load(bold_file))
    alff_compute_wf = init_compute_alff_wf(
        omp_nthreads=2,
        mem_gb=4,
        TR=TR,
        lowpass=0.08,
        highpass=0.009,
        cifti=False,
        smoothing=6,
    )
    # Let's move to a temporary directory before running
    tempdir = tmp_path_factory.mktemp("test_ALFF_nifti")
    alff_compute_wf.base_dir = tempdir
    alff_compute_wf.inputs.inputnode.bold_mask = bold_mask
    alff_compute_wf.inputs.inputnode.clean_bold = bold_file
    alff_compute_wf.run()
    # Let's get the mean of the ALFF for later comparison
    original_alff = os.path.join(
        tempdir, "compute_alff_wf/alff_compt/sub-color"
        "nest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-"
        "preproc_bold_alff.nii.gz"
    )
    original_alff_data_mean = nb.load(original_alff).get_fdata().mean()
    # Now let's do an FFT 
    original_bold_data = read_ndata(bold_file, bold_mask)
    # Let's work with a single voxel
    voxel_data = original_bold_data[2, :]
    fft_data = fft(voxel_data)
    mean = fft_data.mean()
    # Let's increase the values of the first few frequency's amplitudes
    # to create fake data
    fft_data[:, :11] += 300 * mean
    # Let's convert this back into time domain
    changed_voxel_data = ifft(fft_data)
    # Let's replace the original value with the fake data
    original_bold_data[2, :] = changed_voxel_data
    # Let's write this out
    filename = os.path.join(tempdir, "editedfile.nii.gz")
    write_ndata(
        original_bold_data, template=bold_file, mask=bold_mask, filename=filename
    )
    # Now let's compute ALFF for the new file and see how it compares 
    # to the original ALFF - it should increase since we increased
    # the amplitude in low frequencies for a voxel
    tempdir = tmp_path_factory.mktemp("test_ALFF_nifti_dir2")
    alff_compute_wf.base_dir = tempdir
    alff_compute_wf.inputs.inputnode.bold_mask = bold_mask
    alff_compute_wf.inputs.inputnode.clean_bold = filename
    alff_compute_wf.run()
    # Let's get the new ALFF mean
    new_alff = os.path.join(tempdir, "compute_alff_wf/alff_compt/"
                            "editedfile_alff.nii.gz")
    new_alff_data_mean = nb.load(new_alff).get_fdata().mean()
    # Now let's make sure ALFF has increased ...
    assert new_alff_data_mean > original_alff_data_mean


def test_cifti_alff(data_dir, tmp_path_factory):
    """
    Test ALFF computations as done for Ciftis.

    Get the FFT of a Cifti, add to the amplitude of its lower frequencies
    and confirm the ALFF after addition to lower frequencies
    has changed in the expected direction.
    """
    bold_file = os.path.join(
        data_dir + "fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-res"
        "t_run-2_space-fsLR_den-91k_bold.dtseries.nii"
    )
    bold_mask = os.path.join(
        data_dir + "fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym"
        "_desc-brain_mask.nii.gz"
    )
    # Let's initialize the ALFF node
    TR = _get_tr(nb.load(bold_file))
    alff_compute_wf = init_compute_alff_wf(
        omp_nthreads=2,
        mem_gb=4,
        TR=TR,
        lowpass=0.08,
        highpass=0.009,
        cifti=True,
        smoothing=6,
    )
    # Let's move to a temporary directory before running
    tempdir = tmp_path_factory.mktemp("test_ALFF_cifti")
    alff_compute_wf.base_dir = tempdir
    alff_compute_wf.inputs.inputnode.bold_mask = bold_mask
    alff_compute_wf.inputs.inputnode.clean_bold = bold_file
    alff_compute_wf.run()
    # Let's get the mean of the data for later comparison
    original_alff = os.path.join(
        tempdir, "compute_alff_wf/alff_compt/sub-color"
        "nest001_ses-1_task-rest_run-2_space-fsLR_den-91k_"
        "bold_alff.dtseries.nii"
    )
    original_alff_data_mean = nb.load(original_alff).get_fdata().mean()
    # Now let's do an FFT 
    original_bold_data = read_ndata(bold_file, bold_mask)
    # Let's work with a single voxel
    voxel_data = original_bold_data[2, :]
    fft_data = fft(voxel_data)
    mean = fft_data.mean()
    # Let's increase the amplitudes for the lower frequencies
    fft_data[:, :11] += 300 * mean
    # Let's get this back into the time domain
    changed_voxel_data = ifft(fft_data)
    # Let's replace the original value with the fake data
    original_bold_data[2, :] = changed_voxel_data
    # Let's write this out
    filename = os.path.join(tempdir, "editedfile.dtseries.nii")
    write_ndata(
        original_bold_data, template=bold_file, mask=bold_mask,
        filename=filename
    )
    # Now let's compute ALFF for the new file and see how it compares
    tempdir = tmp_path_factory.mktemp("test_ALFF_cifti_dir2")
    alff_compute_wf.base_dir = tempdir
    alff_compute_wf.inputs.inputnode.bold_mask = bold_mask
    alff_compute_wf.inputs.inputnode.clean_bold = filename
    alff_compute_wf.run()
    # Let's get the new ALFF mean
    new_alff = os.path.join(
        tempdir, "compute_alff_wf/alff_compt/editedfile_alff.dtseries.nii"
    )
    new_alff_data_mean = nb.load(new_alff).get_fdata().mean()
    # Now let's make sure ALFF has increased, as we added
    # to the amplitude of the lower frequencies in a voxel
    assert new_alff_data_mean > original_alff_data_mean
    return
