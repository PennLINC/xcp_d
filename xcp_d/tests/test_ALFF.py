"""Test for alff."""

# Necessary imports

import os

import nibabel as nb
from numpy.fft import fft, ifft

from xcp_d.utils.bids import _get_tr
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflows.restingstate import init_compute_alff_wf


def test_nifti_alff(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test ALFF computations as done for Niftis.

    Get the FFT of a Nifti, add to the amplitude of its lower frequencies
    and confirm the mean ALFF after addition to lower frequencies
    has increased.
    """
    # Get the file names
    bold_file = fmriprep_with_freesurfer_data["nifti_file"]
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]

    # Let's initialize the ALFF node
    TR = _get_tr(nb.load(bold_file))
    compute_alff_wf = init_compute_alff_wf(
        omp_nthreads=2,
        bold_file=bold_file,
        mem_gb=4,
        TR=TR,
        lowpass=0.08,
        highpass=0.01,
        cifti=False,
        smoothing=6,
        name="compute_alff_wf",
    )

    # Let's move to a temporary directory before running
    tempdir = tmp_path_factory.mktemp("test_ALFF_nifti")
    compute_alff_wf.base_dir = tempdir
    compute_alff_wf.inputs.inputnode.bold_mask = bold_mask
    compute_alff_wf.inputs.inputnode.clean_bold = bold_file
    compute_alff_res = compute_alff_wf.run()
    nodes = {node.fullname: node for node in compute_alff_res.nodes}

    # Let's get the mean of the ALFF for later comparison
    original_alff = nodes["compute_alff_wf.alff_compt"].get_output("alff_out")
    original_alff_data_mean = nb.load(original_alff).get_fdata().mean()

    # Now let's do an FFT
    original_bold_data = read_ndata(bold_file, bold_mask)

    # Let's work with a single voxel
    voxel_data = original_bold_data[2, :]
    fft_data = fft(voxel_data)
    mean = fft_data.mean()

    # Let's increase the values of the first few frequency's amplitudes
    # to create fake data
    fft_data[:11] += 300 * mean

    # Let's convert this back into time domain
    changed_voxel_data = ifft(fft_data)
    # Let's replace the original value with the fake data
    original_bold_data[2, :] = changed_voxel_data
    # Let's write this out
    filename = os.path.join(tempdir, "editedfile.nii.gz")
    write_ndata(original_bold_data, template=bold_file, mask=bold_mask, filename=filename)

    # Now let's compute ALFF for the new file and see how it compares
    # to the original ALFF - it should increase since we increased
    # the amplitude in low frequencies for a voxel
    tempdir = tmp_path_factory.mktemp("test_ALFF_nifti_dir2")
    compute_alff_wf.base_dir = tempdir
    compute_alff_wf.inputs.inputnode.bold_mask = bold_mask
    compute_alff_wf.inputs.inputnode.clean_bold = filename
    compute_alff_res = compute_alff_wf.run()
    nodes = {node.fullname: node for node in compute_alff_res.nodes}

    # Let's get the new ALFF mean
    new_alff = nodes["compute_alff_wf.alff_compt"].get_output("alff_out")
    assert os.path.isfile(new_alff)
    new_alff_data_mean = nb.load(new_alff).get_fdata().mean()

    # Now let's make sure ALFF has increased ...
    assert new_alff_data_mean > original_alff_data_mean


def test_cifti_alff(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test ALFF computations as done for Ciftis.

    Get the FFT of a Cifti, add to the amplitude of its lower frequencies
    and confirm the ALFF after addition to lower frequencies
    has changed in the expected direction.
    """
    bold_file = fmriprep_with_freesurfer_data["cifti_file"]
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]

    # Let's initialize the ALFF node
    TR = _get_tr(nb.load(bold_file))
    compute_alff_wf = init_compute_alff_wf(
        omp_nthreads=2,
        bold_file=bold_file,
        mem_gb=4,
        TR=TR,
        lowpass=0.08,
        highpass=0.01,
        cifti=True,
        smoothing=6,
    )

    # Let's move to a temporary directory before running
    tempdir = tmp_path_factory.mktemp("test_ALFF_cifti")
    compute_alff_wf.base_dir = tempdir
    compute_alff_wf.inputs.inputnode.bold_mask = bold_mask
    compute_alff_wf.inputs.inputnode.clean_bold = bold_file
    compute_alff_res = compute_alff_wf.run()
    nodes = {node.fullname: node for node in compute_alff_res.nodes}

    # Let's get the mean of the data for later comparison
    original_alff = nodes["compute_alff_wf.alff_compt"].get_output("alff_out")
    original_alff_data_mean = nb.load(original_alff).get_fdata().mean()

    # Now let's do an FFT
    original_bold_data = read_ndata(bold_file, bold_mask)

    # Let's work with a single voxel
    voxel_data = original_bold_data[2, :]
    fft_data = fft(voxel_data)
    mean = fft_data.mean()
    # Let's increase the amplitudes for the lower frequencies
    fft_data[:11] += 300 * mean
    # Let's get this back into the time domain
    changed_voxel_data = ifft(fft_data)
    # Let's replace the original value with the fake data
    original_bold_data[2, :] = changed_voxel_data

    # Let's write this out
    filename = os.path.join(tempdir, "editedfile.dtseries.nii")
    write_ndata(original_bold_data, template=bold_file, mask=bold_mask, filename=filename)

    # Now let's compute ALFF for the new file and see how it compares
    tempdir = tmp_path_factory.mktemp("test_ALFF_cifti_dir2")
    compute_alff_wf.base_dir = tempdir
    compute_alff_wf.inputs.inputnode.bold_mask = bold_mask
    compute_alff_wf.inputs.inputnode.clean_bold = filename
    compute_alff_res = compute_alff_wf.run()
    nodes = {node.fullname: node for node in compute_alff_res.nodes}

    # Let's get the new ALFF mean
    new_alff = nodes["compute_alff_wf.alff_compt"].get_output("alff_out")
    assert os.path.isfile(new_alff)
    new_alff_data_mean = nb.load(new_alff).get_fdata().mean()

    # Now let's make sure ALFF has increased, as we added
    # to the amplitude of the lower frequencies in a voxel
    assert new_alff_data_mean > original_alff_data_mean
