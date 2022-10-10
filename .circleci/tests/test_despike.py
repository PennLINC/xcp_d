"""Tests for the despike workflow.

This file contains pytest for the despike workflow.
These tests can be run locally and on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.
"""
import os
from pytest import TempPathFactory
import os.path as op
import numpy as np
import nibabel as nb
from nipype.pipeline import engine as pe
from xcp_d.interfaces.regression import CiftiDespike
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.utils.plot import _get_tr

data_dir = '/Users/kahinim/Desktop/xcp_test/data'


def test_nifti_despike(data_dir, tmp_path_factory):
    """
    Test Nifti despiking.

    Confirm that the maximum and minimum voxel values decrease
    after despiking.
    """
    # Read in the necessary inputs
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    maskfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    # Create some spikes in the second voxel
    file_data = read_ndata(boldfile, maskfile)
    voxel_data = file_data[2, :]  # a voxel across time
    voxel_data_mean = np.mean(voxel_data)
    voxel_data_std = np.std(voxel_data)
    voxel_data[2] = voxel_data_mean + 10 * voxel_data_std
    voxel_data[3] = voxel_data_mean - 10 * voxel_data_std
    # What's the min and max, i.e: the amplitude of the spikes created?
    spiked_min = min(voxel_data)
    spiked_max = max(voxel_data)
    # Let's write this temp file out for despiking
    file_data[2, :] = voxel_data
    tempdir = tmp_path_factory.mktemp("test_despike_nifti")
    os.chdir(tempdir)
    filename = "spikedfile.nii.gz"
    write_ndata(data_matrix=file_data, mask=maskfile, template=boldfile, TR=0.8, filename=filename)
    spikedfile = filename
    # Let's despike the image and write it out to a temp file
    tempdir = os.getcwd()
    os.system("3dDespike -NEW -prefix " + tempdir + "/3dDespike.nii.gz " + spikedfile)
    despiked_file = tempdir + "/3dDespike.nii.gz"
    file_data = read_ndata(despiked_file, maskfile)
    voxel_data = file_data[2, :]
    # What's the min and max of the despiked file?
    despiked_min = min(voxel_data)
    despiked_max = max(voxel_data)
    # Have the spikes been reduced?
    # i.e: has the minimum increased after despiking, and the maximum decreased?
    # Additionally, have the affines changed?
    assert spiked_min < despiked_min
    assert spiked_max > despiked_max
    assert np.array_equal(nb.load(despiked_file).affine, nb.load(boldfile).affine)



def test_cifti_despike(data_dir, tmp_path_factory):
    """
    Test Cifti despiking.

    Confirm that the maximum and minimum voxel values decrease
    after despiking.
    """
    boldfile = data_dir + "/fmriprep/sub-colornest001/ses-1/func/"\
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
    # Let's add some noise
    file_data = read_ndata(boldfile)
    voxel_data = file_data[2, :]  # a voxel across time
    voxel_data_mean = np.mean(voxel_data)
    voxel_data_std = np.std(voxel_data)
    voxel_data[2] = voxel_data_mean + 10 * voxel_data_std
    voxel_data[3] = voxel_data_mean - 10 * voxel_data_std
    # What's the maximum and minimum values of the data?
    spiked_max = max(voxel_data)
    spiked_min = min(voxel_data)
    # Let's write this out
    file_data[2, :] = voxel_data
    tempdir = tmp_path_factory.mktemp("test_despike_cifti")
    os.chdir(tempdir)
    filename = "test.nii"
    write_ndata(data_matrix=file_data, template=boldfile, TR=0.8, filename=filename)
    # Let's despike the data
    # Run the node the same way it's run in XCP
    in_file = os.getcwd() + "/" +filename
    TR = _get_tr(nb.load(filename))
    despike3d = pe.Node(CiftiDespike(TR=TR),
                        name="cifti_despike",
                        mem_gb=4,
                        n_procs=2)
    despike3d.inputs.in_file = in_file
    results = despike3d.run()
    # Let's write out the file and read it in as a matrix
    despiked_file = results.outputs.des_file
    despiked_data = read_ndata(despiked_file)
    # What are the minimum and maximum values of the data?
    despiked_max = max(despiked_data[2, :])
    despiked_min = min(despiked_data[2, :])
    # Have the spikes been reduced?
    assert spiked_min < despiked_min
    assert spiked_max > despiked_max
    # Has the intent code changed?
    despiked_intent = nb.load(despiked_file).nifti_header.get_intent()
    original_intent = nb.load(boldfile).nifti_header.get_intent()
    assert despiked_intent[0] == original_intent[0]

    return

