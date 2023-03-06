"""Tests for the despike workflow.

This file contains pytest for the despike workflow.
These tests can be run locally and on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.
"""
import os

import nibabel as nb
import numpy as np
from nipype.pipeline import engine as pe

from xcp_d.interfaces.restingstate import DespikePatch
from xcp_d.interfaces.workbench import CiftiConvert
from xcp_d.utils.write_save import read_ndata, write_ndata


def test_nifti_despike(data_dir, tmp_path_factory):
    """Test Nifti despiking.

    Confirm that the maximum and minimum voxel values decrease
    after despiking.
    """
    # Read in the necessary inputs
    data_dir = os.path.join(data_dir, "fmriprepwithoutfreesurfer/fmriprep/")
    tempdir = tmp_path_factory.mktemp("test_despike_nifti")
    boldfile = os.path.join(
        data_dir,
        (
            "sub-01/func/"
            "sub-01_task-mixedgamblestask_run-1"
            "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        ),
    )
    maskfile = os.path.join(
        data_dir,
        (
            "sub-01/func/"
            "sub-01_task-mixedgamblestask_run-1_"
            "space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        ),
    )

    # Create some spikes in the second voxel
    file_data = read_ndata(boldfile, maskfile)
    voxel_data = file_data[2, :]  # a voxel across time
    voxel_data_mean = np.mean(voxel_data)
    voxel_data_std = np.std(voxel_data)
    voxel_data[2] = voxel_data_mean + 10 * voxel_data_std
    voxel_data[3] = voxel_data_mean - 10 * voxel_data_std

    # What's the min and max, i.e: the amplitude of the spikes created?
    spiked_min = np.min(voxel_data)
    spiked_max = np.max(voxel_data)

    # Let's write this temp file out for despiking
    file_data[2, :] = voxel_data
    spikedfile = os.path.join(tempdir, "spikedfile.nii.gz")
    write_ndata(
        data_matrix=file_data,
        mask=maskfile,
        template=boldfile,
        TR=0.8,
        filename=spikedfile,
    )

    # Let's despike the image and write it out to a temp file
    despike_nifti = pe.Node(DespikePatch(outputtype="NIFTI_GZ", args="-NEW"), name="Despike")
    despike_nifti.inputs.in_file = spikedfile
    res = despike_nifti.run()
    despiked_file = res.outputs.out_file

    assert os.path.isfile(despiked_file)

    file_data = read_ndata(res.outputs.out_file, maskfile)
    voxel_data = file_data[2, :]

    # What's the min and max of the despiked file?
    despiked_min = np.min(voxel_data)
    despiked_max = np.max(voxel_data)

    # Have the spikes been reduced?
    # i.e: has the minimum increased after despiking, and the maximum decreased?
    # Additionally, have the affines changed?
    assert spiked_min < despiked_min
    assert spiked_max > despiked_max

    # Are the affines the same before and after despiking?
    assert np.array_equal(nb.load(despiked_file).affine, nb.load(boldfile).affine)


def test_cifti_despike(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test Cifti despiking.

    Confirm that the maximum and minimum voxel values decrease
    after despiking.
    """
    boldfile = fmriprep_with_freesurfer_data["cifti_file"]
    TR = 0.8

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
    filename = os.path.join(tempdir, "test.dtseries.nii")

    write_ndata(data_matrix=file_data, template=boldfile, TR=TR, filename=filename)

    # Let's despike the data
    # first, convert the cifti to a nifti
    convert_to_nifti = CiftiConvert(target="to")
    convert_to_nifti.inputs.in_file = filename
    convert_to_nifti_results = convert_to_nifti.run(cwd=tempdir)

    # next, run 3dDespike
    despike3d = DespikePatch(outputtype="NIFTI_GZ", args="-nomask -NEW")
    despike3d.inputs.in_file = convert_to_nifti_results.outputs.out_file
    despike3d_results = despike3d.run(cwd=tempdir)

    # finally, convert the despiked nifti back to cifti
    convert_to_cifti = CiftiConvert(target="from", TR=TR)
    convert_to_cifti.inputs.in_file = despike3d_results.outputs.out_file
    convert_to_cifti.inputs.cifti_template = filename
    convert_to_cifti_results = convert_to_cifti.run(cwd=tempdir)

    # Let's write out the file and read it in as a matrix
    despiked_file = convert_to_cifti_results.outputs.out_file
    assert os.path.isfile(despiked_file)
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
