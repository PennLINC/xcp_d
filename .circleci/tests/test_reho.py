"""Test for ReHo."""
import os

import nibabel as nb
import numpy as np

from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflow.restingstate import init_3d_reho_wf, init_surface_reho_wf


def noisy(image):
    """
    Add Gaussian noise.

    Source: "https://stackoverflow.com/questions/22937589/" \
    "how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-" \
    "in-python-with-opencv"
    """
    row, col = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy_img = image + gauss * 200
    return noisy_img


def test_nifti_reho(data_dir, tmp_path_factory):
    """
    Test Nifti ReHo Computation.

    Confirm that ReHo decreases after adding noise to a
    Nifti image.
    """
    # Get the names of the files
    bold_file = os.path.join(
        data_dir, "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_",
        "task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    bold_mask = os.path.join(
        data_dir, "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_",
        "task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )
    # Set up and run the ReHo wf in a tempdir
    reho_wf = init_3d_reho_wf(omp_nthreads=2, mem_gb=4)
    reho_wf.inputs.inputnode.bold_mask = bold_mask
    tempdir = tmp_path_factory.mktemp("test_REHO_nifti")
    reho_wf.base_dir = tempdir
    reho_wf.inputs.inputnode.clean_bold = bold_file
    reho_wf.run()
    # Get the original mean of the ReHo for later comparison
    original_reho = os.path.join(reho_wf.base_dir,
                                 "afni_reho_wf/reho_3d/reho.nii.gz")
    original_reho_mean = nb.load(original_reho).get_fdata().mean()
    original_bold_data = read_ndata(bold_file, bold_mask)
    # Add some noise to the original data and write it out
    noisy_bold_data = noisy(original_bold_data)
    filename = os.path.join(tempdir, "/test.nii.gz")
    write_ndata(noisy_bold_data, template=bold_file, mask=bold_mask, filename=filename)
    # Run ReHo again
    reho_wf.inputs.inputnode.clean_bold = filename
    reho_wf.run()
    # Has the new ReHo's mean decreased?
    new_reho = os.path.join(reho_wf.base_dir,
                            "afni_reho_wf/reho_3d/reho.nii.gz")
    new_reho_mean = nb.load(new_reho).get_fdata().mean()
    assert new_reho_mean < original_reho_mean
    return


def test_cifti_reho(data_dir, tmp_path_factory):
    """
    Test Cifti ReHo Computation.

    Confirm that ReHo decreases after adding noise to a
    Cifti image.
    """
    # Get the names of the files
    bold_file = os.path.join(
        data_dir, "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_",
        "task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
    )
    # Set up and run the ReHo wf in a tempdir
    reho_wf = init_surface_reho_wf(omp_nthreads=2, mem_gb=4)
    tempdir = tmp_path_factory.mktemp("test_REHO_cifti")
    reho_wf.base_dir = tempdir
    reho_wf.inputs.inputnode.clean_bold = bold_file
    reho_wf.run()
    # Get the original mean of the ReHo for later comparison
    original_reho = os.path.join(
        tempdir, "surface_reho_wf/reho_lh",
        "correlation_matrix_",
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR",
        "_den-91k_bold.dtseries.shape.gii"
    )
    original_reho_mean = nb.load(original_reho).agg_data().mean()
    original_bold_data = read_ndata(bold_file)
    # Add some noise to the original data and write it out
    noisy_bold_data = noisy(original_bold_data)
    filename = os.path.join(tempdir, "/test.dtseries.nii")
    write_ndata(noisy_bold_data, template=bold_file, filename=filename)
    # Run ReHo again
    reho_wf.inputs.inputnode.clean_bold = filename
    reho_wf.run()
    # Has the new ReHo's mean decreased?
    new_reho = os.path.join(
        tempdir, "surface_reho_wf/reho_lh/correlation_matrix_",
        "test.dtseries.shape.gii"
    )
    new_reho_mean = nb.load(new_reho).agg_data().mean()
    assert new_reho_mean < original_reho_mean
    return
