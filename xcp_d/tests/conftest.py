"""Fixtures for the CircleCI tests."""
import os

import pytest


def pytest_addoption(parser):
    """Collect pytest parameters for running tests."""
    parser.addoption("--working_dir", action="store", default="/tmp")
    parser.addoption("--data_dir", action="store")
    parser.addoption("--output_dir", action="store")


# Set up the commandline options as fixtures
@pytest.fixture
def data_dir(request):
    """Grab data directory."""
    return request.config.getoption("--data_dir")


@pytest.fixture
def working_dir(request):
    """Grab working directory."""
    return request.config.getoption("--working_dir")


@pytest.fixture
def output_dir(request):
    """Grab output directory."""
    return request.config.getoption("--output_dir")


@pytest.fixture
def datasets(data_dir):
    """Locate downloaded datasets."""
    dsets = {}
    dsets["ds001419"] = os.path.join(data_dir, "ds001419-fmriprep")
    dsets["nibabies"] = os.path.join(data_dir, "nibabies_test_data/derivatives/nibabies")
    dsets["fmriprep_without_freesurfer"] = os.path.join(
        data_dir,
        "fmriprepwithoutfreesurfer/fmriprep",
    )
    return dsets


@pytest.fixture
def fmriprep_with_freesurfer_data(datasets):
    """Collect a list of files from ds001419 that will be used by misc. tests."""
    subj_dir = os.path.join(datasets["ds001419"], "sub-01")
    func_dir = os.path.join(subj_dir, "func")
    anat_dir = os.path.join(subj_dir, "anat")

    files = {}
    files["nifti_file"] = os.path.join(
        func_dir,
        "sub-01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    )
    files["cifti_file"] = os.path.join(
        func_dir,
        "sub-01_task-rest_space-fsLR_den-91k_bold.dtseries.nii",
    )
    files["gifti_file"] = os.path.join(
        func_dir,
        "sub-01_task-rest_hemi-L_space-fsaverage5_bold.func.gii",
    )
    files["brain_mask_file"] = os.path.join(
        func_dir,
        "sub-01_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
    )
    files["confounds_file"] = os.path.join(
        func_dir,
        "sub-01_task-rest_desc-confounds_timeseries.tsv",
    )
    files["t1w_to_template_xform"] = os.path.join(
        anat_dir,
        "sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    )
    files["template_to_t1w_xform"] = os.path.join(
        anat_dir,
        "sub-01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
    )
    files["t1w_to_native_xform"] = os.path.join(
        func_dir,
        "sub-01_task-rest_from-T1w_to-scanner_mode-image_xfm.txt",
    )
    files["boldref"] = os.path.join(
        func_dir,
        "sub-01_task-rest_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz",
    )

    return files


@pytest.fixture
def fmriprep_without_freesurfer_data(datasets):
    """Collect a list of fmriprepwithoutfreesurfer files that will be used by misc. tests."""
    subj_dir = os.path.join(datasets["fmriprep_without_freesurfer"], "sub-01")
    func_dir = os.path.join(subj_dir, "func")

    files = {}
    files["nifti_file"] = os.path.join(
        func_dir,
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    files["brain_mask_file"] = os.path.join(
        func_dir,
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )
    files["confounds_file"] = os.path.join(
        func_dir,
        "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv",
    )
    files["boldref"] = os.path.join(
        func_dir,
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz",
    )

    return files
