"""Fixtures for the CircleCI tests."""

import base64
import os

import pytest


def pytest_addoption(parser):
    """Collect pytest parameters for running tests."""
    parser.addoption(
        "--working_dir",
        action="store",
        default=(
            "/usr/local/miniconda/lib/python3.10/site-packages/xcp_d/xcp_d/tests/data/test_data/"
            "run_pytests/work"
        ),
    )
    parser.addoption(
        "--data_dir",
        action="store",
        default=(
            "/usr/local/miniconda/lib/python3.10/site-packages/xcp_d/xcp_d/tests/data/test_data"
        ),
    )
    parser.addoption(
        "--output_dir",
        action="store",
        default=(
            "/usr/local/miniconda/lib/python3.10/site-packages/xcp_d/xcp_d/tests/data/test_data/"
            "run_pytests/out"
        ),
    )


# Set up the commandline options as fixtures
@pytest.fixture(scope="session")
def data_dir(request):
    """Grab data directory."""
    return request.config.getoption("--data_dir")


@pytest.fixture(scope="session")
def working_dir(request):
    """Grab working directory."""
    workdir = request.config.getoption("--working_dir")
    os.makedirs(workdir, exist_ok=True)
    return workdir


@pytest.fixture(scope="session")
def output_dir(request):
    """Grab output directory."""
    outdir = request.config.getoption("--output_dir")
    os.makedirs(outdir, exist_ok=True)
    return outdir


@pytest.fixture(scope="session")
def datasets(data_dir):
    """Locate downloaded datasets."""
    dsets = {}
    dsets["ds001419"] = os.path.join(data_dir, "ds001419-fmriprep")
    dsets["pnc"] = os.path.join(data_dir, "pnc")
    dsets["nibabies"] = os.path.join(data_dir, "nibabies/derivatives/nibabies")
    dsets["fmriprep_without_freesurfer"] = os.path.join(
        data_dir,
        "fmriprepwithoutfreesurfer",
    )
    return dsets


@pytest.fixture(scope="session")
def ds001419_data(datasets):
    """Collect a list of files from ds001419 that will be used by misc. tests."""
    subj_dir = os.path.join(datasets["ds001419"], "sub-01")
    func_dir = os.path.join(subj_dir, "func")
    anat_dir = os.path.join(subj_dir, "anat")
    if not os.path.isdir(subj_dir):
        raise Exception(os.listdir(datasets["ds001419"]))

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
    files["confounds_json"] = os.path.join(
        func_dir,
        "sub-01_task-rest_desc-confounds_timeseries.json",
    )
    files["anat_to_template_xfm"] = os.path.join(
        anat_dir,
        "sub-01_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    )
    files["template_to_anat_xfm"] = os.path.join(
        anat_dir,
        "sub-01_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
    )
    files["boldref"] = os.path.join(
        func_dir,
        "sub-01_task-rest_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz",
    )
    files["boldref_t1w"] = os.path.join(func_dir, "sub-01_task-rest_space-T1w_boldref.nii.gz")
    files["t1w"] = os.path.join(anat_dir, "sub-01_desc-preproc_T1w.nii.gz")
    files["t1w_mni"] = os.path.join(
        anat_dir,
        "sub-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz",
    )
    files["anat_dseg"] = os.path.join(anat_dir, "sub-01_desc-aseg_dseg.nii.gz")

    return files


@pytest.fixture(scope="session")
def pnc_data(datasets):
    """Collect a list of files from pnc that will be used by misc. tests."""
    subj_dir = os.path.join(datasets["pnc"], "sub-1648798153", "ses-PNC1")
    func_dir = os.path.join(subj_dir, "func")
    anat_dir = os.path.join(subj_dir, "anat")

    files = {}
    files["nifti_file"] = os.path.join(
        func_dir,
        (
            "sub-1648798153_ses-PNC1_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_"
            "desc-preproc_bold.nii.gz"
        ),
    )
    files["cifti_file"] = os.path.join(
        func_dir,
        "sub-1648798153_ses-PNC1_task-rest_acq-singleband_space-fsLR_den-91k_bold.dtseries.nii",
    )
    files["brain_mask_file"] = os.path.join(
        func_dir,
        "sub-1648798153_ses-PNC1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
    )
    files["confounds_file"] = os.path.join(
        func_dir,
        "sub-1648798153_ses-PNC1_task-rest_desc-confounds_timeseries.tsv",
    )
    files["confounds_json"] = os.path.join(
        func_dir,
        "sub-1648798153_ses-PNC1_task-rest_desc-confounds_timeseries.json",
    )
    files["anat_to_template_xfm"] = os.path.join(
        anat_dir,
        "sub-1648798153_ses-PNC1_acq-refaced_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
    )
    files["template_to_anat_xfm"] = os.path.join(
        anat_dir,
        "sub-1648798153_ses-PNC1_acq-refaced_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
    )
    files["boldref"] = os.path.join(
        func_dir,
        "sub-1648798153_ses-PNC1_task-rest_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz",
    )
    files["boldref_t1w"] = os.path.join(
        func_dir,
        "sub-1648798153_ses-PNC1_task-rest_space-T1w_boldref.nii.gz",
    )
    files["t1w"] = os.path.join(anat_dir, "sub-1648798153_acq-refaced_desc-preproc_T1w.nii.gz")
    files["t1w_mni"] = os.path.join(
        anat_dir,
        (
            "sub-1648798153_ses-PNC1_acq-refaced_space-MNI152NLin2009cAsym_"
            "res-2_desc-preproc_T1w.nii.gz"
        ),
    )
    files["anat_dseg"] = os.path.join(
        anat_dir,
        "sub-1648798153_ses-PNC1_acq-refaced_desc-aseg_dseg.nii.gz",
    )

    return files


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session", autouse=True)
def fslicense(working_dir):
    """Set the FreeSurfer license as an environment variable."""
    FS_LICENSE = os.path.join(working_dir, "license.txt")
    os.environ["FS_LICENSE"] = FS_LICENSE
    LICENSE_CODE = (
        "bWF0dGhldy5jaWVzbGFrQHBzeWNoLnVjc2IuZWR1CjIwNzA2CipDZmVWZEg1VVQ4clkKRlNCWVouVWtlVElDdwo="
    )
    with open(FS_LICENSE, "w") as f:
        f.write(base64.b64decode(LICENSE_CODE).decode())


@pytest.fixture(scope="session")
def base_config():
    from xcp_d.tests.tests import mock_config

    return mock_config


@pytest.fixture(scope="session")
def base_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Test parser")
    return parser
