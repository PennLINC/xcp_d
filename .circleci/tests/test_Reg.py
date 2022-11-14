"""Tests for regression methods."""
import os

import numpy as np
import scipy

from xcp_d.interfaces.regression import Regress
from xcp_d.utils.confounds import load_confound_matrix
from xcp_d.utils.write_save import read_ndata


def test_regression_nifti(data_dir, tmp_path_factory):
    """Test NIFTI regression."""
    data_dir = os.path.join(data_dir, "fmriprepwithfreesurfer")
    temp_dir = tmp_path_factory.mktemp("test_regression_nifti")

    # Specify inputs
    TR = 0.5
    in_file = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    confounds = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-2_desc-confounds_timeseries.tsv"
    )
    mask = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )

    # Read in confounds. Confounds must be selected before running Regress.
    df = load_confound_matrix(confound_tsv=confounds, params="36P")
    assert df.shape[1] == 36
    selected_confounds_file = os.path.join(temp_dir, "temp.tsv")
    df.to_csv(selected_confounds_file, sep="\t", index=False)

    # Run regression
    regression = Regress(
        mask=mask,
        in_file=in_file,
        original_file=in_file,
        confounds=selected_confounds_file,
        TR=TR,
        params="36P",
    )
    results = regression.run(cwd=temp_dir)

    # Loop through each column in the confounds matrix, creating a list of
    # regressors for correlation
    list_of_regressors = []
    for column in df:
        list_of_regressors.append(df[column].tolist())

    regressed_file_data = read_ndata(results.outputs.res_file, mask)

    # Picking a random voxel...
    # Correlate regressed and unregressed image with confounds
    regressed = regressed_file_data[5, :]
    regressed_correlations = []
    for regressor in list_of_regressors:
        regressor = np.array(regressor)
        regressor[~np.isfinite(regressor)] = 0
        r, p = scipy.stats.pearsonr(regressor, regressed)
        regressed_correlations.append(abs(r))
    # The strongest correlation should be less than 0.01
    print(max(regressed_correlations))
    assert (max(regressed_correlations)) < 0.01


def test_regression_cifti(data_dir, tmp_path_factory):
    """Test CIFTI regression."""
    # Specify inputs
    data_dir = os.path.join(data_dir, "fmriprepwithfreesurfer")
    temp_dir = tmp_path_factory.mktemp("test_regression_cifti")

    TR = 0.5
    in_file = os.path.join(
        data_dir,
        "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii",
    )
    confounds = os.path.join(
        data_dir,
        "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv",
    )
    mask = os.path.join(
        data_dir,
        "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    )

    # Read in confounds. Confounds must be selected before running Regress.
    df = load_confound_matrix(confound_tsv=confounds, params="36P")
    assert df.shape[1] == 36
    selected_confounds_file = os.path.join(temp_dir, "temp.tsv")
    df.to_csv(selected_confounds_file, sep="\t", index=False)

    # Run regression
    regression = Regress(
        mask=mask,
        in_file=in_file,
        original_file=in_file,
        confounds=selected_confounds_file,
        TR=TR,
        params="36P",
    )
    regression.base_dir = temp_dir
    results = regression.run(cwd=temp_dir)

    # Loop through each column in the confounds matrix, creating a list of
    # regressors for correlation
    list_of_regressors = []
    for column in df:
        list_of_regressors.append(df[column].tolist())

    regressed_file_data = read_ndata(results.outputs.res_file, mask)

    # Picking a random voxel...
    # Correlate regressed and unregressed image with confounds
    regressed = regressed_file_data[5, :]
    regressed_correlations = []
    for regressor in list_of_regressors:
        regressor = np.array(regressor)
        regressor[~np.isfinite(regressor)] = 0
        r, p = scipy.stats.pearsonr(regressor, regressed)
        regressed_correlations.append(abs(r))
    # The strongest correlation should be less than 0.01
    print((regressed_correlations))
    assert (max(regressed_correlations)) < 0.01
