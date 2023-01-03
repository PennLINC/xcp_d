"""Tests for regression methods."""
import os

import numpy as np
import scipy

from xcp_d.interfaces.regression import Regress
from xcp_d.utils.confounds import load_confound_matrix
from xcp_d.utils.write_save import read_ndata


def test_regression_nifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test NIFTI regression."""
    temp_dir = tmp_path_factory.mktemp("test_regression_nifti")

    # Specify inputs
    TR = 0.5
    in_file = fmriprep_with_freesurfer_data["nifti_file"]
    mask = fmriprep_with_freesurfer_data["brain_mask_file"]

    # Read in confounds. Confounds must be selected before running Regress.
    df = load_confound_matrix(img_file=in_file, params="36P")
    assert df.shape[1] == 36
    selected_confounds_file = os.path.join(temp_dir, "temp.tsv")
    df.to_csv(selected_confounds_file, sep="\t", index=False)

    # Run regression
    regression = Regress(
        mask=mask,
        in_file=in_file,
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
        r, _ = scipy.stats.pearsonr(regressor, regressed)
        regressed_correlations.append(abs(r))
    # The strongest correlation should be less than 0.01
    print(max(regressed_correlations))
    assert (max(regressed_correlations)) < 0.01


def test_regression_cifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test CIFTI regression."""
    # Specify inputs
    temp_dir = tmp_path_factory.mktemp("test_regression_cifti")

    TR = 0.5
    in_file = fmriprep_with_freesurfer_data["cifti_file"]
    mask = fmriprep_with_freesurfer_data["brain_mask_file"]

    # Read in confounds. Confounds must be selected before running Regress.
    df = load_confound_matrix(img_file=in_file, params="36P")
    assert df.shape[1] == 36
    selected_confounds_file = os.path.join(temp_dir, "temp.tsv")
    df.to_csv(selected_confounds_file, sep="\t", index=False)

    # Run regression
    regression = Regress(
        mask=mask,
        in_file=in_file,
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
        r, _ = scipy.stats.pearsonr(regressor, regressed)
        regressed_correlations.append(abs(r))
    # The strongest correlation should be less than 0.01
    print((regressed_correlations))
    assert (max(regressed_correlations)) < 0.01
