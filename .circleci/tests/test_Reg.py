"""Tests for regression methods."""
import os

import numpy as np
import pandas as pd
import scipy

from xcp_d.interfaces.regression import Regress
from xcp_d.utils.write_save import read_ndata


def test_Reg_Nifti(data_dir):
    """Test NIFTI regression."""
    data_dir = os.path.join(data_dir,
                            "data/fmriprepwithfreesurfer") 
    #  Specify inputs
    in_file = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    confounds = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-2_desc-confounds_timeseries.tsv"
    )
    TR = 0.5
    mask = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )
    # Run regression
    test_nifti = Regress(
        mask=mask,
        in_file=in_file,
        original_file=in_file,
        confounds=confounds,
        TR=TR,
        params="36P",
    )
    results = test_nifti.run()

    # Read in_file and regression results in, but ignore the 'Unnamed:0' column if present
    df = pd.read_table(results.outputs.confound_matrix)
    if df.shape[1] > 36:  # If that column is present
        df = pd.read_table(results.outputs.confound_matrix, index_col=[0])
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
    assert (max(regressed_correlations)) < 0.01


def test_Reg_Cifti(data_dir):
    """Test CIFTI regression."""
    # Specify inputs
    data_dir = os.path.join(data_dir,
                            "data/fmriprepwithfreesurfer") 
    in_file = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
    )
    confounds = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
    )
    TR = 0.5
    mask = (
        data_dir + "/fmriprep/sub-colornest001/ses-1/func/"
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    )
    # Run regression
    test_cifti = Regress(
        mask=mask,
        in_file=in_file,
        original_file=in_file,
        confounds=confounds,
        TR=TR,
        params="36P",
    )
    results = test_cifti.run()

    # Read in_file and regression results in, but ignore the 'Unnamed:0' column if present
    df = pd.read_table(results.outputs.confound_matrix)
    if df.shape[1] > 36:  # If that column is present
        df = pd.read_table(results.outputs.confound_matrix, index_col=[0])

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
    assert (max(regressed_correlations)) < 0.01
