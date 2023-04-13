"""Tests for framewise displacement calculation."""
import os

import pandas as pd

from xcp_d.interfaces.censoring import FlagMotionOutliers


def test_fd_interface_cifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Check results."""
    tmpdir = tmp_path_factory.mktemp("test_fd_interface_cifti")
    confounds_file = fmriprep_with_freesurfer_data["confounds_file"]

    df = pd.read_table(confounds_file)

    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]
    df.loc[4:6, "trans_y"] = [7, 8, 9]
    df.loc[7:9, "trans_z"] = [12, 8, 9]

    # Rename with same convention as initial confounds tsv
    confounds_tsv = os.path.join(tmpdir, f"edited_{confounds_file.split('/func/')[1]}")
    df.to_csv(confounds_tsv, sep="\t", index=False, header=True)

    # Run workflow
    cscrub = FlagMotionOutliers()
    cscrub.inputs.TR = 0.8
    cscrub.inputs.fd_thresh = 0.2
    cscrub.inputs.motion_filter_type = None
    cscrub.inputs.motion_filter_order = 4
    cscrub.inputs.band_stop_min = 0
    cscrub.inputs.band_stop_max = 0
    cscrub.inputs.fmriprep_confounds_file = confounds_tsv
    cscrub.inputs.head_radius = 50
    cscrub.run(cwd=tmpdir)

    # Confirming that the df values are changed as expected
    confounds_df = pd.read_table(confounds_tsv)
    assert confounds_df.loc[1:3, "trans_x"].tolist() == [6, 8, 9]
    assert confounds_df.loc[4:6, "trans_y"].tolist() == [7, 8, 9]
    assert confounds_df.loc[7:9, "trans_z"].tolist() == [12, 8, 9]


def test_fd_interface_nifti(data_dir, tmp_path_factory):
    """Check results."""
    tmpdir = tmp_path_factory.mktemp("test_fd_interface_nifti")

    data_dir = os.path.join(data_dir, "fmriprepwithoutfreesurfer/fmriprep/")
    confounds_file = os.path.join(
        data_dir,
        "sub-01/func",
        "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv",
    )
    df = pd.read_table(confounds_file)

    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]
    df.loc[4:6, "trans_y"] = [7, 8, 9]
    df.loc[7:9, "trans_z"] = [12, 8, 9]

    # Rename with same convention as initial confounds tsv
    confounds_tsv = os.path.join(tmpdir, f"edited_{confounds_file.split('/func/')[1]}")
    df.to_csv(confounds_tsv, sep="\t", index=False, header=True)

    # Run workflow
    cscrub = FlagMotionOutliers()
    cscrub.inputs.TR = 0.8
    cscrub.inputs.fd_thresh = 0.2
    cscrub.inputs.motion_filter_type = None
    cscrub.inputs.motion_filter_order = 4
    cscrub.inputs.band_stop_min = 0
    cscrub.inputs.band_stop_max = 0
    cscrub.inputs.fmriprep_confounds_file = confounds_tsv
    cscrub.inputs.head_radius = 50
    cscrub.run(cwd=tmpdir)

    # Confirming that the df values are changed as expected
    confounds_df = pd.read_table(confounds_tsv)
    assert confounds_df.loc[1:3, "trans_x"].tolist() == [6, 8, 9]
    assert confounds_df.loc[4:6, "trans_y"].tolist() == [7, 8, 9]
    assert confounds_df.loc[7:9, "trans_z"].tolist() == [12, 8, 9]


# Testing with CUSTOM CONFOUNDS

# Note: I had to test this locally as I don't have the permissions to share the
# data I used here at the moment. These tests passed locally.

# def test_fd_interface_cifti_custom(data_dir):  # Checking results
#     boldfile = data_dir + '/fmriprep/sub-colornest001/ses-1/func/sub-col'\
#         'ornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii'
#     confounds_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
#     custom_confounds_tsv = """Put the path to your file here"""
#     df = pd.read_table(confounds_file)
#     # Replace confounds tsv values with values that should be omitted
#     df.loc[1:3, "trans_x"] = [6, 8, 9]
#     df.loc[4:6, "trans_y"] = [7, 8, 9]
#     df.loc[7:9, "trans_z"] = [12, 8, 9]
#     tmpdir = tempfile.mkdtemp
#     os.chdir(tmpdir)
#     # Rename with same convention as initial confounds tsv
#     confounds_tsv = "edited_" + confounds_file.split('/func/')[1]
#     df.to_csv(confounds_tsv, sep='\t', index=False, header=True)

#     # Run workflow
#     cscrub = FlagMotionOutliers()
#     cscrub.inputs.in_file = boldfile
#     cscrub.inputs.TR = 0.8
#     cscrub.inputs.fd_thresh = 0.2
#     cscrub.inputs.motion_filter_type = None
#     cscrub.inputs.band_stop_min = 0
#     cscrub.inputs.band_stop_max = 0
#     cscrub.inputs.fmriprep_confounds_file = confounds_tsv
#     cscrub.inputs.confounds_file = custom_confounds_tsv
#     cscrub.inputs.head_radius = 50
#     results = cscrub.run()
#     # Load in censored image and confounds tsv
#     censored_image = nb.load(results.outputs.bold_censored)
#     censored_confounds_timeseries = pd.read_table(results.outputs.custom_confounds_censored)
#     # Assert the length of the confounds is the same as the nvol of the image
#     try:
#         assert len(censored_confounds_timeseries) == censored_image.get_fdata().shape[0]
#         print(len(censored_confounds_timeseries))
#     except Exception as exc:
#         exc = len(censored_confounds_timeseries), censored_image.get_fdata().shape[0]
#         raise Exception(f"Sorry, the shapes are: {exc}.")


# def test_fd_interface_nifti_custom(data_dir):  # Checking results
#     boldfile = data_dir + "sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#     confounds_file = data_dir + "sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
#     custom_confounds_tsv = """Put in file path here"""
#     df = pd.read_table(confounds_file)
#     # Replace confounds tsv values with values that should be omitted
#     df.loc[1:3, "trans_x"] = [6, 8, 9]
#     df.loc[4:6, "trans_y"] = [7, 8, 9]
#     df.loc[7:9, "trans_z"] = [12, 8, 9]
#     tmpdir = tempfile.mkdtemp
#     os.chdir(tmpdir)
#     # Rename with same convention as initial confounds tsv
#     confounds_tsv = "edited_" + confounds_file.split('/func/')[1]
#     df.to_csv(confounds_tsv, sep='\t', index=False, header=True)

#     # Run workflow
#     cscrub = FlagMotionOutliers()
#     cscrub.inputs.in_file = boldfile
#     cscrub.inputs.TR = 0.8
#     cscrub.inputs.fd_thresh = 0.2
#     cscrub.inputs.motion_filter_type = None
#     cscrub.inputs.band_stop_min = 0
#     cscrub.inputs.band_stop_max = 0
#     cscrub.inputs.fmriprep_confounds_file = confounds_tsv
#     cscrub.inputs.confounds_file = custom_confounds_tsv
#     cscrub.inputs.head_radius = 50
#     results = cscrub.run()
#     # Load in censored image and confounds tsv
#     censored_image = nb.load(results.outputs.bold_censored)
#     censored_confounds_timeseries = pd.read_table(results.outputs.custom_confounds_censored)
#     # Assert the length of the confounds is the same as the nvol of the image
#     try:
#         assert len(censored_confounds_timeseries) == censored_image.get_fdata().shape[3]
#         print(len(censored_confounds_timeseries))
#     except Exception as exc:
#         exc = len(censored_confounds_timeseries), censored_image.get_fdata().shape[3]
#         raise Exception(f"Sorry, the shapes are: {exc}.")
