"""Tests for removing volumes from files.

This file is an example of running pytests either locally or on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.
"""
import os.path as op


def test_data_availability(data_dir, working_dir, output_dir):
    """Make sure that we have access to all the testing data."""
    assert op.exists(output_dir)
    assert op.exists(working_dir)
    assert op.exists(data_dir)
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    assert op.exists(boldfile)


# Testing with CUSTOM CONFOUNDS
# Note: I had to test this locally as I don't have the permissions to share the
# data I used here at the moment.
# def test_fd_interface_cifti_custom(data_dir):  # Checking results
#     boldfile = data_dir + '/fmriprep/sub-colornest001/ses-1/func/sub-col'\
#         'ornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii'
#     confounds_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
#     custom_confounds_tsv = data_dir + "/fmriprep/sub-colornest001/ses-1/func/customcifti.tsv"

#     # Run workflow
#     remvtr = RemoveTR()
#     remvtr.inputs.bold_file = boldfile
#     remvtr.inputs.fmriprep_confounds_file = confounds_tsv
#     remvtr.inputs.custom_confounds = custom_confounds_tsv
#     remvtr.inputs.initial_volumes_to_drop = 5
#     results = remvtr.run()

#     # Load in dropped image and confounds tsv
#     dropped_image = nb.load(results.outputs.bold_file_dropped_TR)
#     dropped_confounds_timeseries = pd.read_table(results.outputs.custom_confounds_dropped)

#     # Assert the length of the confounds is the same as the nvol of the image
#     try:
#         assert len(dropped_confounds_timeseries) == dropped_image.get_fdata().shape[0]
#         print(len(dropped_confounds_timeseries))
#     except Exception as exc:
#         exc = len(dropped_confounds_timeseries), dropped_image.get_fdata().shape[0]
#         raise Exception(f"Sorry, the shapes are: {exc}.")


# def test_fd_interface_nifti_custom(data_dir):  # Checking results
#     boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#     confounds_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
#     custom_confounds_tsv = data_dir + "/withoutfreesurfer/sub-01/func/customnifti.tsv"
#     # Run workflow
#     remvtr = RemoveTR()
#     remvtr.inputs.bold_file = boldfile
#     remvtr.inputs.fmriprep_confounds_file = confounds_tsv
#     remvtr.inputs.custom_confounds = custom_confounds_tsv
#     remvtr.inputs.initial_volumes_to_drop = 5
#     results = remvtr.run()

#     # Load in dropped image and confounds tsv
#     dropped_image = nb.load(results.outputs.bold_file_dropped_TR)
#     dropped_confounds_timeseries = pd.read_table(results.outputs.custom_confounds_dropped)

#     # Assert the length of the confounds is the same as the nvol of the image
#     try:
#         assert len(dropped_confounds_timeseries) == dropped_image.get_fdata().shape[3]
#         print(len(dropped_confounds_timeseries))
#     except Exception as exc:
#         exc = len(dropped_confounds_timeseries), dropped_image.get_fdata().shape[3]
#         raise Exception(f"Sorry, the shapes are: {exc}.")
