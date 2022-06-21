from xcp_d.workflow.postprocessing import init_censoring_wf
from xcp_d.interfaces.prepostcleaning import CensorScrub
import os
import pandas as pd


def test_fd_interface():  # Checking results
    input_file = '/Users/kahinim/Desktop/FD_test/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
    mask = '/Users/kahinim/Desktop/FD_test/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz'
    confounds_tsv = '/Users/kahinim/Desktop/FD_test/fmriprep/sub-99964/ses-10105/func/sub-99964_ses-10105_task-rest_acq-singleband_desc-confounds_timeseries.tsv'
    df = pd.read_table(confounds_tsv)
    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]
    df.loc[4:6, "trans_y"] = [7, 8, 9]
    df.loc[7:9, "trans_z"] = [12, 8, 9]
    tmpdir = '/Users/kahinim/Desktop/FD_test'  # So we can see results
    os.chdir(tmpdir)
    confounds_tsv = "edited_" + confounds_tsv.split('/func/')[1]  # Rename with same convention as initial confounds tsv
    df.to_csv(confounds_tsv, sep='\t', index=False)

    # Run workflow
    cscrub = CensorScrub()
    cscrub.inputs.in_file = input_file
    cscrub.inputs.TR = 0.8
    cscrub.inputs.fd_thresh = 0.5
    cscrub.inputs.fmriprep_confounds = confounds_tsv
    cscrub.inputs.mask_file = mask
    cscrub.inputs.time_todrop = 0
    cscrub.inputs.head_radius = 50
    cscrub.run()

    # Confirming that the df values are changed as expected
    confounds_df = pd.read_table(confounds_tsv)
    assert confounds_df.loc[1:3, "trans_x"].tolist() == [6, 8, 9]
    assert confounds_df.loc[4:6, "trans_y"].tolist() == [7, 8, 9]
    assert confounds_df.loc[7:9, "trans_z"].tolist() == [12, 8, 9]


test_fd_interface()  # Call the function


test_fd_interface_dummy_signal()  # Call the function
