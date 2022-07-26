
from xcp_d.interfaces.prepostcleaning import CensorScrub, interpolate
import os
import pandas as pd
import tempfile
import nibabel as nb


def test_interpolate_cifti(data_dir):  # Checking results - first must censor file
    boldfile = data_dir + '/fmriprep/sub-colornest001/ses-1/func/sub-col'\
        'ornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii'
    confounds_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
    df = pd.read_table(confounds_file)
    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]  # Let's ensure some values are censored
    tmpdir = tempfile.mkdtemp()  # edit this if you want to see the edited confounds
    # on your Desktop, etc.
    os.chdir(tmpdir)
    # Rename with same convention as initial confounds tsv
    confounds_tsv = "edited_" + confounds_file.split('/func/')[1]
    df.to_csv(confounds_tsv, sep='\t', index=False, header=True)

    # Run workflow
    cscrub = CensorScrub()
    cscrub.inputs.in_file = boldfile
    cscrub.inputs.TR = 0.8
    cscrub.inputs.fd_thresh = 0.2
    cscrub.inputs.motion_filter_type = 'None'
    cscrub.inputs.motion_filter_order = 4
    cscrub.inputs.low_freq = 0
    cscrub.inputs.high_freq = 0
    cscrub.inputs.fmriprep_confounds_file = confounds_tsv
    cscrub.inputs.head_radius = 50
    results = cscrub.run()

    # Write out censored files
    censored_file = nb.load(results.outputs.bold_censored)
    # Start testing interpolation - feed in input
    interpolation = interpolate()
    interpolation.inputs.tmask = results.outputs.tmask
    interpolation.inputs.in_file = results.outputs.bold_censored
    interpolation.inputs.bold_file = boldfile
    interpolation.inputs.TR = 1
    interpolation.inputs.mask_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    results = interpolation.run()
    interpolated_file = nb.load(results.outputs.bold_interpolated).get_fdata()
    original_file = nb.load(boldfile).get_fdata()
    # assert all values were interpolated in, and censoring took place
    assert censored_file.shape != original_file.shape
    assert interpolated_file.shape == original_file.shape


def test_interpolate_nifti(data_dir):  # Checking results - first must censor file
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
    df = pd.read_table(confounds_file)
    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]  # Let's ensure some values are censored
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    # Rename with same convention as initial confounds tsv
    confounds_tsv = "edited_" + confounds_file.split('/func/')[1]
    df.to_csv(confounds_tsv, sep='\t', index=False, header=True)

    # Run workflow
    cscrub = CensorScrub()
    cscrub.inputs.in_file = boldfile
    cscrub.inputs.TR = 0.8
    cscrub.inputs.fd_thresh = 0.2
    cscrub.inputs.motion_filter_type = 'None'
    cscrub.inputs.motion_filter_order = 4
    cscrub.inputs.low_freq = 0
    cscrub.inputs.high_freq = 0
    cscrub.inputs.fmriprep_confounds_file = confounds_tsv
    cscrub.inputs.head_radius = 50
    results = cscrub.run()
    # Write out censored files
    censored_file = nb.load(results.outputs.bold_censored)
    # Start testing interpolation - feed in input
    interpolation = interpolate()
    interpolation.inputs.tmask = results.outputs.tmask
    interpolation.inputs.in_file = results.outputs.bold_censored
    interpolation.inputs.bold_file = boldfile
    interpolation.inputs.TR = 1
    interpolation.inputs.mask_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    results = interpolation.run()
    interpolated_file = nb.load(results.outputs.bold_interpolated).get_fdata()
    original_file = nb.load(boldfile).get_fdata()
    # assert all values were interpolated in, and censoring took place
    assert censored_file.shape != original_file.shape
    assert interpolated_file.shape == original_file.shape
