# import os.path as op
# import nibabel as nb
from xcp_d.interfaces import regress
from xcp_d.utils import read_ndata, write_ndata
import numpy as np
import scipy


def test_Reg_Nifti(data_dir):
    in_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
    TR = 0.5
    mask = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    in_file_data = read_ndata(in_file, mask)  # Read in data in format voxels*timepoints

    # Generate linear noise
    ntimepoints = in_file_data.shape[1]
    linear_drift = (abs(np.arange(ntimepoints))).astype(float)
    linear_drift *= np.mean(in_file_data[5, :])  # Scale it to be close to the original signal
    in_file_edited = in_file_data
    in_file_edited[5, :] = linear_drift  # Add this linear noise to the edited file
    # Find correlation between linear noise
    r1, p1 = scipy.stats.pearsonr(linear_drift, in_file_data[5, :])
    # and original input file
    # Write file back in
    output_file_name = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "edited_sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym" \
        "_desc-preproc_bold.nii.gz"
    write_ndata(data_matrix=in_file_edited, mask=mask,
                template=in_file, filename=output_file_name, tr=TR)
    in_file_edited = output_file_name
    # Run regression on the file with the linear noise
    test_nifti = regress(mask=mask, in_file=in_file_edited,
                         original_file=in_file, confounds=confounds, TR=TR)
    results = test_nifti.run()
    # Read the output file
    out_file_data = read_ndata(results.outputs.res_file, mask)
    # See how the output file  correlates with the linear_drift
    r2, p2 = scipy.stats.pearsonr(linear_drift, out_file_data[5, :])
    assert r1 > r2  # Has correlation with noise decreased after regression?


def test_Reg_Cifti(data_dir):
    in_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
    confounds = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
    TR = 0.5
    mask = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    in_file_data = read_ndata(in_file, mask)  # Read in data in format voxels*timepoints
    # Generate linear noise
    ntimepoints = in_file_data.shape[1]
    linear_drift = (abs(np.arange(ntimepoints))).astype(float)
    linear_drift *= np.mean(in_file_data[5, :])  # Scale it to be close to the original signal
    in_file_edited = in_file_data
    in_file_edited[5, :] = linear_drift  # Add this linear noise to the edited file
    # Find correlation between linear noise
    r1, p1 = scipy.stats.pearsonr(linear_drift, in_file_data[5, :])
    # and original input file
    # Write file back in
    output_file_name = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "edited_sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
    write_ndata(data_matrix=in_file_edited, mask=mask,
                template=in_file, filename=output_file_name, tr=TR)
    in_file_edited = output_file_name
    # Run regression on the file with the linear noise
    test_cifti = regress(mask=mask, in_file=in_file,
                         original_file=in_file, confounds=confounds, TR=TR)
    results = test_cifti.run()
    out_file_data = read_ndata(results.outputs.res_file, mask)
    # See how the output file  correlates with the linear_drift
    r2, p2 = scipy.stats.pearsonr(linear_drift, out_file_data[5, :])
    assert r1 > r2  # Has correlation with noise decreased after regression?