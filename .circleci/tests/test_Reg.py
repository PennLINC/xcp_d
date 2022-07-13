
# # import os.path as op
# # import nibabel as nb
# from xcp_d.interfaces import regress
# from xcp_d.utils import read_ndata
# import pandas as pd
# import scipy


# def test_Reg_Nifti(data_dir):
#     #  Specify inputs
#     in_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#     confounds = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
#     TR = 0.5
#     mask = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
#     # Run regression
#     test_nifti = regress(mask=mask, in_file=in_file,
#                          original_file=in_file, confounds=confounds, TR=TR)
#     results = test_nifti.run()

#     # Read in_file and regression results in, but ignore the 'Unnamed:0' column if present
#     df = pd.read_table(results.outputs.confound_matrix)
#     if df.shape[1] > 36:  # If that column is present
#         df = pd.read_table(results.outputs.confound_matrix, index_col=[0])

#     # Loop through each column in the confounds matrix, creating a list of
#     # regressors for correlation
#     list_of_regressors = []
#     for column in df:
#         list_of_regressors.append(df[column].tolist())

#     in_file_data = read_ndata(in_file, mask)
#     regressed_file_data = read_ndata(results.outputs.res_file, mask)

#     # Picking a random voxel...
#     # Correlate regressed and unregressed image with confounds
#     regressed = regressed_file_data[5, :]
#     regressed_correlations = []
#     unregressed = in_file_data[5, :]
#     unregressed_correlations = []
#     for regressor in list_of_regressors:
#         r, p = scipy.stats.pearsonr(regressor, regressed)
#         regressed_correlations.append((r))
#         r1, p1 = scipy.stats.pearsonr(regressor, unregressed)
#         unregressed_correlations.append((r1))



# def test_Reg_Cifti(data_dir):
#     # Specify inputs
#     in_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
#     confounds = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
#     TR = 0.5
#     mask = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
#     # Run regression
#     test_cifti = regress(mask=mask, in_file=in_file,
#                          original_file=in_file, confounds=confounds, TR=TR)
#     results = test_cifti.run()

#     # Read in_file and regression results in, but ignore the 'Unnamed:0' column if present
#     df = pd.read_table(results.outputs.confound_matrix)
#     if df.shape[1] > 36:  # If that column is present
#         df = pd.read_table(results.outputs.confound_matrix, index_col=[0])

#     # Loop through each column in the confounds matrix, creating a list of
#     # regressors for correlation
#     list_of_regressors = []
#     for column in df:
#         list_of_regressors.append(df[column].tolist())

#     in_file_data = read_ndata(in_file, mask)
#     regressed_file_data = read_ndata(results.outputs.res_file, mask)

#     # Picking a random voxel...
#     # Correlate regressed and unregressed image with confounds
#     regressed = regressed_file_data[5, :]
#     regressed_correlations = []
#     unregressed = in_file_data[5, :]
#     unregressed_correlations = []
#     for regressor in list_of_regressors:
#         r, p = scipy.stats.pearsonr(regressor, regressed)
#         regressed_correlations.append((r))
#         r1, p1 = scipy.stats.pearsonr(regressor, unregressed)
#         unregressed_correlations.append((r1))


