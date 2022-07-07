# # import os.path as op
# # import nibabel as nb
# from xcp_d.interfaces import regress
# data_dir = '/Users/kahinim/Desktop/xcp_test/data'


# def test_Reg_Nifti(data_dir):
#     in_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#     confounds = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
#     TR = 0.5
#     mask = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
#     test_nifti = regress(mask=mask, in_file=in_file,
#                          original_file=in_file, confounds=confounds, TR=TR)
#     results = test_nifti.run()


# test_Reg_Nifti(data_dir)


# def test_Reg_Cifti(data_dir):
#     in_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
#     confounds = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
#     TR = 0.5
#     mask = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
#     test_cifti = regress(mask=mask, in_file=in_file,
#                          original_file=in_file, confounds=confounds, TR=TR)
#     results = test_cifti.run()
