#!/usr/bin/env python

"""
This file is an example of running pytests either locally or on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.



"""
import os.path as op
import nibabel as nb
from xcp_d.interfaces.prepostcleaning import RemoveTR
import os
import pandas as pd

def test_data_availability(data_dir, working_dir, output_dir):
    """Makes sure that we have access to all the testing data
    """
    assert op.exists(output_dir)
    assert op.exists(working_dir)
    assert op.exists(data_dir)
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    assert op.exists(boldfile)


def test_RemoveTR_nifti(data_dir):
    # Test RemoveTR() for NIFTI input data
    from xcp_d.interfaces.prepostcleaning import RemoveTR
    import pandas as pd

    # Define inputs
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"

    # Find the original number of volumes acc. to nifti & confounds timeseries
    original_confounds = pd.read_csv(confounds_file, sep="\t")
    original_nvols_nifti = nb.load(boldfile).get_fdata().shape[3]

    # Test a nifti file with 0 volumes to remove
    remove_nothing = RemoveTR(
        bold_file=boldfile,
        fmriprep_confounds_file=confounds_file,
        initial_volumes_to_drop=0)
    results = remove_nothing.run()
    undropped_confounds = pd.read_table(results.outputs.fmriprep_confounds_file_dropped_TR)
    # Were the files created?
    assert op.exists(results.outputs.bold_file_dropped_TR)
    assert op.exists(results.outputs.fmriprep_confounds_file_dropped_TR)
    # Have the confounds stayed the same shape?
    assert undropped_confounds.shape == original_confounds.shape
    # Has the nifti stayed the same shape?
    assert nb.load(results.
                   outputs.bold_file_dropped_TR).get_fdata().shape[3] == original_nvols_nifti

    # Test a nifti file with 'n' volumes to remove
    for n in range(0, original_nvols_nifti-1):  # Testing all n values till
        # original_nvols_nifti - 1
        remove_n_vols = RemoveTR(
            bold_file=boldfile,
            fmriprep_confounds_file=confounds_file,
            initial_volumes_to_drop=n)
        results = remove_n_vols.run()
        dropped_confounds = pd.read_table(results.outputs.fmriprep_confounds_file_dropped_TR)
        # Were the files created?
        assert op.exists(results.outputs.bold_file_dropped_TR)
        assert op.exists(results.outputs.fmriprep_confounds_file_dropped_TR)
        # Have the confounds changed correctly?
        assert dropped_confounds.shape[0] == original_confounds.shape[0] - n
        # Has the nifti changed correctly?
        try:
            assert nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[3]\
                == original_nvols_nifti - n
        except Exception as exc:
            exc = nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[3]
            print("Tests failing at N = {}.".format(n))
            raise Exception("Number of volumes in dropped nifti is {}.".format(exc))


def test_RemoveTR_cifti(data_dir):
    # Test RemoveTR() for CIFTI input data
    from xcp_d.interfaces.prepostcleaning import RemoveTR
    import pandas as pd
    # Define inputs
    boldfile = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii"
    confounds_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"

    # Find the original number of volumes acc. to cifti & confounds timeseries
    original_confounds = pd.read_csv(confounds_file, sep="\t")
    original_nvols_cifti = nb.load(boldfile).get_fdata().shape[0]

    # Test a cifti file with 0 volumes to remove
    remove_nothing = RemoveTR(
        bold_file=boldfile,
        fmriprep_confounds_file=confounds_file,
        initial_volumes_to_drop=0)
    results = remove_nothing.run()
    undropped_confounds = pd.read_table(results.outputs.fmriprep_confounds_file_dropped_TR)
    # Were the files created?
    assert op.exists(results.outputs.bold_file_dropped_TR)
    assert op.exists(results.outputs.fmriprep_confounds_file_dropped_TR)
    # Have the confounds stayed the same shape?
    assert undropped_confounds.shape == original_confounds.shape
    # Has the cifti stayed the same shape?
    assert nb.load(results.outputs.bold_file_dropped_TR).get_fdata(
    ).shape[0] == original_nvols_cifti

    # Test a cifti file with 'n' volumes to remove
    for n in range(0, original_nvols_cifti-1):  # Testing all n values till
        # original_nvols_cifti - 1
        remove_n_vols = RemoveTR(
            bold_file=boldfile,
            fmriprep_confounds_file=confounds_file,
            initial_volumes_to_drop=n)
#         print(n)
        results = remove_n_vols.run()
        dropped_confounds = pd.read_table(results.outputs.fmriprep_confounds_file_dropped_TR)
        # Were the files created?
        assert op.exists(results.outputs.bold_file_dropped_TR)
        assert op.exists(results.outputs.fmriprep_confounds_file_dropped_TR)
        # Have the confounds changed correctly?
        assert dropped_confounds.shape[0] == original_confounds.shape[0] - n
        # Has the cifti changed correctly?
        try:
            assert nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[0]\
                == original_nvols_cifti - n
        except Exception as exc:
            exc = nb.load(results.outputs.bold_file_dropped_TR).get_fdata().shape[0]
            print("Tests failing at N = {}.".format(n))
            raise Exception("Number of volumes in dropped cifti is {}.".format(exc))

# Testing with CUSTOM CONFOUNDS

# # Note: I had to test this locally as I don't have the permissions to share the
# # data I used here at the moment. 
# from xcp_d.interfaces.prepostcleaning import RemoveTR
# def test_fd_interface_cifti_custom(data_dir):  # Checking results
#     boldfile = data_dir + '/fmriprep/sub-colornest001/ses-1/func/sub-col'\
#         'ornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii'
#     confounds_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
#         "sub-colornest001_ses-1_task-rest_run-1_desc-confounds_timeseries.tsv"
#     custom_confounds_tsv = data_dir + "sub-SLmobile_ses-CAMRISSatterthwaite_task-rest_acq-multiband_desc-custom_timeseries.tsv"
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
#     remvtr = RemoveTR()
#     remvtr.inputs.bold_file = boldfile
#     remvtr.inputs.fmriprep_confounds_file = confounds_tsv
#     remvtr.inputs.custom_confounds = custom_confounds_tsv
#     remvtr.initial_volumes_to_drop = 5
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
#         raise Exception("Sorry, the shapes are: {}.".format(exc))



# def test_fd_interface_nifti_custom(data_dir):  # Checking results
#     boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#     confounds_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
#         "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
#     custom_confounds_tsv = data_dir + "sub-SLmobile_ses-CAMRISSatterthwaite_task-rest_acq-multiband_desc-custom_timeseries.tsv"
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
#     remvtr = RemoveTR()
#     remvtr.inputs.bold_file = boldfile
#     remvtr.inputs.fmriprep_confounds_file = confounds_tsv
#     remvtr.inputs.custom_confounds = custom_confounds_tsv
#     remvtr.initial_volumes_to_drop = 5
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
#         raise Exception("Sorry, the shapes are: {}.".format(exc))