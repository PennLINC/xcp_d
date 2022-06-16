#!/usr/bin/env python

"""
This file is an example of running pytests either locally or on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.



"""
import os.path as op
import nibabel as nb

data_dir = '/Users/kahinim/Desktop/xcp_test/data'


def test_data_availability(data_dir, working_dir, output_dir):
    """Makes sure that we have access to all the testing data
    """
    assert op.exists(output_dir)
    assert op.exists(working_dir)
    assert op.exists(data_dir)
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    assert op.exists(boldfile)


def test_removeTR(data_dir):
    from xcp_d.interfaces.prepostcleaning import removeTR
    import pandas as pd

    # Define inputs
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
    mask_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

    # Find the original number of volumes acc. to nifti & confounds timeseries
    original_confounds = pd.read_csv(confounds_file, sep="\t")
    original_nvols_nifti = nb.load(boldfile).get_fdata().shape[3]

    # Test a nifti file with 0 volumes to remove
    remove_nothing = removeTR(
        bold_file=boldfile,
        fmriprep_conf=confounds_file,
        initial_volumes_to_drop=0,
        mask_file=mask_file)
    results = remove_nothing.run()
    uncensored_confounds = pd.read_table(results.outputs.fmrip_confdropTR)
    # Were the files created?
    assert op.exists(results.outputs.bold_file_TR)
    assert op.exists(results.outputs.fmrip_confdropTR)
    # Have the confounds stayed the same shape?
    assert uncensored_confounds.shape == original_confounds.shape
    # Has the nifti stayed the same shape?
    assert nb.load(results.outputs.bold_file_TR).get_fdata().shape[3] == original_nvols_nifti

    # Test a nifti file with 'n' volumes to remove
    for n in range(0, original_nvols_nifti-1):  # Testing all n values till original_nvols_nifti - 1
        remove_n_vols = removeTR(
            bold_file=boldfile,
            fmriprep_conf=confounds_file,
            initial_volumes_to_drop=n,
            mask_file=mask_file)
        results = remove_n_vols.run()
        censored_confounds = pd.read_table(results.outputs.fmrip_confdropTR)
        # Were the files created?
        assert op.exists(results.outputs.bold_file_TR)
        assert op.exists(results.outputs.fmrip_confdropTR)
        # Have the confounds changed correctly?
        assert censored_confounds.shape[0] == original_confounds.shape[0] - n
        # Has the nifti changed correctly?
        try:
            assert nb.load(results.outputs.bold_file_TR).get_fdata().shape[3]\
                == original_nvols_nifti - n
        except Exception as exc:
            exc = nb.load(results.outputs.bold_file_TR).get_fdata().shape[3]
            print("Tests failing at N = {}.".format(n))
            raise Exception("Number of volumes in censored nifti is {}.".format(exc))

