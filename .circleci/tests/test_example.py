#!/usr/bin/env python

"""
This file is an example of running pytests either locally or on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.



"""
import pytest
import os.path as op


def test_data_availability(data_dir, working_dir, output_dir):
    """Makes sure that we have access to all the testing data
    """
    assert op.exists(output_dir)
    assert op.exists(working_dir)
    assert op.exists(data_dir)
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    assert op.exists(boldfile)


def test_removeTR(data_dir, working_dir, output_dir):
    from xcp_d.interfaces.prepostcleaning import removeTR
    import pandas as pd

    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_desc-confounds_timeseries.tsv"
    mask_file = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"

    original_confounds = pd.read_csv(confounds_file, sep="\t")

    remove_nothing = removeTR(
        bold_file=boldfile,
        fmriprep_conf=confounds_file,
        TR=2,
        time_todrop=0,
        mask_file=mask_file)
    results = remove_nothing.run()
    print(results.outputs)

    nothing_outputs = pd.read_csv(results.outputs.fmrip_confdropTR)

    assert op.exists(results.outputs.fmrip_confdropTR)
