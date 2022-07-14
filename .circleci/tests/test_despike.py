#!/usr/bin/env python

"""
This file contains pytest for the despike workflow.
These tests can be run locally and on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.



"""
import pytest
import os.path as op
import os
import pandas as pd
import nibabel as nb


def test_data_availability(data_dir, working_dir, output_dir):
    """Makes sure that we have access to all the testing data
    """
    assert op.exists(output_dir)
    assert op.exists(working_dir)
    assert op.exists(data_dir)
    boldfile = data_dir + "/withoutfreesurfer/sub-01/func/" \
        "sub-01_task-mixedgamblestask_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    assert op.exists(boldfile)

#def test_despike():
    



