#!/usr/bin/env python

"""
This file is an example of running pytests either locally or on circleci.

Arguments have to be passed to these functions because the data may be
mounted in a container somewhere unintuitively.



"""
import pytest
import os.path as op


def test_data_availibility(data_dir, working_dir, output_dir):
    """Makes sure that we have access to all the testing data
    """
    assert op.exists(output_dir)
    assert op.exists(working_dir)
    assert op.exists(data_dir)
