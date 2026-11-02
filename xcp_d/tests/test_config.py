# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Check the configuration module and file."""

import pytest

from xcp_d import config


def _reset_config():
    """
    Forcibly reload the configuration module to restore defaults.
    .. caution::
      `importlib.reload` creates new sets of objects, but will not remove
      previous references to those objects."""
    import importlib

    importlib.reload(config)


def test_reset_config():
    execution = config.execution
    execution.fmri_dir = 'TESTING'
    assert config.execution.fmri_dir == 'TESTING'
    _reset_config()
    assert config.execution.fmri_dir is None
    # Even though the config module was reset,
    # previous references to config classes
    # have not been touched.
    assert execution.fmri_dir == 'TESTING'


def test_bids_filters_non_string_values(tmp_path):
    """Load BIDS filters with integer, list-of-integer, and serialized Query values."""
    import json

    from bids.layout import Query

    fmri_dir = tmp_path / 'derivatives'
    fmri_dir.mkdir()
    (fmri_dir / 'dataset_description.json').write_text(
        json.dumps(
            {
                'Name': 'Test',
                'BIDSVersion': '1.9.0',
                'DatasetType': 'derivative',
                'GeneratedBy': [{'Name': 'fMRIPrep'}],
            },
        ),
    )

    try:
        config.from_dict(
            {
                'fmri_dir': str(fmri_dir),
                'work_dir': str(tmp_path / 'work'),
                'bids_filters': {
                    'bold': {
                        'run': 1,
                        'echo': [1, 2],
                        'acquisition': '<Query.NONE: 1>',
                        # Query.ANY is an alias of Query.REQUIRED, so it serializes this way
                        'space': ['MNI152NLin6Asym', '<Query.REQUIRED: 2>'],
                    },
                },
            },
            init=['execution'],
        )
        bold_filters = config.execution.bids_filters['bold']
        assert bold_filters['run'] == 1
        assert bold_filters['echo'] == [1, 2]
        assert bold_filters['acquisition'] is Query.NONE
        assert bold_filters['space'] == ['MNI152NLin6Asym', Query.ANY]
    finally:
        _reset_config()


@pytest.mark.skip(reason='This test is not working')
def test_hash_config():
    # This may change with changes to config defaults / new attributes!
    expected = 'cfee5aaf'
    assert config.hash_config(config.get()) == expected
    _reset_config()

    config.execution.log_level = 5  # non-vital attributes do not matter
    assert config.hash_config(config.get()) == expected
    _reset_config()

    # but altering a vital attribute will create a new hash
    config.workflow.surface_recon_method = 'mcribs'
    assert config.hash_config(config.get()) != expected
    _reset_config()
