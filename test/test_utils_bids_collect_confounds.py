"""Tests for collect_confounds using simulated BIDS derivatives."""

import os
from pathlib import Path

import pytest
import yaml
from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

import xcp_d.utils.bids as xbids
from xcp_d.data import load as load_data


def _load_confounds_spec(name: str) -> dict:
    """Load a built-in confounds config by name (e.g., '36P')."""
    conf_path = load_data.readable(f'nuisance/{name}.yml')
    return yaml.safe_load(conf_path.read_text())


def _build_layout_from_skeleton(tmp_path_factory, skeleton_relpath: str) -> BIDSLayout:
    """Generate a temporary BIDS directory from a skeleton and return a BIDSLayout."""
    bids_dir = tmp_path_factory.mktemp(Path(skeleton_relpath).stem) / 'bids'
    skeleton = load_data(f'tests/skeletons/{skeleton_relpath}')
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(bids_dir, validate=False, config=['bids', 'derivatives', xcp_d_config])
    return layout


def _get_bold_preproc_files(layout: BIDSLayout) -> list[str]:
    """Return all preproc BOLD files in the layout (derivatives)."""
    return layout.get(return_type='file', suffix='bold', desc='preproc')


def test_collect_confounds_single_run(tmp_path_factory):
    """Confounds are collected for a single run with a standard confounds spec."""
    layout = _build_layout_from_skeleton(tmp_path_factory, 'confounds_single_run.yml')
    confounds_spec = _load_confounds_spec('36P')

    bold_files = _get_bold_preproc_files(layout)
    assert len(bold_files) == 1

    confounds = xbids.collect_confounds(
        bold_file=bold_files[0],
        preproc_dataset=layout,
        derivatives_datasets=None,
        confound_spec=confounds_spec,
    )
    # 36P spec defines a single entry 'preproc_confounds'
    assert 'preproc_confounds' in confounds
    conf_path = confounds['preproc_confounds']['file']
    assert conf_path.endswith('.tsv')
    # Should match the same run/entities as the BOLD file
    bold_base = os.path.basename(bold_files[0])
    conf_base = os.path.basename(conf_path)
    # Same task and run; confounds file should not include acq
    assert '_task-rest_' in bold_base
    assert '_task-rest_' in conf_base
    assert '_run-' in bold_base
    assert '_run-' in conf_base
    assert 'acq-3echo' not in conf_base


def test_collect_confounds_mixed_acq_runs(tmp_path_factory):
    """Two runs: one without acq, one with acq-3echo; confounds should match entities."""
    layout = _build_layout_from_skeleton(tmp_path_factory, 'confounds_acq_mixed.yml')
    confounds_spec = _load_confounds_spec('36P')

    bold_files = sorted(_get_bold_preproc_files(layout))
    assert len(bold_files) == 2

    for bf in bold_files:
        confounds = xbids.collect_confounds(
            bold_file=bf,
            preproc_dataset=layout,
            derivatives_datasets=None,
            confound_spec=confounds_spec,
        )
        conf_path = confounds['preproc_confounds']['file']
        conf_base = os.path.basename(conf_path)
        bold_base = os.path.basename(bf)
        # Verify common entities match
        assert '_task-rest_' in conf_base
        # Verify acq entity behavior: present only when bold has acq-3echo
        has_acq = 'acq-3echo' in bold_base
        if has_acq:
            assert 'acq-3echo' in conf_base
        else:
            assert 'acq-3echo' not in conf_base


def test_collect_confounds_missing_dataset_raises(tmp_path_factory):
    """If a confound spec references a dataset not provided, raise ValueError."""
    layout = _build_layout_from_skeleton(tmp_path_factory, 'confounds_single_run.yml')
    # Custom spec that requires a non-preprocessed dataset
    confounds_spec = {
        'confounds': {
            'other_conf': {
                'dataset': 'other',
                'query': {
                    'desc': 'confounds',
                    'suffix': 'timeseries',
                    'extension': '.tsv',
                },
            }
        }
    }
    bold_files = _get_bold_preproc_files(layout)
    assert len(bold_files) == 1
    with pytest.raises(ValueError, match='Missing dataset required by confound spec'):
        xbids.collect_confounds(
            bold_file=bold_files[0],
            preproc_dataset=layout,
            derivatives_datasets=None,  # 'other' not provided
            confound_spec=confounds_spec,
        )


def test_collect_confounds_missing_file_raises(tmp_path_factory):
    """If no confounds file matches the query, raise FileNotFoundError."""
    layout = _build_layout_from_skeleton(tmp_path_factory, 'confounds_no_confounds.yml')
    confounds_spec = _load_confounds_spec('36P')
    bold_files = _get_bold_preproc_files(layout)
    assert len(bold_files) == 1
    with pytest.raises(FileNotFoundError):
        xbids.collect_confounds(
            bold_file=bold_files[0],
            preproc_dataset=layout,
            derivatives_datasets=None,
            confound_spec=confounds_spec,
        )
