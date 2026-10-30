"""Tests for collect_run_data."""

import os

import pytest
from bids.layout import BIDSLayout, parse_file_entities
from niworkflows.utils.testing import generate_bids_skeleton

import xcp_d.utils.bids as xbids
from xcp_d.data import load as load_data

# Pairs of runs in the sibling_runs skeleton whose names differ only by one entity
# that one run has and the other lacks.
SIBLING_RUNS = {
    'acq_with_run': ('task-rest_run-01', 'task-rest_acq-4ME_run-01'),
    'acq_without_run': ('task-sst', 'task-sst_acq-4ME'),
    'ce': ('task-mid', 'task-mid_ce-gad'),
    'rec': ('task-gng', 'task-gng_rec-norm'),
    'dir': ('task-nback', 'task-nback_dir-PA'),
    'run': ('task-faces', 'task-faces_run-01'),
}
RUN_ENTITIES = ('task', 'acquisition', 'ceagent', 'reconstruction', 'direction', 'run', 'echo')
BOLD_SUFFIXES = {
    'cifti': 'space-fsLR_den-91k_bold.dtseries.nii',
    'nifti': 'space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz',
}


def _generate_sibling_runs(tmp_path_factory):
    """Generate the sibling_runs skeleton and return its func directory."""
    bids_dir = tmp_path_factory.mktemp('sibling_runs') / 'bids'
    skeleton = load_data('tests/skeletons/sibling_runs.yml')
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    return bids_dir / 'sub-01' / 'ses-1' / 'func'


def _get_layout(func_dir):
    """Return a BIDSLayout for the dataset containing func_dir."""
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    bids_dir = func_dir.parents[2]
    return BIDSLayout(bids_dir, validate=False, config=['bids', 'derivatives', xcp_d_config])


def _run_entities(path):
    """Return the entities that identify the run a file belongs to."""
    entities = parse_file_entities(str(path))
    return {k: entities.get(k) for k in RUN_ENTITIES}


@pytest.mark.parametrize('file_format', ['cifti', 'nifti'])
@pytest.mark.parametrize('pair', list(SIBLING_RUNS.keys()))
def test_collect_run_data_sibling_runs(tmp_path_factory, monkeypatch, pair, file_format):
    """Each run gets its own files, not those of a sibling run with an extra entity.

    Regression test for https://github.com/PennLINC/xcp_d/issues/1647.
    """
    # The skeleton's images are empty, so skip the image header TR check.
    monkeypatch.setattr(xbids, '_check_tr_header_vs_sidecar', lambda *args: None)

    func_dir = _generate_sibling_runs(tmp_path_factory)
    layout = _get_layout(func_dir)
    for run in SIBLING_RUNS[pair]:
        bold_file = str(func_dir / f'sub-01_ses-1_{run}_{BOLD_SUFFIXES[file_format]}')
        run_data = xbids.collect_run_data(layout, bold_file, file_format, 'MNI152NLin6Asym')
        collected = {k: v for k, v in run_data.items() if isinstance(v, str)}
        assert 'motion_file' in collected
        wrong = {
            k: os.path.basename(v)
            for k, v in collected.items()
            if _run_entities(v) != _run_entities(bold_file)
        }
        assert not wrong, f'{run} collected files from another run: {wrong}'


@pytest.mark.parametrize('pair', list(SIBLING_RUNS.keys()))
def test_collect_run_data_sibling_runs_missing_file(tmp_path_factory, monkeypatch, pair):
    """A run missing its own file raises an error instead of using a sibling run's file.

    Regression test for https://github.com/PennLINC/xcp_d/issues/1647.
    """
    monkeypatch.setattr(xbids, '_check_tr_header_vs_sidecar', lambda *args: None)

    func_dir = _generate_sibling_runs(tmp_path_factory)
    run = SIBLING_RUNS[pair][0]
    (func_dir / f'sub-01_ses-1_{run}_space-MNI152NLin6Asym_res-2_boldref.nii.gz').unlink()
    layout = _get_layout(func_dir)
    bold_file = str(func_dir / f'sub-01_ses-1_{run}_{BOLD_SUFFIXES["cifti"]}')
    with pytest.raises(FileNotFoundError, match='No boldref file found'):
        xbids.collect_run_data(layout, bold_file, 'cifti', 'MNI152NLin6Asym')


def test_collect_run_data_tr_header_mismatch(ds001419_data):
    """Warn when the sidecar RepetitionTime differs from the image header TR."""
    from unittest.mock import patch

    bold_file = ds001419_data['nifti_file']
    sidecar_tr = 2.0  # header TR for this file is 3.0 s

    with patch('xcp_d.utils.bids._get_tr', return_value=3.0):
        with patch('xcp_d.utils.bids.LOGGER') as mock_logger:
            xbids._check_tr_header_vs_sidecar(bold_file, sidecar_tr)

    mock_logger.warning.assert_called_once()
    warning_msg = mock_logger.warning.call_args[0][0]
    assert 'RepetitionTime' in warning_msg
    assert '2' in warning_msg or '3' in warning_msg


def test_collect_run_data_tr_header_matches(ds001419_data):
    """No warning when sidecar RepetitionTime matches the image header TR."""
    from unittest.mock import patch

    bold_file = ds001419_data['nifti_file']
    header_tr = xbids._get_tr(bold_file)

    with patch('xcp_d.utils.bids._get_tr', return_value=header_tr):
        with patch('xcp_d.utils.bids.LOGGER') as mock_logger:
            xbids._check_tr_header_vs_sidecar(bold_file, header_tr)

    mock_logger.warning.assert_not_called()
