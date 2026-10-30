"""Tests for collecting per-run files when sibling runs differ by a single entity.

Regression tests for https://github.com/PennLINC/xcp_d/issues/1647.
"""

import json

import pytest
import yaml
from bids.layout import BIDSLayout, parse_file_entities

import xcp_d.utils.bids as xbids
from xcp_d.data import load as load_data

SUBJECT_PREFIX = 'sub-01_ses-1'

# Pairs of runs whose names differ only by one entity that one run has and the other lacks.
SIBLING_RUNS = {
    'acq_with_run': ('task-rest_run-01', 'task-rest_acq-4ME_run-01'),
    'acq_without_run': ('task-sst', 'task-sst_acq-4ME'),
    'ce': ('task-mid', 'task-mid_ce-gad'),
    'rec': ('task-gng', 'task-gng_rec-norm'),
    'dir': ('task-nback', 'task-nback_dir-PA'),
    'run': ('task-faces', 'task-faces_run-01'),
}

RUN_ENTITIES = ('task', 'acquisition', 'ceagent', 'reconstruction', 'direction', 'run', 'echo')


def _run_entities(path):
    """Return the entities that identify the run a file belongs to."""
    entities = parse_file_entities(str(path))
    return {k: entities.get(k) for k in RUN_ENTITIES}


def _build_layout(tmp_path, runs, skip_files=None):
    """Build a fake fMRIPrep derivatives dataset with the requested runs."""
    skip_files = skip_files or []
    root = tmp_path / 'fmriprep'
    func_dir = root / 'sub-01' / 'ses-1' / 'func'
    func_dir.mkdir(parents=True)
    (root / 'dataset_description.json').write_text(
        json.dumps(
            {
                'Name': 'fake',
                'BIDSVersion': '1.9.0',
                'DatasetType': 'derivative',
                'GeneratedBy': [{'Name': 'fMRIPrep'}],
            }
        )
    )
    sidecar = json.dumps({'RepetitionTime': 0.8})
    for run in runs:
        files = {
            'space-fsLR_den-91k_bold.dtseries.nii': '',
            'space-fsLR_den-91k_bold.json': sidecar,
            'space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz': '',
            'space-MNI152NLin6Asym_res-2_desc-preproc_bold.json': sidecar,
            'space-MNI152NLin6Asym_res-2_boldref.nii.gz': '',
            'space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz': '',
            'desc-confounds_timeseries.tsv': 'trans_x\n0\n',
            'desc-confounds_timeseries.json': '{}',
        }
        for suffix, content in files.items():
            fname = f'{SUBJECT_PREFIX}_{run}_{suffix}'
            if fname not in skip_files:
                (func_dir / fname).write_text(content)

    return BIDSLayout(
        root,
        validate=False,
        config=['bids', 'derivatives', str(load_data('xcp_d_bids_config2.json'))],
    )


def _get_bold_file(layout, run, file_format):
    """Return the BOLD file XCP-D would process for a run."""
    if file_format == 'cifti':
        suffix = 'space-fsLR_den-91k_bold.dtseries.nii'
    else:
        suffix = 'space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'

    bold_file = layout.get_file(f'sub-01/ses-1/func/{SUBJECT_PREFIX}_{run}_{suffix}')
    return bold_file.path


def _load_confounds_spec(name):
    """Load a built-in confounds config by name (e.g., '36P')."""
    return yaml.safe_load(load_data.readable(f'nuisance/{name}.yml').read_text())


@pytest.mark.parametrize('file_format', ['cifti', 'nifti'])
@pytest.mark.parametrize('pair', list(SIBLING_RUNS.keys()))
def test_collect_run_data_sibling_runs(tmp_path, monkeypatch, pair, file_format):
    """Each run must get its own files, not those of a sibling run with an extra entity."""
    # The fake images are empty, so skip the header TR check.
    monkeypatch.setattr(xbids, '_check_tr_header_vs_sidecar', lambda *args: None)

    runs = SIBLING_RUNS[pair]
    layout = _build_layout(tmp_path, runs)
    for run in runs:
        bold_file = _get_bold_file(layout, run, file_format)
        run_data = xbids.collect_run_data(layout, bold_file, file_format, 'MNI152NLin6Asym')
        collected = {k: v for k, v in run_data.items() if isinstance(v, str)}
        assert 'motion_file' in collected
        wrong = {
            k: v for k, v in collected.items() if _run_entities(v) != _run_entities(bold_file)
        }
        assert not wrong, f'{run} collected files from another run: {wrong}'


@pytest.mark.parametrize('pair', list(SIBLING_RUNS.keys()))
def test_collect_run_data_missing_file_raises(tmp_path, monkeypatch, pair):
    """A run missing its own file must raise, not fall back to a sibling run's file."""
    monkeypatch.setattr(xbids, '_check_tr_header_vs_sidecar', lambda *args: None)

    run, sibling = SIBLING_RUNS[pair]
    layout = _build_layout(
        tmp_path,
        (run, sibling),
        skip_files=[f'{SUBJECT_PREFIX}_{run}_space-MNI152NLin6Asym_res-2_boldref.nii.gz'],
    )
    bold_file = _get_bold_file(layout, run, 'cifti')
    with pytest.raises(FileNotFoundError, match='No boldref file found'):
        xbids.collect_run_data(layout, bold_file, 'cifti', 'MNI152NLin6Asym')


@pytest.mark.parametrize('pair', list(SIBLING_RUNS.keys()))
def test_collect_confounds_sibling_runs(tmp_path, pair):
    """Each run must get its own confounds, not those of a sibling run with an extra entity."""
    runs = SIBLING_RUNS[pair]
    layout = _build_layout(tmp_path, runs)
    confounds_spec = _load_confounds_spec('36P')
    for run in runs:
        bold_file = _get_bold_file(layout, run, 'cifti')
        confounds = xbids.collect_confounds(bold_file, layout, None, confounds_spec)
        conf_file = confounds['preproc_confounds']['file']
        assert _run_entities(conf_file) == _run_entities(bold_file)


@pytest.mark.parametrize('pair', list(SIBLING_RUNS.keys()))
def test_collect_confounds_missing_file_raises(tmp_path, pair):
    """A run missing its own confounds must raise, not fall back to a sibling run's file."""
    run, sibling = SIBLING_RUNS[pair]
    layout = _build_layout(
        tmp_path,
        (run, sibling),
        skip_files=[f'{SUBJECT_PREFIX}_{run}_desc-confounds_timeseries.tsv'],
    )
    bold_file = _get_bold_file(layout, run, 'cifti')
    confounds_spec = _load_confounds_spec('36P')
    with pytest.raises(FileNotFoundError, match='Could not find confound file'):
        xbids.collect_confounds(bold_file, layout, None, confounds_spec)
