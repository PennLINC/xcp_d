"""Tests for the --skip parameter functionality."""

import logging
from copy import deepcopy
from pathlib import Path

import pytest

from xcp_d.cli import parser

build_log = logging.getLogger()
build_log.setLevel(10)


class FakeOptions:
    """A structure to mimic argparse opts."""

    def __init__(self, **entries):
        self.__dict__.update(entries)


@pytest.fixture
def skip_opts():
    """Create base options with skip_outputs field."""
    opts_dict = {
        'fmri_dir': Path('dset'),
        'output_dir': Path('out'),
        'work_dir': Path('work'),
        'analysis_level': 'participant',
        'datasets': {},
        'mode': 'linc',
        'file_format': 'auto',
        'input_type': 'auto',
        'report_output_level': 'auto',
        'confounds_config': 'auto',
        'high_pass': 0.01,
        'low_pass': 0.1,
        'bandpass_filter': True,
        'fd_thresh': 'auto',
        'min_time': 240,
        'motion_filter_type': None,
        'band_stop_min': None,
        'band_stop_max': None,
        'motion_filter_order': None,
        'process_surfaces': 'auto',
        'atlases': ['Glasser'],
        'min_coverage': 'auto',
        'correlation_lengths': None,
        'despike': 'auto',
        'abcc_qc': 'auto',
        'linc_qc': 'auto',
        'smoothing': 'auto',
        'combine_runs': 'auto',
        'output_type': 'auto',
        'fs_license_file': None,
        'skip_outputs': [],
    }
    return FakeOptions(**opts_dict)


def test_skip_alff_with_bandpass_filter(skip_opts, base_parser):
    """Test skipping ALFF calculation with bandpass filter enabled."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['alff']
    opts.bandpass_filter = True

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'alff' in validated_opts.skip_outputs
    assert validated_opts.bandpass_filter is True


def test_skip_alff_without_bandpass_filter(skip_opts, base_parser, caplog):
    """Test skipping ALFF calculation with bandpass filter disabled - should warn."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['alff']
    opts.bandpass_filter = False

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'alff' in validated_opts.skip_outputs
    assert validated_opts.bandpass_filter is False
    assert 'Skipping ALFF has no effect when bandpass filtering is disabled' in caplog.text


def test_skip_reho(skip_opts, base_parser):
    """Test skipping ReHo calculation."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['reho']

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'reho' in validated_opts.skip_outputs
    # Atlases should not be affected
    assert validated_opts.atlases == ['Glasser']


def test_skip_parcellation(skip_opts, base_parser, caplog):
    """Test skipping parcellation - should set atlases to empty list."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['parcellation']
    opts.atlases = ['Glasser', 'Gordon']

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'parcellation' in validated_opts.skip_outputs
    assert validated_opts.atlases == []
    assert 'Skipping parcellation as requested' in caplog.text


def test_skip_connectivity(skip_opts, base_parser, caplog):
    """Test skipping connectivity - should set atlases to empty list."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['connectivity']
    opts.atlases = ['Glasser']

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'connectivity' in validated_opts.skip_outputs
    assert validated_opts.atlases == []
    assert 'Skipping connectivity (and parcellation) as requested' in caplog.text


def test_skip_multiple_outputs(skip_opts, base_parser):
    """Test skipping multiple outputs simultaneously."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['alff', 'reho', 'connectivity']
    opts.atlases = ['Glasser', 'Gordon']
    opts.bandpass_filter = True

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'alff' in validated_opts.skip_outputs
    assert 'reho' in validated_opts.skip_outputs
    assert 'connectivity' in validated_opts.skip_outputs
    assert validated_opts.atlases == []


def test_skip_all_outputs(skip_opts, base_parser):
    """Test skipping all possible outputs."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['alff', 'reho', 'parcellation', 'connectivity']
    opts.atlases = ['Glasser']
    opts.bandpass_filter = True

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert set(validated_opts.skip_outputs) == {'alff', 'reho', 'parcellation', 'connectivity'}
    assert validated_opts.atlases == []


def test_skip_empty_list(skip_opts, base_parser):
    """Test with empty skip_outputs list - no skipping."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = []

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert validated_opts.skip_outputs == []
    assert validated_opts.atlases == ['Glasser']  # Should not be modified


def test_skip_parcellation_with_empty_atlases(skip_opts, base_parser, caplog):
    """Test skipping parcellation when atlases is already empty."""
    caplog.clear()
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['parcellation']
    opts.atlases = []

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'parcellation' in validated_opts.skip_outputs
    assert validated_opts.atlases == []
    # Should not log the "Skipping parcellation as requested" message
    # since atlases is already empty
    assert 'Skipping parcellation as requested' not in caplog.text


def test_skip_connectivity_with_empty_atlases(skip_opts, base_parser, caplog):
    """Test skipping connectivity when atlases is already empty."""
    caplog.clear()
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['connectivity']
    opts.atlases = []

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'connectivity' in validated_opts.skip_outputs
    assert validated_opts.atlases == []
    # Should not log the message since atlases is already empty
    assert 'Skipping connectivity (and parcellation) as requested' not in caplog.text


def test_skip_alff_and_reho_only(skip_opts, base_parser):
    """Test skipping only ALFF and ReHo, keeping parcellation and connectivity."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['alff', 'reho']
    opts.atlases = ['Glasser', 'Gordon']
    opts.bandpass_filter = True

    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert 'alff' in validated_opts.skip_outputs
    assert 'reho' in validated_opts.skip_outputs
    # Atlases should not be affected - parcellation/connectivity not skipped
    assert validated_opts.atlases == ['Glasser', 'Gordon']


def test_skip_config_propagation(skip_opts, base_parser):
    """Test that skip_outputs is properly set in config.workflow."""
    opts = deepcopy(skip_opts)
    opts.skip_outputs = ['alff', 'reho']

    # This would normally be done during workflow initialization
    # For testing, we'll just verify the parameter is validated correctly
    validated_opts = parser._validate_parameters(opts, build_log, parser=base_parser)

    assert validated_opts.skip_outputs == ['alff', 'reho']


def test_skip_parameter_choices():
    """Test that the --skip parameter accepts only valid choices."""
    from xcp_d.cli.parser import _build_parser

    arg_parser = _build_parser()

    # Get the skip argument
    skip_action = None
    for action in arg_parser._actions:
        if action.dest == 'skip_outputs':
            skip_action = action
            break

    assert skip_action is not None, 'skip_outputs argument not found'
    assert skip_action.choices == ['alff', 'reho', 'parcellation', 'connectivity']
    assert skip_action.default == []


def test_skip_parameter_help_text():
    """Test that the --skip parameter has appropriate help text."""
    from xcp_d.cli.parser import _build_parser

    arg_parser = _build_parser()

    # Get the skip argument
    skip_action = None
    for action in arg_parser._actions:
        if action.dest == 'skip_outputs':
            skip_action = action
            break

    assert skip_action is not None, 'skip_outputs argument not found'
    assert 'Skip specific outputs during postprocessing' in skip_action.help
    assert 'alff' in skip_action.help
    assert 'reho' in skip_action.help
    assert 'parcellation' in skip_action.help
    assert 'connectivity' in skip_action.help
