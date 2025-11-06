"""Tests for functions in the cli.parser module."""

import logging
import os
from copy import deepcopy
from pathlib import Path

import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from xcp_d import config
from xcp_d.cli import parser
from xcp_d.data import load as load_data
from xcp_d.tests.test_config import _reset_config
from xcp_d.tests.utils import modified_environ

build_log = logging.getLogger()
build_log.setLevel(10)


class FakeOptions:
    """A structure to mimic argparse opts."""

    def __init__(self, **entries):
        self.__dict__.update(entries)


@pytest.fixture(scope='module')
def base_opts():
    """Create base options."""
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
    }
    opts = FakeOptions(**opts_dict)
    return opts


def test_validate_parameters_01(base_opts, base_parser):
    """Test parser._validate_parameters with the pre-set options."""
    opts = deepcopy(base_opts)
    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)


def test_validate_parameters_02(base_opts, base_parser, caplog):
    """Test parser._validate_parameters with censoring disabled."""
    opts = deepcopy(base_opts)

    # Disable censoring
    opts.fd_thresh = 0

    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.min_time == 0
    assert 'Framewise displacement-based scrubbing is disabled.' in caplog.text


def test_validate_parameters_03(base_opts, base_parser):
    """Test parser._validate_parameters with the dcan input type."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.input_type = 'dcan'
    opts.process_surfaces = False

    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.process_surfaces is False


def test_validate_parameters_04(base_opts, base_parser, capsys):
    """Test parser._validate_parameters with nifti outputs and surface processing."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.process_surfaces = True
    opts.file_format = 'nifti'

    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'you must enable cifti processing' in capsys.readouterr().err


def test_validate_parameters_05(base_opts, base_parser, caplog):
    """Test parser._validate_parameters with no-parcellation + min_coverage."""
    opts = deepcopy(base_opts)
    opts.atlases = []
    opts.min_coverage = 0.1

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'When no atlases are selected' in caplog.text


def test_validate_parameters_06(base_opts, base_parser, capsys):
    """Test parser._validate_parameters nifti + process_surfaces."""
    opts = deepcopy(base_opts)
    opts.input_type = 'ukb'
    opts.file_format = 'nifti'
    opts.process_surfaces = True

    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    stderr = capsys.readouterr().err
    assert '--warp-surfaces-native2std is not supported' in stderr
    assert 'In order to perform surface normalization' in stderr


def test_validate_parameters_motion_filtering(base_opts, base_parser, caplog, capsys):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set notch filter with no min or max
    opts.motion_filter_type = 'notch'
    opts.band_stop_min = None
    opts.band_stop_max = None

    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'Please set both' in capsys.readouterr().err

    # Set min > max for notch filter
    opts.motion_filter_type = 'notch'
    opts.band_stop_min = 18
    opts.band_stop_max = 12

    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'must be lower than' in capsys.readouterr().err

    # Set min <1 for notch filter
    opts.motion_filter_type = 'notch'
    opts.band_stop_min = 0.01
    opts.band_stop_max = 15

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'suspiciously low.' in caplog.text

    # Set lp without min
    opts.motion_filter_type = 'lp'
    opts.band_stop_min = None
    opts.band_stop_max = None

    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "Please set '--band-stop-min'" in capsys.readouterr().err

    # Set min > max for notch filter
    opts.motion_filter_type = 'lp'
    opts.band_stop_min = 0.01
    opts.band_stop_max = None

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'suspiciously low.' in caplog.text

    # Set min > max for notch filter
    opts.motion_filter_type = 'lp'
    opts.band_stop_min = 12
    opts.band_stop_max = 18

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "'--band-stop-max' is ignored" in caplog.text

    # Set min > max for notch filter
    opts.motion_filter_type = None
    opts.band_stop_min = 12
    opts.band_stop_max = 18

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "'--band-stop-min' and '--band-stop-max' are ignored" in caplog.text


def test_validate_parameters_bandpass_filter(base_opts, base_parser, caplog, capsys):
    """Test parser._validate_parameters with bandpass filter modifications."""
    opts = deepcopy(base_opts)

    assert opts.bandpass_filter is True

    # Disable bandpass_filter to False indirectly
    opts.high_pass = -1
    opts.low_pass = -1

    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.bandpass_filter is False
    assert 'Bandpass filtering is disabled.' in caplog.text

    # Set upper BPF below lower one
    opts.bandpass_filter = True
    opts.high_pass = 0.01
    opts.low_pass = 0.001

    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'must be lower than' in capsys.readouterr().err


def _test_validate_parameters_fs_license(base_opts, base_parser, caplog, capsys, tmp_path_factory):
    """Ensure parser._validate_parameters returns 2 when fs_license_file doesn't exist.

    Not run now that the license isn't required.
    """
    tmpdir = tmp_path_factory.mktemp('test_validate_parameters_fs_license')

    opts = deepcopy(base_opts)
    opts.fs_license_file = None

    # FS_LICENSE exists (set in conftest)
    parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
    assert 'A valid FreeSurfer license file is required.' not in caplog.text

    # FS_LICENSE doesn't exist
    with pytest.raises(SystemExit, match='2'):
        with modified_environ(FS_LICENSE='/path/to/missing/file.txt'):
            parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'A valid FreeSurfer license file is required.' in capsys.readouterr().err

    # FS_LICENSE is an existing file
    license_file = os.path.join(tmpdir, 'license.txt')
    with open(license_file, 'w') as fo:
        fo.write('TEMP')

    # If file exists, return_code should be 0
    opts.fs_license_file = Path(license_file)
    parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
    assert 'Freesurfer license DNE' not in caplog.text

    # If file doesn't exist, return_code should be 1
    opts.fs_license_file = Path('/path/to/missing/file.txt')
    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert 'Freesurfer license DNE' in capsys.readouterr().err


def test_validate_parameters_linc_mode(base_opts, base_parser, capsys):
    """Test parser._validate_parameters with linc mode."""
    opts = deepcopy(base_opts)
    opts.mode = 'linc'

    # linc mode doesn't use abcc_qc but does use linc_qc
    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.abcc_qc is False
    assert opts.linc_qc is True
    assert opts.file_format == 'cifti'
    assert opts.min_coverage == 0.5
    assert opts.smoothing == 6.0
    assert opts.correlation_lengths == ['all']
    assert opts.report_output_level == 'root'

    # --create-matrices is not supported
    opts.correlation_lengths = ['300']
    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    stderr = capsys.readouterr().err
    assert "'--create-matrices' is not supported" in stderr


def test_validate_parameters_abcd_mode(base_opts, base_parser, capsys):
    """Test parser._validate_parameters with abcd mode."""
    opts = deepcopy(base_opts)
    opts.mode = 'abcd'
    opts.motion_filter_type = 'lp'
    opts.band_stop_min = 10

    # abcd mode does use abcc_qc but doesn't use linc_qc
    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.abcc_qc is True
    assert opts.combine_runs is True
    assert opts.correlation_lengths == []
    assert opts.despike is True
    assert opts.fd_thresh == 0.3
    assert opts.file_format == 'cifti'
    assert opts.input_type == 'fmriprep'
    assert opts.linc_qc is True
    assert opts.min_coverage == 0.5
    assert opts.process_surfaces is True
    assert opts.smoothing == 6.0
    assert opts.report_output_level == 'session'
    opts.correlation_lengths = ['300', 'all']
    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
    assert opts.correlation_lengths == ['300', 'all']

    # --motion-filter-type is required
    opts.motion_filter_type = None
    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    stderr = capsys.readouterr().err
    assert "'--motion-filter-type' is required for" in stderr


def test_validate_parameters_hbcd_mode(base_opts, base_parser, capsys):
    """Test parser._validate_parameters with hbcd mode."""
    opts = deepcopy(base_opts)
    opts.mode = 'hbcd'
    opts.motion_filter_type = 'lp'
    opts.band_stop_min = 10

    # hbcd mode does use abcc_qc but doesn't use linc_qc
    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.abcc_qc is True
    assert opts.combine_runs is True
    assert opts.correlation_lengths == []
    assert opts.despike is True
    assert opts.fd_thresh == 0.3
    assert opts.file_format == 'cifti'
    assert opts.input_type == 'nibabies'
    assert opts.linc_qc is True
    assert opts.min_coverage == 0.5
    assert opts.process_surfaces is True
    assert opts.report_output_level == 'session'
    assert opts.smoothing == 6.0

    opts.correlation_lengths = ['300', 'all']
    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
    assert opts.correlation_lengths == ['300', 'all']

    # --motion-filter-type is required
    opts.motion_filter_type = None
    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    stderr = capsys.readouterr().err
    assert "'--motion-filter-type' is required for" in stderr


def test_validate_parameters_nichart_mode(base_opts, base_parser, capsys):
    """Test parser._validate_parameters with nichart mode."""
    opts = deepcopy(base_opts)
    opts.mode = 'nichart'

    # linc mode doesn't use abcc_qc but does use linc_qc
    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.abcc_qc is False
    assert opts.linc_qc is True
    assert opts.file_format == 'nifti'
    assert opts.min_coverage == 0.4
    assert opts.smoothing == 0
    assert opts.report_output_level == 'root'


def test_validate_parameters_none_mode(base_opts, base_parser, capsys):
    """Test parser._validate_parameters with none mode."""
    opts = deepcopy(base_opts)
    opts.mode = 'none'

    with pytest.raises(SystemExit, match='2'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    stderr = capsys.readouterr().err
    assert "'--abcc-qc' (y or n) is required for 'none' mode." in stderr
    assert "'--combine-runs' (y or n) is required for 'none' mode." in stderr
    assert "'--despike' (y or n) is required for 'none' mode." in stderr
    assert "'--fd-thresh' is required for 'none' mode." in stderr
    assert "'--file-format' is required for 'none' mode." in stderr
    assert "'--input-type' is required for 'none' mode." in stderr
    assert "'--linc-qc' (y or n) is required for 'none' mode." in stderr
    assert "'--min-coverage' is required for 'none' mode." in stderr
    assert "'--motion-filter-type' is required for 'none' mode." in stderr
    assert "'--nuisance-regressors' is required for 'none' mode." in stderr
    assert "'--output-type' is required for 'none' mode." in stderr
    assert "'--smoothing' is required for 'none' mode." in stderr
    assert "'--warp-surfaces-native2std' (y or n) is required for 'none' mode." in stderr

    opts.abcc_qc = False
    opts.combine_runs = False
    opts.confounds_config = 'none'
    opts.despike = False
    opts.fd_thresh = 0
    opts.file_format = 'nifti'
    opts.input_type = 'fmriprep'
    opts.linc_qc = False
    opts.min_coverage = 0.5
    opts.motion_filter_type = 'none'
    opts.output_type = 'censored'
    opts.params = '36P'
    opts.process_surfaces = False
    opts.smoothing = 0
    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
    assert opts.report_output_level == 'root'


def test_validate_parameters_other_mode(base_opts, base_parser, capsys):
    """Test parser._validate_parameters with 'other' mode."""
    opts = deepcopy(base_opts)
    opts.mode = 'other'

    with pytest.raises(AssertionError, match='Unsupported mode "other"'):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)


def test_build_parser_01(tmp_path_factory):
    """Test parser._build_parser with abcd mode."""
    tmpdir = tmp_path_factory.mktemp('test_build_parser_01')
    data_dir = os.path.join(tmpdir, 'data')
    data_path = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, 'out')
    out_path = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Parameters for abcd mode
    base_args = [
        data_dir,
        out_dir,
        'participant',
        '--mode',
        'abcd',
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
    ]
    parser_obj = parser._build_parser()

    opts = parser_obj.parse_args(args=base_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.despike == 'auto'

    test_args = base_args[:]
    test_args.extend(['--create-matrices', 'all', '300', '480'])
    opts = parser_obj.parse_args(args=test_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.correlation_lengths == ['all', '300.0', '480.0']


def test_build_parser_02(tmp_path_factory):
    """Test parser._build_parser with hbcd mode."""
    tmpdir = tmp_path_factory.mktemp('test_build_parser_02')
    data_dir = os.path.join(tmpdir, 'data')
    data_path = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, 'out')
    out_path = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Parameters for hbcd mode
    base_args = [
        data_dir,
        out_dir,
        'participant',
        '--mode',
        'hbcd',
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
    ]
    parser_obj = parser._build_parser()

    opts = parser_obj.parse_args(args=base_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.despike == 'auto'

    test_args = base_args[:]
    test_args.extend(['--create-matrices', 'all', '300', '480'])
    opts = parser_obj.parse_args(args=test_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.correlation_lengths == ['all', '300.0', '480.0']


@pytest.mark.parametrize(
    ('mode', 'combine_runs', 'expectation'),
    [
        ('linc', 'auto', False),
        ('abcd', 'auto', True),
        ('hbcd', 'auto', True),
        ('linc', None, True),
        ('abcd', None, True),
        ('hbcd', None, True),
        ('linc', 'y', True),
        ('abcd', 'y', True),
        ('hbcd', 'y', True),
        ('linc', 'n', False),
        ('abcd', 'n', False),
        ('hbcd', 'n', False),
    ],
)
def test_build_parser_03(tmp_path_factory, mode, combine_runs, expectation):
    """Test processing of the "combine_runs" parameter."""
    tmpdir = tmp_path_factory.mktemp('test_build_parser_03')
    data_dir = os.path.join(tmpdir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Parameters for hbcd mode
    base_args = [
        data_dir,
        out_dir,
        'participant',
        '--mode',
        mode,
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
    ]
    if combine_runs not in ('auto', None):
        base_args += ['--combine-runs', combine_runs]
    elif combine_runs is None:
        base_args += ['--combine-runs']

    parser_obj = parser._build_parser()
    opts = parser_obj.parse_args(args=base_args, namespace=None)
    if combine_runs == 'auto':
        assert opts.combine_runs == 'auto'

    opts = parser._validate_parameters(opts=opts, build_log=build_log, parser=parser_obj)

    assert opts.combine_runs is expectation


@pytest.mark.parametrize(
    ('mode', 'despike', 'expectation'),
    [
        ('linc', 'auto', True),
        ('abcd', 'auto', True),
        ('hbcd', 'auto', True),
        ('linc', None, True),
        ('abcd', None, True),
        ('hbcd', None, True),
        ('linc', 'y', True),
        ('abcd', 'y', True),
        ('hbcd', 'y', True),
        ('linc', 'n', False),
        ('abcd', 'n', False),
        ('hbcd', 'n', False),
    ],
)
def test_build_parser_04(tmp_path_factory, mode, despike, expectation):
    """Test processing of the "despike" parameter."""
    tmpdir = tmp_path_factory.mktemp('test_build_parser_04')
    data_dir = os.path.join(tmpdir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Parameters for hbcd mode
    base_args = [
        data_dir,
        out_dir,
        'participant',
        '--mode',
        mode,
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
    ]
    if despike not in ('auto', None):
        base_args += ['--despike', despike]
    elif despike is None:
        base_args += ['--despike']

    parser_obj = parser._build_parser()
    opts = parser_obj.parse_args(args=base_args, namespace=None)
    if despike == 'auto':
        assert opts.despike == 'auto'
    else:
        assert opts.despike is expectation

    opts = parser._validate_parameters(opts=opts, build_log=build_log, parser=parser_obj)

    assert opts.despike is expectation


@pytest.mark.parametrize(
    ('mode', 'process_surfaces', 'expectation'),
    [
        ('linc', 'auto', False),
        ('abcd', 'auto', True),
        ('hbcd', 'auto', True),
        ('linc', None, True),
        ('abcd', None, True),
        ('hbcd', None, True),
        ('linc', 'y', True),
        ('abcd', 'y', True),
        ('hbcd', 'y', True),
        ('linc', 'n', False),
        ('abcd', 'n', False),
        ('hbcd', 'n', False),
    ],
)
def test_build_parser_05(tmp_path_factory, mode, process_surfaces, expectation):
    """Test processing of the "process_surfaces" parameter."""
    tmpdir = tmp_path_factory.mktemp('test_build_parser_05')
    data_dir = os.path.join(tmpdir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Parameters for hbcd mode
    base_args = [
        data_dir,
        out_dir,
        'participant',
        '--mode',
        mode,
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
        '--file-format',
        'cifti',
    ]
    if process_surfaces not in ('auto', None):
        base_args += ['--warp-surfaces-native2std', process_surfaces]
    elif process_surfaces is None:
        base_args += ['--warp-surfaces-native2std']

    parser_obj = parser._build_parser()
    opts = parser_obj.parse_args(args=base_args, namespace=None)
    if process_surfaces == 'auto':
        assert opts.process_surfaces == 'auto'

    opts = parser._validate_parameters(opts=opts, build_log=build_log, parser=parser_obj)

    assert opts.process_surfaces is expectation


@pytest.mark.parametrize(
    ('mode', 'file_format', 'expectation'),
    [
        ('linc', 'auto', 'cifti'),
        ('abcd', 'auto', 'cifti'),
        ('hbcd', 'auto', 'cifti'),
        ('linc', 'nifti', 'nifti'),
        ('abcd', 'nifti', 'nifti'),
        ('hbcd', 'nifti', 'nifti'),
        ('linc', 'cifti', 'cifti'),
        ('abcd', 'cifti', 'cifti'),
        ('hbcd', 'cifti', 'cifti'),
    ],
)
def test_build_parser_06(tmp_path_factory, mode, file_format, expectation):
    """Test processing of the "file_format" parameter."""
    tmpdir = tmp_path_factory.mktemp('test_build_parser_06')
    data_dir = os.path.join(tmpdir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, 'out')
    os.makedirs(out_dir, exist_ok=True)

    # Parameters for hbcd mode
    base_args = [
        data_dir,
        out_dir,
        'participant',
        '--mode',
        mode,
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
        '--warp-surfaces-native2std',
        'n',
    ]
    if file_format != 'auto':
        base_args += ['--file-format', file_format]

    parser_obj = parser._build_parser()
    opts = parser_obj.parse_args(args=base_args, namespace=None)
    if file_format == 'auto':
        assert opts.file_format == 'auto'

    opts = parser._validate_parameters(opts=opts, build_log=build_log, parser=parser_obj)

    assert opts.file_format == expectation


def test_parse_args_01(tmp_path_factory):
    """Test parser._build_parser with nibabies input type and one-to-all mapping."""
    skeleton = load_data('tests/skeletons/nibabies_longitudinal_one_to_all.yml')
    tmpdir = tmp_path_factory.mktemp('test_parse_args_01')
    bids_dir = tmpdir / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    out_dir = tmpdir / 'out'
    out_dir.mkdir(exist_ok=True, parents=True)

    # Parameters for nibabies input type
    base_args = [
        str(bids_dir),
        str(out_dir),
        'participant',
        '--mode',
        'hbcd',
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
        '--input-type',
        'nibabies',
    ]
    parser.parse_args(args=base_args, namespace=None)

    assert config.execution.fmri_dir == bids_dir
    assert config.execution.output_dir == out_dir
    assert config.execution.processing_list == [['01', '', ['V02', 'V03', 'V04']]]
    _reset_config()


def test_parse_args_02(tmp_path_factory):
    """Test parser._build_parser with nibabies input type and one-to-one mapping."""
    skeleton = load_data('tests/skeletons/nibabies_longitudinal_one_to_one.yml')
    tmpdir = tmp_path_factory.mktemp('test_parse_args_02')
    bids_dir = tmpdir / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    out_dir = tmpdir / 'out'
    out_dir.mkdir(exist_ok=True, parents=True)

    # Parameters for nibabies input type
    base_args = [
        str(bids_dir),
        str(out_dir),
        'participant',
        '--mode',
        'hbcd',
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
        '--input-type',
        'nibabies',
    ]
    parser.parse_args(args=base_args, namespace=None)
    assert config.execution.fmri_dir == bids_dir
    assert config.execution.output_dir == out_dir
    assert config.execution.processing_list == [
        ['01', 'V02', ['V02']],
        ['01', 'V03', ['V03']],
        ['01', 'V04', ['V04']],
    ]
    _reset_config()


def test_parse_args_03(tmp_path_factory):
    """Test parser._build_parser with nibabies input type and one-anat-session mapping."""
    skeleton = load_data('tests/skeletons/nibabies_longitudinal_one_anat_session.yml')
    tmpdir = tmp_path_factory.mktemp('test_parse_args_03')
    bids_dir = tmpdir / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    out_dir = tmpdir / 'out'
    out_dir.mkdir(exist_ok=True, parents=True)

    # Parameters for nibabies input type
    base_args = [
        str(bids_dir),
        str(out_dir),
        'participant',
        '--mode',
        'hbcd',
        '--motion-filter-type',
        'lp',
        '--band-stop-min',
        '10',
        '--input-type',
        'nibabies',
    ]
    parser.parse_args(args=base_args, namespace=None)
    assert config.execution.fmri_dir == bids_dir
    assert config.execution.output_dir == out_dir
    assert config.execution.processing_list == [['01', 'V02', ['V02', 'V03', 'V04']]]
    _reset_config()
