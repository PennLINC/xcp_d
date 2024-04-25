"""Tests for functions in the cli.parser module."""

import logging
import os
from copy import deepcopy
from pathlib import Path

import pytest

from xcp_d.cli import parser
from xcp_d.tests.utils import modified_environ

build_log = logging.getLogger()
build_log.setLevel(10)


class FakeOptions:
    """A structure to mimic argparse opts."""

    def __init__(self, **entries):
        self.__dict__.update(entries)


@pytest.fixture(scope="module")
def base_opts():
    """Create base options."""
    opts_dict = {
        "fmri_dir": Path("dset"),
        "output_dir": Path("out"),
        "work_dir": Path("work"),
        "analysis_level": "participant",
        "mode": "linc",
        "file_format": "nifti",
        "input_type": "fmriprep",
        "lower_bpf": 0.01,
        "upper_bpf": 0.08,
        "bandpass_filter": True,
        "fd_thresh": 0,
        "min_time": 240,
        "motion_filter_type": None,
        "band_stop_min": None,
        "band_stop_max": None,
        "motion_filter_order": None,
        "process_surfaces": False,
        "fs_license_file": Path(os.environ["FS_LICENSE"]),
        "atlases": ["Glasser"],
        "custom_confounds": None,
        "dcan_correlation_lengths": None,
        "despike": None,
        "abcc_qc": False,
        "combineruns": False,
    }
    opts = FakeOptions(**opts_dict)
    return opts


def test_validate_parameters_01(base_opts, base_parser):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)
    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)


def test_validate_parameters_04(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    assert opts.bandpass_filter is True

    # Disable bandpass_filter to False indirectly
    opts.lower_bpf = -1
    opts.upper_bpf = -1

    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.bandpass_filter is False
    assert "Bandpass filtering is disabled." in caplog.text


def test_validate_parameters_05(base_opts, base_parser, capsys):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set upper BPF below lower one
    opts.lower_bpf = 0.01
    opts.upper_bpf = 0.001

    with pytest.raises(SystemExit, match="2"):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "must be lower than" in capsys.readouterr().err


def test_validate_parameters_06(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Disable censoring
    opts.fd_thresh = 0

    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.min_time == 0
    assert "Framewise displacement-based scrubbing is disabled." in caplog.text


def test_validate_parameters_07(base_opts, base_parser, capsys):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set notch filter with no min or max
    opts.motion_filter_type = "notch"
    opts.band_stop_min = None
    opts.band_stop_max = None

    with pytest.raises(SystemExit, match="2"):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "Please set both" in capsys.readouterr().err


def test_validate_parameters_08(base_opts, base_parser, capsys):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.motion_filter_type = "notch"
    opts.band_stop_min = 18
    opts.band_stop_max = 12

    with pytest.raises(SystemExit, match="2"):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "must be lower than" in capsys.readouterr().err


def test_validate_parameters_09(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min <1 for notch filter
    opts.motion_filter_type = "notch"
    opts.band_stop_min = 0.01

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "suspiciously low." in caplog.text


def test_validate_parameters_10(base_opts, base_parser, capsys):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set lp without min
    opts.motion_filter_type = "lp"
    opts.band_stop_min = None
    opts.band_stop_max = None

    with pytest.raises(SystemExit, match="2"):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "Please set '--band-stop-min'" in capsys.readouterr().err


def test_validate_parameters_11(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.motion_filter_type = "lp"
    opts.band_stop_min = 0.01
    opts.band_stop_max = None

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "suspiciously low." in caplog.text


def test_validate_parameters_12(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.motion_filter_type = "lp"
    opts.band_stop_min = 12
    opts.band_stop_max = 18

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "'--band-stop-max' is ignored" in caplog.text


def test_validate_parameters_13(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.motion_filter_type = None
    opts.band_stop_min = 12
    opts.band_stop_max = 18

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "'--band-stop-min' and '--band-stop-max' are ignored" in caplog.text


def test_validate_parameters_14(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.input_type = "dcan"
    opts.file_format = "nifti"

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "cifti processing (--cifti) will be enabled automatically." in caplog.text


def test_validate_parameters_15(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.input_type = "dcan"
    opts.process_surfaces = False

    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.file_format == "cifti"
    assert "(--warp-surfaces-native2std) will be enabled automatically." in caplog.text


def test_validate_parameters_16(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.input_type = "dcan"
    opts.process_surfaces = False

    opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert opts.process_surfaces is True
    assert "(--warp-surfaces-native2std) will be enabled automatically." in caplog.text


def test_validate_parameters_17(base_opts, base_parser, capsys):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)

    # Set min > max for notch filter
    opts.process_surfaces = True
    opts.file_format = "nifti"

    with pytest.raises(SystemExit, match="2"):
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "you must enable cifti processing" in capsys.readouterr().err


def test_validate_parameters_18(base_opts, base_parser, caplog, capsys):
    """Ensure parser._validate_parameters returns 0 when no fs_license_file is provided.

    This should work as long as the environment path exists.
    """
    opts = deepcopy(base_opts)
    opts.fs_license_file = None

    # FS_LICENSE exists (set in conftest)
    parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
    assert "A valid FreeSurfer license file is required." not in caplog.text

    # FS_LICENSE doesn't exist
    with pytest.raises(SystemExit, match="2"):
        with modified_environ(FS_LICENSE="/path/to/missing/file.txt"):
            parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "A valid FreeSurfer license file is required." in capsys.readouterr().err


def test_validate_parameters_19(base_opts, base_parser, caplog, capsys, tmp_path_factory):
    """Ensure parser._validate_parameters returns 1 when fs_license_file doesn't exist."""
    tmpdir = tmp_path_factory.mktemp("test_validate_parameters_19")
    license_file = os.path.join(tmpdir, "license.txt")
    with open(license_file, "w") as fo:
        fo.write("TEMP")

    opts = deepcopy(base_opts)

    # If file exists, return_code should be 0
    opts.fs_license_file = Path(license_file)
    parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
    assert "Freesurfer license DNE" not in caplog.text

    # If file doesn't exist, return_code should be 1
    with pytest.raises(SystemExit, match="2"):
        opts.fs_license_file = Path("/path/to/missing/file.txt")
        parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "Freesurfer license DNE" in capsys.readouterr().err


def test_validate_parameters_20(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)
    opts.atlases = []
    opts.min_coverage = 0.1

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "When no atlases are selected" in caplog.text


def test_validate_parameters_21(base_opts, base_parser, caplog):
    """Test parser._validate_parameters."""
    opts = deepcopy(base_opts)
    opts.input_type = "ukb"
    opts.file_format = "nifti"
    opts.process_surfaces = True

    _ = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    assert "cifti processing (--cifti) will be disabled automatically." in caplog.text
    assert "(--warp-surfaces-native2std) will be disabled automatically." in caplog.text


@pytest.mark.parametrize(
    "mode,despike,expectation",
    [
        ("linc", "auto", True),
        ("abcd", "auto", True),
        ("hbcd", "auto", True),
        ("linc", True, True),
        ("abcd", True, True),
        ("hbcd", True, True),
        ("linc", False, False),
        ("abcd", False, False),
        ("hbcd", False, False),
        ("linc", "n", AssertionError),
        ("abcd", "n", AssertionError),
        ("hbcd", "n", AssertionError),
    ],
)
def test_validate_parameters_22(base_opts, base_parser, mode, despike, expectation):
    """Test processing of the "despike" parameter."""
    opts = deepcopy(base_opts)
    # Pass in some parameters to satisfy "abcd" and "hbcd" modes
    opts.motion_filter_type = "lp"
    opts.band_stop_min = 10

    opts.mode = mode
    opts.despike = despike
    if isinstance(expectation, type) and issubclass(expectation, Exception):
        with pytest.raises(expectation):
            parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)

    else:
        mod_opts = parser._validate_parameters(deepcopy(opts), build_log, parser=base_parser)
        assert mod_opts.despike is expectation


def test_build_parser(tmp_path_factory):
    """Test parser._build_parser."""
    tmpdir = tmp_path_factory.mktemp("test_build_parser")
    data_dir = os.path.join(tmpdir, "data")
    data_path = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, "out")
    out_path = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base_args = [
        data_dir,
        out_dir,
        "participant",
        "--mode",
        "linc",
    ]
    parser_obj = parser._build_parser()

    opts = parser_obj.parse_args(args=base_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.despike == "auto"

    test_args = base_args[:]
    test_args.extend(["--despike"])
    opts = parser_obj.parse_args(args=test_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.despike is True

    test_args = base_args[:]
    test_args.extend(["--despike", "y"])
    opts = parser_obj.parse_args(args=test_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.despike is True

    test_args = base_args[:]
    test_args.extend(["--despike", "n"])
    opts = parser_obj.parse_args(args=test_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.despike is False

    # Parameters for abcd mode
    base_args = [
        data_dir,
        out_dir,
        "participant",
        "--mode",
        "abcd",
        "--motion-filter-type",
        "lp",
        "--band-stop-min",
        "10",
    ]
    parser_obj = parser._build_parser()

    opts = parser_obj.parse_args(args=base_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.despike == "auto"

    test_args = base_args[:]
    test_args.extend(["--create-matrices", "all", "300", "450"])
    opts = parser_obj.parse_args(args=test_args, namespace=None)
    assert opts.fmri_dir == data_path
    assert opts.output_dir == out_path
    assert opts.dcan_correlation_lengths == ["all", 300, 450]
