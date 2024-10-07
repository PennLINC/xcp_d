"""Test confounds handling."""

import pytest
import yaml

from xcp_d.data import load as load_data
from xcp_d.utils import boilerplate


def test_describe_motion_parameters():
    """Test boilerplate.describe_motion_parameters."""
    desc = boilerplate.describe_motion_parameters(
        motion_filter_type=None,
        motion_filter_order=None,
        band_stop_min=None,
        band_stop_max=None,
        TR=0.8,
    )
    assert "filtered to remove signals" not in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="notch",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        TR=0.8,
    )
    assert "band-stop filtered to remove signals" in desc
    assert "automatically modified" not in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="notch",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        TR=3,
    )
    assert "band-stop filtered to remove signals" in desc
    assert "automatically modified" in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="lp",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        TR=0.8,
    )
    assert "low-pass filtered" in desc
    assert "automatically modified" not in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="lp",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        TR=3,
    )
    assert "low-pass filtered" in desc
    assert "automatically modified" in desc


def test_describe_censoring():
    """Test boilerplate.describe_censoring."""
    motion_filter_type = "notch"
    fd_thresh = 0.2
    exact_scans = []
    desc = boilerplate.describe_censoring(
        motion_filter_type=motion_filter_type,
        head_radius=50,
        fd_thresh=fd_thresh,
        exact_scans=exact_scans,
    )
    assert "Volumes with filtered framewise displacement" in desc

    motion_filter_type = None
    fd_thresh = 0.2
    exact_scans = []
    desc = boilerplate.describe_censoring(
        motion_filter_type=motion_filter_type,
        head_radius=50,
        fd_thresh=fd_thresh,
        exact_scans=exact_scans,
    )
    assert "Volumes with framewise displacement" in desc

    motion_filter_type = None
    fd_thresh = 0.2
    exact_scans = [100, 200, 300]
    desc = boilerplate.describe_censoring(
        motion_filter_type=motion_filter_type,
        head_radius=50,
        fd_thresh=fd_thresh,
        exact_scans=exact_scans,
    )
    assert "Volumes with framewise displacement" in desc
    assert "limited to 100, 200, and 300 volumes" in desc

    motion_filter_type = None
    fd_thresh = 0
    exact_scans = [100, 200, 300]
    desc = boilerplate.describe_censoring(
        motion_filter_type=motion_filter_type,
        head_radius=50,
        fd_thresh=fd_thresh,
        exact_scans=exact_scans,
    )
    assert "Volumes were randomly selected for censoring" in desc
    assert "limited to 100, 200, and 300 volumes" in desc

    motion_filter_type = "notch"
    fd_thresh = 0
    exact_scans = [100, 200, 300]
    desc = boilerplate.describe_censoring(
        motion_filter_type=motion_filter_type,
        head_radius=50,
        fd_thresh=fd_thresh,
        exact_scans=exact_scans,
    )
    assert "Volumes were randomly selected for censoring" in desc
    assert "limited to 100, 200, and 300 volumes" in desc


def test_describe_regression(tmp_path_factory):
    """Test boilerplate.describe_regression."""
    _check_describe_regression_result("24P", "24 nuisance regressors were selected")
    _check_describe_regression_result("27P", "27 nuisance regressors were selected")
    _check_describe_regression_result("36P", "36 nuisance regressors were selected")
    _check_describe_regression_result("acompcor", "The top 5 aCompCor principal components")
    _check_describe_regression_result("acompcor_gsr", "The top 5 aCompCor principal components")
    _check_describe_regression_result("aroma", "AROMA motion-labeled components")
    _check_describe_regression_result("aroma_gsr", "AROMA motion-labeled components")
    _check_describe_regression_result(None, "No nuisance regression was performed")

    # Try with motion filter
    config = load_data.readable("nuisance/24P.yml")
    config = yaml.safe_load(config.read_text())

    result = boilerplate.describe_regression(
        confounds_config=config,
        motion_filter_type="lp",
        motion_filter_order=4,
        band_stop_min=6,
        band_stop_max=0,
        TR=0.8,
        fd_thresh=0,
    )
    assert isinstance(result, str)

    # Fails. Need to replace with a better test.
    with pytest.raises(TypeError, match="string indices must be integers"):
        boilerplate.describe_regression(
            confounds_config="test",
            motion_filter_type=None,
            motion_filter_order=0,
            band_stop_min=0,
            band_stop_max=0,
            TR=0.8,
            fd_thresh=0,
        )


def _check_describe_regression_result(config, match):
    if isinstance(config, str):
        config = load_data.readable(f"nuisance/{config}.yml")
        config = yaml.safe_load(config.read_text())

    result = boilerplate.describe_regression(
        confounds_config=config,
        motion_filter_type=None,
        motion_filter_order=0,
        band_stop_min=0,
        band_stop_max=0,
        TR=0.8,
        fd_thresh=0,
    )
    assert isinstance(result, str)
    assert match in result

    return result


def test_describe_atlases():
    """Test boilerplate.describe_atlases."""
    atlases = ["4S156Parcels", "4S256Parcels", "Glasser"]
    atlas_desc = boilerplate.describe_atlases(atlases)
    assert "156 and 256 parcels" in atlas_desc
    assert "Glasser" in atlas_desc

    atlases = ["Glasser", "Tian"]
    atlas_desc = boilerplate.describe_atlases(atlases)
    assert "Tian" in atlas_desc
    assert "Glasser" in atlas_desc

    # This no longer fails. It just adds the missing atlas to the description.
    atlases = ["4S156Parcels", "4S256Parcels", "Glasser", "fail"]
    assert "the fail atlas" in boilerplate.describe_atlases(atlases)
