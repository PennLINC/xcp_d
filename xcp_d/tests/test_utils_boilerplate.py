"""Test confounds handling."""

import os

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix

from xcp_d.utils import boilerplate


def test_describe_motion_parameters():
    """Test boilerplate.describe_motion_parameters."""
    desc = boilerplate.describe_motion_parameters(
        motion_filter_type=None,
        motion_filter_order=None,
        band_stop_min=None,
        band_stop_max=None,
        head_radius=50,
        TR=0.8,
    )
    assert "filtered to remove signals" not in desc
    assert "Framewise displacement was calculated" in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="notch",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        head_radius=50,
        TR=0.8,
    )
    assert "band-stop filtered to remove signals" in desc
    assert "automatically modified" not in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="notch",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        head_radius=50,
        TR=3,
    )
    assert "band-stop filtered to remove signals" in desc
    assert "automatically modified" in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="lp",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        head_radius=50,
        TR=0.8,
    )
    assert "low-pass filtered" in desc
    assert "automatically modified" not in desc

    desc = boilerplate.describe_motion_parameters(
        motion_filter_type="lp",
        motion_filter_order=1,
        band_stop_min=12,
        band_stop_max=20,
        head_radius=50,
        TR=3,
    )
    assert "low-pass filtered" in desc
    assert "automatically modified" in desc


def test_describe_censoring():
    """Test boilerplate.describe_censoring."""
    motion_filter_type = "notch"
    fd_thresh = 0.2
    exact_scans = []
    desc = boilerplate.describe_censoring(motion_filter_type, fd_thresh, exact_scans)
    assert "Volumes with filtered framewise displacement" in desc

    motion_filter_type = None
    fd_thresh = 0.2
    exact_scans = []
    desc = boilerplate.describe_censoring(motion_filter_type, fd_thresh, exact_scans)
    assert "Volumes with framewise displacement" in desc

    motion_filter_type = None
    fd_thresh = 0.2
    exact_scans = [100, 200, 300]
    desc = boilerplate.describe_censoring(motion_filter_type, fd_thresh, exact_scans)
    assert "Volumes with framewise displacement" in desc
    assert "limited to 100, 200, and 300 volumes" in desc

    motion_filter_type = None
    fd_thresh = 0
    exact_scans = [100, 200, 300]
    desc = boilerplate.describe_censoring(motion_filter_type, fd_thresh, exact_scans)
    assert "Volumes were randomly selected for censoring" in desc
    assert "limited to 100, 200, and 300 volumes" in desc

    motion_filter_type = "notch"
    fd_thresh = 0
    exact_scans = [100, 200, 300]
    desc = boilerplate.describe_censoring(motion_filter_type, fd_thresh, exact_scans)
    assert "Volumes were randomly selected for censoring" in desc
    assert "limited to 100, 200, and 300 volumes" in desc


def test_describe_regression(tmp_path_factory):
    """Test boilerplate.describe_regression."""
    tempdir = tmp_path_factory.mktemp("test_describe_regression")
    N_VOLUMES, TR = 60, 2.5

    events_df = pd.DataFrame(
        {
            "onset": [10, 30, 50],
            "duration": [5, 10, 5],
            "trial_type": (["signal__condition01"] * 2) + (["condition02"] * 1),
        },
    )
    custom_confounds = make_first_level_design_matrix(
        np.arange(N_VOLUMES) * TR,
        events_df,
        drift_model=None,
        hrf_model="spm",
        high_pass=None,
    )
    custom_confounds = custom_confounds.drop(columns="constant")
    custom_confounds_file = os.path.join(tempdir, "file.tsv")
    custom_confounds.to_csv(custom_confounds_file, sep="\t", index=False)

    _check_describe_regression_result("24P", "24 nuisance regressors were selected")
    _check_describe_regression_result("27P", "27 nuisance regressors were selected")
    _check_describe_regression_result("36P", "36 nuisance regressors were selected")
    _check_describe_regression_result("acompcor", "The top 5 aCompCor principal components")
    _check_describe_regression_result("acompcor_gsr", "The top 5 aCompCor principal components")
    _check_describe_regression_result("aroma", "AROMA motion-labeled components")
    _check_describe_regression_result("aroma_gsr", "AROMA motion-labeled components")
    _check_describe_regression_result("custom", "A custom set of regressors was used")
    _check_describe_regression_result("none", "No nuisance regression was performed")

    with pytest.raises(ValueError, match="Unrecognized parameter string"):
        boilerplate.describe_regression(
            params="test",
            custom_confounds_file=None,
            motion_filter_type=None,
        )

    # Test with custom confounds file
    desc = boilerplate.describe_regression(
        params="24P",
        custom_confounds_file=custom_confounds_file,
        motion_filter_type=None,
    )
    assert isinstance(desc, str)
    assert "custom confounds were also included" in desc
    assert "24 nuisance regressors were selected" in desc

    desc = boilerplate.describe_regression(
        params="custom",
        custom_confounds_file=custom_confounds_file,
        motion_filter_type=None,
    )
    assert isinstance(desc, str)
    assert "A custom set of regressors was used" in desc


def _check_describe_regression_result(params, match):
    result = boilerplate.describe_regression(
        params=params,
        custom_confounds_file=None,
        motion_filter_type=None,
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

    atlases = ["4S156Parcels", "4S256Parcels", "Glasser", "fail"]
    with pytest.raises(ValueError, match="Unrecognized atlas"):
        boilerplate.describe_atlases(atlases)
