"""Test confounds handling."""

import pytest

from xcp_d.utils import boilerplate


def test_describe_motion_parameters():
    """Test boilerplate.describe_motion_parameters."""
    pass


def test_describe_censoring():
    """Test boilerplate.describe_censoring."""
    pass


def test_describe_regression():
    """Ensure that xcp_d loads the right confounds."""
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
        custom_confounds_file="file.tsv",
        motion_filter_type=None,
    )
    assert isinstance(desc, str)
    assert "custom confounds were also included" in desc
    assert "24 nuisance regressors were selected" in desc

    desc = boilerplate.describe_regression(
        params="custom",
        custom_confounds_file="file.tsv",
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
