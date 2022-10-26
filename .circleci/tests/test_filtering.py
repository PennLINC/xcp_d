"""Tests for filtering methods."""
import os.path as op

import numpy as np
import pytest

from xcp_d.utils.confounds import motion_regression_filter


@pytest.fixture(scope="module")
def data_files():
    """Grab files for testing filters.

    These files were kindly provided by the DCAN lab to directly compare their
    MATLAB filtering results to our Python filtering results.

    We have converted the single-column CSV files provided to TXT files for easier I/O.
    """
    data_path = op.abspath(op.join(op.dirname(__file__), "data"))
    data_files_ = {
        "raw_data": op.join(data_path, "raw_data.txt"),
        "lowpass_filtered": op.join(data_path, "low_passed_MR_data.txt"),
        "notch_filtered": op.join(data_path, "notched_MR_data.txt"),
        "butterworth_filtered": op.join(data_path, "band_passed_bold_data.txt"),
    }
    return data_files_


def test_motion_filtering_lp(data_files):
    """Run lowpass filter on toy data, compare to results that have been verified."""
    raw_data = np.loadtxt(data_files["raw_data"])
    lowpass_data_true = np.loadtxt(data_files["lowpass_filtered"])
    raw_data = raw_data[None, :]  # add singleton row dimension

    band_stop_min = 12
    band_stop_max = 20

    # Confirm the LP filter runs with reasonable parameters
    with pytest.warns(match="The parameter 'band_stop_max' will be ignored."):
        lowpass_data_test = motion_regression_filter(
            raw_data,
            TR=0.8,
            motion_filter_type="lp",
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_order=2,
        )

    # What's the difference from the verified data?
    assert np.allclose(lowpass_data_test, lowpass_data_true, atol=1e-4)


def test_motion_filtering_notch(data_files):
    """Run notch filter on toy data, compare to results that have been verified.

    Notes
    -----
    This test requires a much more liberal tolerance value, because the notch filter
    doesn't replicate well across languages.
    """
    raw_data = np.loadtxt(data_files["raw_data"])
    notch_data_true = np.loadtxt(data_files["notch_filtered"])
    raw_data = raw_data[None, :]  # add singleton row dimension

    band_stop_min = 12
    band_stop_max = 20

    # Repeat for notch filter
    notch_data_test = motion_regression_filter(
        raw_data,
        TR=0.8,
        motion_filter_type="notch",
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_order=2,
    )
    assert np.allclose(notch_data_test, notch_data_true, atol=1e-1)
