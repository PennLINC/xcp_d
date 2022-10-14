"""Tests for filtering methods."""
import numpy as np
import pandas as pd

from xcp_d.interfaces.filtering import butter_bandpass
from xcp_d.utils.confounds import motion_regression_filter


def test_motion_filtering():
    """Run LP/Notch on toy data, compare to results that have been verified."""
    raw_data_file = "data/raw_data.csv"
    raw_data_df = pd.read_table(raw_data_file, header=None)
    raw_data = raw_data_df.to_numpy().T.copy()

    lowpass_file = "data/low_passed_MR_data.csv"
    lowpass_data_df = pd.read_table(lowpass_file, header=None)
    notch_file = "data/notched_MR_data.csv"
    notch_data_df = pd.read_table(notch_file, header=None)

    band_stop_min = 12
    band_stop_max = 20

    # Confirm the LP filter runs with reasonable parameters
    LP_data = motion_regression_filter(
        raw_data,
        TR=0.8,
        motion_filter_type="lp",
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_order=2,
    )

    # What's the difference from the verified data?
    lp_data_comparator = lowpass_data_df.to_numpy().T
    assert np.allclose(LP_data, lp_data_comparator, atol=1e-4)

    # Repeat for notch filter
    notch_data = motion_regression_filter(
        raw_data,
        TR=0.8,
        motion_filter_type="notch",
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_order=2,
    )
    notch_data_comparator = notch_data_df.to_numpy().T
    assert np.allclose(notch_data, notch_data_comparator, atol=1e-4)


def test_bandpass_filtering():
    """Run Butterworth on toy data, compare to results that have been verified."""
    raw_data_file = "data/raw_data.csv"
    raw_data_df = pd.read_table(raw_data_file, header=None)
    raw_data = raw_data_df.to_numpy().T.copy()

    butterworth_file = "data/band_passed_bold_data.csv"
    butterworth_data_df = pd.read_table(butterworth_file, header=None)

    # Confirm the butterworth filter runs with reasonable parameters
    butterworth_data = butter_bandpass(
        raw_data,
        fs=1 / 0.8,
        highpass=0.009,
        lowpass=0.080,
        order=2,
    )
    butterworth_data_comparator = butterworth_data_df.to_numpy().T
    assert np.allclose(butterworth_data, butterworth_data_comparator, atol=1e-4)

