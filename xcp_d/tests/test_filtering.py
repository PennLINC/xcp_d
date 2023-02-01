"""Tests for filtering methods."""
import numpy as np
import pytest
from scipy import signal

from xcp_d.interfaces.filtering import butter_bandpass
from xcp_d.utils.confounds import motion_regression_filter


def test_motion_filtering_lp():
    """Run lowpass filter on toy data, compare to simplified results."""
    raw_data = np.random.random(500)

    band_stop_min = 6
    TR = 0.8

    lowpass = band_stop_min / 60
    b, a = signal.butter(
        1,
        lowpass,
        btype="lowpass",
        output="ba",
        fs=1 / TR,
    )
    lowpass_data_true = signal.filtfilt(b, a, raw_data, padtype="constant")

    # Confirm the LP filter runs with reasonable parameters
    raw_data = raw_data[:, None]  # add singleton row dimension
    with pytest.warns(match="The parameter 'band_stop_max' will be ignored."):
        lowpass_data_test = motion_regression_filter(
            raw_data,
            TR=TR,
            motion_filter_type="lp",
            band_stop_min=band_stop_min,
            band_stop_max=None,
            motion_filter_order=2,
        )

    # What's the difference from the verified data?
    assert np.allclose(np.squeeze(lowpass_data_test), lowpass_data_true)


def test_motion_filtering_notch():
    """Run notch filter on toy data, compare to simplified results."""
    raw_data = np.random.random(500)

    band_stop_min, band_stop_max = 12, 20
    TR = 0.8

    lowcut, highcut = band_stop_min / 60, band_stop_max / 60
    stopband_hz_adjusted = [lowcut, highcut]
    freq_to_remove = np.mean(stopband_hz_adjusted)
    bandwidth = np.abs(np.diff(stopband_hz_adjusted))

    # Create filter coefficients.
    b, a = signal.iirnotch(freq_to_remove, freq_to_remove / bandwidth, fs=1 / TR)
    notch_data_true = signal.filtfilt(b, a, raw_data, padtype="constant")

    # Repeat for notch filter
    raw_data = raw_data[:, None]  # add singleton row dimension
    notch_data_test = motion_regression_filter(
        raw_data,
        TR=TR,
        motion_filter_type="notch",
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_order=2,
    )
    assert np.allclose(np.squeeze(notch_data_test), notch_data_true)


def test_bandpass_filtering():
    """Run Butterworth on toy data, compare to results that have been verified."""
    raw_data = np.random.random(500)

    highpass, lowpass = 0.009, 0.08

    b, a = signal.butter(
        1,
        [highpass, lowpass],
        btype="bandpass",
        output="ba",
    )
    butterworth_data_true = signal.filtfilt(b, a, raw_data, padtype="constant")

    # Confirm the butterworth filter runs with reasonable parameters
    raw_data = raw_data[:, None]  # add singleton row dimension
    butterworth_data_test = butter_bandpass(
        raw_data,
        fs=1 / 0.8,
        highpass=highpass,
        lowpass=lowpass,
        order=2,
    )
    assert np.allclose(np.squeeze(butterworth_data_test), butterworth_data_true)
