"""Tests for xcp_d.utils.confounds."""

import re

import numpy as np
import pytest
from scipy import signal

from xcp_d.utils import confounds


def test_modify_motion_filter():
    """Run a simple test of the motion filter modification function."""
    band_stop_min = 6
    TR = 0.8

    with pytest.warns(match="The parameter 'band_stop_max' will be ignored."):
        band_stop_min2, _, is_modified = confounds._modify_motion_filter(
            motion_filter_type='lp',
            band_stop_min=band_stop_min,
            band_stop_max=18,
            TR=TR,
        )
    assert band_stop_min2 == band_stop_min
    assert is_modified is False

    # Use freq above Nyquist, and function will automatically modify the filter.
    with pytest.warns(
        UserWarning,
        match=re.escape(
            'Low-pass filter frequency is above Nyquist frequency (37.5 BPM), '
            'so it has been changed (42 --> 33.0 BPM).'
        ),
    ):
        band_stop_min2, _, is_modified = confounds._modify_motion_filter(
            TR=TR,  # 1.25 Hz
            motion_filter_type='lp',
            band_stop_min=42,  # 0.7 Hz > (1.25 / 2)
            band_stop_max=None,
        )
    assert band_stop_min2 == 33.0
    assert is_modified is True

    # Now test band-stop filter
    # Use band above Nyquist, and function will automatically modify the filter.
    # NOTE: In this case, the min and max end up flipped for some reason.
    with pytest.warns(
        UserWarning,
        match=re.escape(
            'One or both filter frequencies are above Nyquist frequency (37.5 BPM), '
            'so they have been changed (42 --> 33.0, 45 --> 30.0 BPM).'
        ),
    ):
        band_stop_min2, band_stop_max2, is_modified = confounds._modify_motion_filter(
            TR=TR,
            motion_filter_type='notch',
            band_stop_min=42,
            band_stop_max=45,  # 0.7 Hz > (1.25 / 2)
        )

    assert band_stop_min2 == 33.0
    assert band_stop_max2 == 30.0
    assert is_modified is True

    # Notch without modification
    band_stop_min2, band_stop_max2, is_modified = confounds._modify_motion_filter(
        TR=TR,
        motion_filter_type='notch',
        band_stop_min=30,
        band_stop_max=33,
    )

    assert band_stop_min2 == 30
    assert band_stop_max2 == 33
    assert is_modified is False


def test_motion_filtering_lp():
    """Run low-pass filter on toy data, compare to simplified results."""
    raw_data = np.random.random(500)

    band_stop_min = 6
    TR = 0.8

    low_pass = band_stop_min / 60
    b, a = signal.butter(
        1,
        low_pass,
        btype='lowpass',
        output='ba',
        fs=1 / TR,
    )
    lowpass_data_true = signal.filtfilt(
        b,
        a,
        raw_data,
        padtype='constant',
        padlen=raw_data.size - 1,
    )

    # Confirm the LP filter runs with reasonable parameters
    raw_data = raw_data[:, None]  # add singleton row dimension
    lowpass_data_test = confounds.filter_motion(
        raw_data,
        TR=TR,
        motion_filter_type='lp',
        band_stop_min=band_stop_min,
        band_stop_max=None,
        motion_filter_order=2,
    )

    # What's the difference from the verified data?
    assert np.allclose(np.squeeze(lowpass_data_test), lowpass_data_true)

    # Using a filter type other than notch or lp should raise an exception.
    with pytest.raises(ValueError, match="Motion filter type 'fail' not supported."):
        confounds.filter_motion(
            raw_data,
            TR=TR,
            motion_filter_type='fail',
            band_stop_min=band_stop_min,
            band_stop_max=None,
            motion_filter_order=2,
        )


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
    notch_data_true = signal.filtfilt(
        b,
        a,
        raw_data,
        padtype='constant',
        padlen=raw_data.size - 1,
    )

    # Repeat for notch filter
    raw_data = raw_data[:, None]  # add singleton row dimension
    notch_data_test = confounds.filter_motion(
        raw_data,
        TR=TR,  # 1.25 Hz
        motion_filter_type='notch',
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_order=4,
    )
    notch_data_test = np.squeeze(notch_data_test)
    assert np.allclose(notch_data_test, notch_data_true)
