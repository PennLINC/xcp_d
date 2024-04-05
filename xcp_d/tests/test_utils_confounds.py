"""Tests for xcp_d.utils.confounds."""

import os
import re

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy import signal

from xcp_d.utils import confounds


def test_custom_confounds(ds001419_data, tmp_path_factory):
    """Ensure that custom confounds can be loaded without issue."""
    tempdir = tmp_path_factory.mktemp("test_custom_confounds")
    bold_file = ds001419_data["nifti_file"]
    confounds_file = ds001419_data["confounds_file"]
    confounds_json = ds001419_data["confounds_json"]

    N_VOLUMES = 60
    TR = 2.5

    frame_times = np.arange(N_VOLUMES) * TR
    events_df = pd.DataFrame(
        {
            "onset": [10, 30, 50],
            "duration": [5, 10, 5],
            "trial_type": (["condition01"] * 2) + (["condition02"] * 1),
        },
    )
    custom_confounds = make_first_level_design_matrix(
        frame_times,
        events_df,
        drift_model=None,
        hrf_model="spm",
        high_pass=None,
    )
    # The design matrix will include a constant column, which we should drop
    custom_confounds = custom_confounds.drop(columns="constant")

    # Save to file
    custom_confounds_file = os.path.join(
        tempdir,
        "sub-01_task-rest_desc-confounds_timeseries.tsv",
    )
    custom_confounds.to_csv(custom_confounds_file, sep="\t", index=False)

    combined_confounds, _ = confounds.load_confound_matrix(
        params="24P",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
        custom_confounds=custom_confounds_file,
    )
    # We expect n params + 2 (one for each condition in custom confounds)
    assert combined_confounds.shape == (N_VOLUMES, 26)
    assert "condition01" in combined_confounds.columns
    assert "condition02" in combined_confounds.columns

    custom_confounds, _ = confounds.load_confound_matrix(
        params="custom",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
        custom_confounds=custom_confounds_file,
    )
    # We expect 2 (one for each condition in custom confounds)
    assert combined_confounds.shape == (N_VOLUMES, 26)
    assert "condition01" in combined_confounds.columns
    assert "condition02" in combined_confounds.columns


def test_load_confounds(ds001419_data):
    """Ensure that xcp_d loads the right confounds."""
    bold_file = ds001419_data["nifti_file"]
    confounds_file = ds001419_data["confounds_file"]
    confounds_json = ds001419_data["confounds_json"]

    N_VOLUMES = 60

    confounds_df, _ = confounds.load_confound_matrix(
        params="24P",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert confounds_df.shape == (N_VOLUMES, 24)

    confounds_df, _ = confounds.load_confound_matrix(
        params="27P",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert confounds_df.shape == (N_VOLUMES, 27)

    confounds_df, _ = confounds.load_confound_matrix(
        params="36P",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert confounds_df.shape == (N_VOLUMES, 36)

    confounds_df, _ = confounds.load_confound_matrix(
        params="acompcor",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert confounds_df.shape == (N_VOLUMES, 28)

    confounds_df, _ = confounds.load_confound_matrix(
        params="acompcor_gsr",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert confounds_df.shape == (N_VOLUMES, 29)

    confounds_df, _ = confounds.load_confound_matrix(
        params="aroma",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert confounds_df.shape == (N_VOLUMES, 48)

    confounds_df, _ = confounds.load_confound_matrix(
        params="aroma_gsr",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert confounds_df.shape == (N_VOLUMES, 49)

    confounds_df, _ = confounds.load_confound_matrix(
        params="none",
        img_file=bold_file,
        confounds_file=confounds_file,
        confounds_json_file=confounds_json,
    )
    assert not confounds_df

    with pytest.raises(ValueError, match="Unrecognized parameter string"):
        confounds.load_confound_matrix(
            params="test",
            img_file=bold_file,
            confounds_file=confounds_file,
            confounds_json_file=confounds_json,
        )

    with pytest.raises(ValueError):
        confounds.load_confound_matrix(
            params="custom",
            img_file=bold_file,
            confounds_file=confounds_file,
            confounds_json_file=confounds_json,
        )


def test_modify_motion_filter():
    """Run a simple test of the motion filter modification function."""
    band_stop_min = 6
    TR = 0.8

    with pytest.warns(match="The parameter 'band_stop_max' will be ignored."):
        band_stop_min2, _, is_modified = confounds._modify_motion_filter(
            motion_filter_type="lp",
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
            "Low-pass filter frequency is above Nyquist frequency (37.5 BPM), "
            "so it has been changed (42 --> 33.0 BPM)."
        ),
    ):
        band_stop_min2, _, is_modified = confounds._modify_motion_filter(
            TR=TR,  # 1.25 Hz
            motion_filter_type="lp",
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
            "One or both filter frequencies are above Nyquist frequency (37.5 BPM), "
            "so they have been changed (42 --> 33.0, 45 --> 30.0 BPM)."
        ),
    ):
        band_stop_min2, band_stop_max2, is_modified = confounds._modify_motion_filter(
            TR=TR,
            motion_filter_type="notch",
            band_stop_min=42,
            band_stop_max=45,  # 0.7 Hz > (1.25 / 2)
        )

    assert band_stop_min2 == 33.0
    assert band_stop_max2 == 30.0
    assert is_modified is True

    # Notch without modification
    band_stop_min2, band_stop_max2, is_modified = confounds._modify_motion_filter(
        TR=TR,
        motion_filter_type="notch",
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
        btype="lowpass",
        output="ba",
        fs=1 / TR,
    )
    lowpass_data_true = signal.filtfilt(
        b,
        a,
        raw_data,
        padtype="constant",
        padlen=raw_data.size - 1,
    )

    # Confirm the LP filter runs with reasonable parameters
    raw_data = raw_data[:, None]  # add singleton row dimension
    lowpass_data_test = confounds.motion_regression_filter(
        raw_data,
        TR=TR,
        motion_filter_type="lp",
        band_stop_min=band_stop_min,
        band_stop_max=None,
        motion_filter_order=2,
    )

    # What's the difference from the verified data?
    assert np.allclose(np.squeeze(lowpass_data_test), lowpass_data_true)

    # Using a filter type other than notch or lp should raise an exception.
    with pytest.raises(ValueError, match="Motion filter type 'fail' not supported."):
        confounds.motion_regression_filter(
            raw_data,
            TR=TR,
            motion_filter_type="fail",
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
        padtype="constant",
        padlen=raw_data.size - 1,
    )

    # Repeat for notch filter
    raw_data = raw_data[:, None]  # add singleton row dimension
    notch_data_test = confounds.motion_regression_filter(
        raw_data,
        TR=TR,  # 1.25 Hz
        motion_filter_type="notch",
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_order=2,
    )
    notch_data_test = np.squeeze(notch_data_test)
    assert np.allclose(notch_data_test, notch_data_true)
