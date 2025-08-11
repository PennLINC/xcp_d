# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Confound matrix selection based on Ciric et al. 2007."""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from nipype import logging
from scipy.signal import butter, filtfilt, iirnotch

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger('nipype.utils')


def volterra(df):
    """Perform Volterra expansion."""
    # Ignore pandas SettingWithCopyWarning
    with pd.option_context('mode.chained_assignment', None):
        columns = df.columns.tolist()
        for col in columns:
            new_col = f'{col}_derivative1'
            df[new_col] = df[col].diff()

        columns = df.columns.tolist()
        for col in columns:
            new_col = f'{col}_power2'
            df[new_col] = df[col] ** 2

    return df


@fill_doc
def load_motion(
    confounds_df,
    TR,
    motion_filter_type=None,
    band_stop_min=None,
    band_stop_max=None,
    motion_filter_order=4,
):
    """Load the six basic motion regressors (three rotations, three translations).

    Parameters
    ----------
    confounds_df : pandas.DataFrame
        The confounds DataFrame from which to extract the six basic motion regressors.
    %(TR)s
    %(motion_filter_type)s
        If "lp" or "notch", that filtering will be done in this function.
        Otherwise, no filtering will be applied.
    %(band_stop_min)s
    %(band_stop_max)s
    %(motion_filter_order)s

    Returns
    -------
    motion_confounds_df : pandas.DataFrame
        The six motion regressors.
        The three rotations are listed first, then the three translations.
        Plus rmsd, and possibly filtered motion regressors.

    References
    ----------
    .. footbibliography::
    """
    if motion_filter_type not in ('lp', 'notch', None):
        raise ValueError(f"Motion filter type '{motion_filter_type}' not supported.")

    # Select the motion columns from the overall confounds DataFrame
    if isinstance(confounds_df, str | Path):
        confounds_df = pd.read_table(confounds_df)

    motion_confounds_df = confounds_df[
        ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
    ]

    # Apply LP or notch filter
    if motion_filter_type in ('lp', 'notch'):
        motion_confounds = filter_motion(
            data=motion_confounds_df.to_numpy(),
            TR=TR,
            motion_filter_type=motion_filter_type,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_order=motion_filter_order,
        )
        filtered_motion_confounds_df = pd.DataFrame(
            data=motion_confounds,
            columns=[f'{c}_filtered' for c in motion_confounds_df.columns],
        )
        motion_confounds_df = pd.concat(
            [motion_confounds_df, filtered_motion_confounds_df],
            axis=1,
        )

    # Add RMSD column (used for QC measures later on)
    motion_confounds_df = pd.concat(
        [motion_confounds_df, confounds_df[['rmsd']]],
        axis=1,
    )

    return motion_confounds_df


@fill_doc
def filter_motion(
    data,
    TR,
    motion_filter_type,
    band_stop_min,
    band_stop_max,
    motion_filter_order,
):
    """Filter translation and rotation motion parameters.

    Parameters
    ----------
    data : (T, R) numpy.ndarray
        Data to filter. T = time, R = motion regressors
        The filter will be applied independently to each variable, across time.
    %(TR)s
    %(motion_filter_type)s
        If not "notch" or "lp", an exception will be raised.
    %(band_stop_min)s
    %(band_stop_max)s
    %(motion_filter_order)s

    Returns
    -------
    data : (T, R) numpy.ndarray
        Filtered data. Same shape as the original data.

    Notes
    -----
    Low-pass filtering (``motion_filter_type = "lp"``) is performed with a Butterworth filter,
    as in :footcite:t:`gratton2020removal`.
    The order of the Butterworth filter is determined by ``motion_filter_order``,
    although the original paper used a first-order filter.
    Since filtfilt applies the filter twice, motion_filter_order is divided by 2 before applying
    the filter.
    The original paper also used zero-padding with a padding size of 100.
    We use constant-padding, with the default padding size determined by
    :func:`scipy.signal.filtfilt`.

    Band-stop filtering (``motion_filter_type = "notch"``) is performed with a notch filter,
    as in :footcite:t:`fair2020correction`.
    This filter uses the mean of the stopband frequencies as the target frequency,
    and the range between the two frequencies as the bandwidth.
    Because iirnotch is a second-order filter and filtfilt applies the filter twice,
    motion_filter_order is divided by 4 before applying the filter.
    The filter is applied with constant-padding, using the default padding size determined by
    :func:`scipy.signal.filtfilt`.

    References
    ----------
    .. footbibliography::
    """
    if motion_filter_type not in ('lp', 'notch'):
        raise ValueError(f"Motion filter type '{motion_filter_type}' not supported.")

    lowpass_hz = band_stop_min / 60

    sampling_frequency = 1 / TR

    if motion_filter_type == 'lp':  # low-pass filter
        n_filter_applications = int(np.floor(motion_filter_order / 2))
        b, a = butter(
            n_filter_applications,
            lowpass_hz,
            btype='lowpass',
            output='ba',
            fs=sampling_frequency,
        )
        filtered_data = filtfilt(b, a, data, axis=0, padtype='constant', padlen=data.shape[0] - 1)

    else:  # notch filter
        highpass_hz = band_stop_max / 60
        stopband_hz = np.array([lowpass_hz, highpass_hz])
        # Convert stopband to a single notch frequency.
        freq_to_remove = np.nanmean(stopband_hz)
        bandwidth = np.abs(np.diff(stopband_hz))

        # Create filter coefficients.
        b, a = iirnotch(freq_to_remove, freq_to_remove / bandwidth, fs=sampling_frequency)
        # Both iirnotch and filtfilt are second-order,
        # so we need to divide the motion_filter_order by 4.
        n_filter_applications = int(np.floor(motion_filter_order / 4))

        filtered_data = data.copy()
        for _ in range(n_filter_applications):
            filtered_data = filtfilt(
                b,
                a,
                filtered_data,
                axis=0,
                padtype='constant',
                padlen=data.shape[0] - 1,
            )

    return filtered_data


def _modify_motion_filter(motion_filter_type, band_stop_min, band_stop_max, TR):
    """Modify the motion filter parameters based on the TR.

    Parameters
    ----------
    motion_filter_type : str
        The type of motion filter to apply.
    band_stop_min : float
        The minimum frequency to stop in the filter, in breaths-per-minute.
    band_stop_max : float
        The maximum frequency to stop in the filter, in breaths-per-minute.
    TR : float
        The repetition time of the data.

    Returns
    -------
    band_stop_min_adjusted : float
        The adjusted low-pass filter frequency, in breaths-per-minute.
    band_stop_max_adjusted : float
        The adjusted high-pass filter frequency, in breaths-per-minute.
    is_modified : bool
        Whether the filter parameters were modified.
    """
    sampling_frequency = 1 / TR
    nyquist_frequency = sampling_frequency / 2
    nyquist_bpm = nyquist_frequency * 60

    is_modified = False
    if motion_filter_type == 'lp':  # low-pass filter
        # Remove any frequencies above band_stop_min.
        assert band_stop_min is not None
        assert band_stop_min > 0
        if band_stop_max:
            warnings.warn("The parameter 'band_stop_max' will be ignored.", stacklevel=2)

        lowpass_hz = band_stop_min / 60  # change BPM to right time unit

        # Adjust frequency in case Nyquist is below cutoff.
        # This won't have an effect if the data have a fast enough sampling rate.
        lowpass_hz_adjusted = np.abs(
            lowpass_hz
            - (np.floor((lowpass_hz + nyquist_frequency) / sampling_frequency))
            * sampling_frequency
        )
        band_stop_min_adjusted = lowpass_hz_adjusted * 60  # change Hertz back to BPM
        if band_stop_min_adjusted != band_stop_min:
            warnings.warn(
                f'Low-pass filter frequency is above Nyquist frequency ({nyquist_bpm} BPM), '
                f'so it has been changed ({band_stop_min} --> {band_stop_min_adjusted} BPM).',
                stacklevel=2,
            )
            is_modified = True

        band_stop_max_adjusted = None

    elif motion_filter_type == 'notch':  # notch filter
        # Retain any frequencies *outside* the band_stop_min-band_stop_max range.
        assert band_stop_max is not None
        assert band_stop_min is not None
        assert band_stop_max > 0
        assert band_stop_min > 0
        assert band_stop_min < band_stop_max, f'{band_stop_min} >= {band_stop_max}'

        stopband = np.array([band_stop_min, band_stop_max])
        stopband_hz = stopband / 60  # change BPM to Hertz

        # Adjust frequencies in case Nyquist is within/below band.
        # This won't have an effect if the data have a fast enough sampling rate.
        stopband_hz_adjusted = np.abs(
            stopband_hz
            - (np.floor((stopband_hz + nyquist_frequency) / sampling_frequency))
            * sampling_frequency
        )
        stopband_adjusted = stopband_hz_adjusted * 60  # change Hertz back to BPM
        if not np.array_equal(stopband_adjusted, stopband):
            warnings.warn(
                f'One or both filter frequencies are above Nyquist frequency ({nyquist_bpm} BPM), '
                'so they have been changed '
                f'({stopband[0]} --> {stopband_adjusted[0]}, '
                f'{stopband[1]} --> {stopband_adjusted[1]} BPM).',
                stacklevel=2,
            )
            is_modified = True

        band_stop_min_adjusted, band_stop_max_adjusted = stopband_adjusted
    else:
        band_stop_min_adjusted, band_stop_max_adjusted, is_modified = None, None, False

    return band_stop_min_adjusted, band_stop_max_adjusted, is_modified


def _infer_dummy_scans(dummy_scans, confounds_file=None):
    """Attempt to determine the number of dummy scans from the confounds file.

    This function expects non-steady-state volumes flagged by the preprocessing pipeline
    to be indicated in columns starting with "non_steady_state_outlier" in the confounds file.

    Parameters
    ----------
    dummy_scans : "auto" or int
        The number of dummy scans.
        If an integer, this will be returned without modification.
        If "auto", the confounds file will be loaded and the number of non-steady-state
        volumes estimated by the preprocessing workflow will be determined.
    confounds_file : None or str
        Path to the confounds TSV file from the preprocessing pipeline.
        Only used if dummy_scans is "auto".
        Default is None.

    Returns
    -------
    dummy_scans : int
        Estimated number of dummy scans.
    """
    if dummy_scans == 'auto':
        confounds_df = pd.read_table(confounds_file)

        nss_cols = [c for c in confounds_df.columns if c.startswith('non_steady_state_outlier')]

        if nss_cols:
            initial_volumes_df = confounds_df[nss_cols]
            dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
            dummy_scans = np.where(dummy_scans)[0]

            # reasonably assumes all NSS volumes are contiguous
            dummy_scans = int(dummy_scans[-1] + 1)
            LOGGER.info(f'Found {dummy_scans} dummy scans in {os.path.basename(confounds_file)}')

        else:
            LOGGER.warning(f'No non-steady-state outliers found in {confounds_file}')
            dummy_scans = 0

    return dummy_scans
