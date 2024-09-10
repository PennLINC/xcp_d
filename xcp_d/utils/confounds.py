# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Confound matrix selection based on Ciric et al. 2007."""
import os
import warnings

import numpy as np
import pandas as pd
from nipype import logging
from scipy.signal import butter, filtfilt, iirnotch

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


def volterra(df):
    """Perform Volterra expansion."""
    # Ignore pandas SettingWithCopyWarning
    with pd.option_context("mode.chained_assignment", None):
        columns = df.columns.tolist()
        for col in columns:
            new_col = f"{col}_derivative1"
            df[new_col] = df[col].diff()

        columns = df.columns.tolist()
        for col in columns:
            new_col = f"{col}_power2"
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

    References
    ----------
    .. footbibliography::
    """
    if motion_filter_type not in ("lp", "notch", None):
        raise ValueError(f"Motion filter type '{motion_filter_type}' not supported.")

    # Select the motion columns from the overall confounds DataFrame
    motion_confounds_df = confounds_df[
        ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    ]

    # Apply LP or notch filter
    if motion_filter_type in ("lp", "notch"):
        motion_confounds = filter_motion(
            data=motion_confounds_df.to_numpy(),
            TR=TR,
            motion_filter_type=motion_filter_type,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_order=motion_filter_order,
        )
        motion_confounds_df = pd.DataFrame(
            data=motion_confounds,
            columns=motion_confounds_df.columns,
        )

    return motion_confounds_df


def load_confounds(
    confounds_dict,
    confound_config,
    n_volumes,
    TR,
    motion_filter_type=None,
    band_stop_min=None,
    band_stop_max=None,
    motion_filter_order=4,
):
    """Load confounds according to a configuration file.

    This function basically checks that each confounds file has the right number of volumes,
    selects the requisite columns from each input tabular file, and puts those columns into a
    single tabular file.

    Parameters
    ----------
    confounds_dict : dict
        Dictionary of confound names and path to corresponding files.
    confound_config : dict
        Configuration file for confounds.
    n_volumes : int
        Number of volumes in the fMRI data.

    Returns
    -------
    confounds_tsv : str or None
        Path to the TSV file containing combined tabular confounds.
        None if no tabular confounds are present.
    confounds_images : list of str
        List of paths to the voxelwise confounds images.
    """
    import re

    import nibabel as nb
    import pandas as pd

    new_confound_df = pd.DataFrame(index=np.arange(n_volumes))

    confounds_images = []
    for confound_name, confound_file in confounds_dict.items():
        confound_params = confound_config["confounds"][confound_name]
        if "columns" in confound_params:
            # Tabular confounds
            confound_df = pd.read_table(confound_file)
            if confound_df.shape[0] != n_volumes:
                raise ValueError(
                    f"Number of volumes in confounds file ({confound_df.shape[0]}) "
                    f"does not match number of volumes in the fMRI data ({n_volumes})."
                )

            available_columns = confound_df.columns.tolist()
            required_columns = confound_params["columns"]
            for column in required_columns:
                if column.startswith("^"):
                    # Regular expression
                    found_columns = [
                        col_name
                        for col_name in available_columns
                        if re.match(column, col_name, re.IGNORECASE)
                    ]
                    if not found_columns:
                        raise ValueError(
                            f"No columns found matching regular expression '{column}'"
                        )

                    for found_column in found_columns:
                        if found_column in new_confound_df:
                            raise ValueError(
                                f"Duplicate column name ({found_column}) in confounds "
                                "configuration."
                            )

                        new_confound_df[found_column] = confound_df[found_column]
                else:
                    if column not in confound_df.columns:
                        raise ValueError(f"Column '{column}' not found in confounds file.")

                    if column in new_confound_df:
                        raise ValueError(
                            f"Duplicate column name ({column}) in confounds configuration."
                        )

                    new_confound_df[column] = confound_df[column]
        else:
            # Voxelwise confounds
            confound_img = nb.load(confound_file)
            if confound_img.ndim == 2:  # CIFTI
                n_volumes_check = confound_img.shape[0]
            else:
                n_volumes_check = confound_img.shape[3]

            if n_volumes_check != n_volumes:
                raise ValueError(
                    f"Number of volumes in confounds image ({n_volumes_check}) "
                    f"does not match number of volumes in the fMRI data ({n_volumes})."
                )

            confounds_images.append(confound_file)

    if new_confound_df.empty:
        return None, confounds_images

    if motion_filter_type:
        # Filter the motion parameters
        # 1. Pop out the 6 basic motion parameters
        # 2. Filter them
        # 3. Calculate the Volterra expansion of the filtered parameters
        # 4. For each selected motion confound, remove that column and replace with the filtered
        #    version. Include `_filtered` in the new column name.
        motion_params = ["trans_x", "trans_y", "tran_z", "rot_x", "rot_y", "rot_z"]
        motion_based_params = [
            c for c in new_confound_df.columns if any(c.startswith(p) for p in motion_params)
        ]
        if len(motion_based_params):
            # Motion-based regressors detected
            base_motion_columns = [c for c in new_confound_df.columns if c in motion_params]
            motion_df = new_confound_df[base_motion_columns]
            motion_df.values = filter_motion(
                data=motion_df.to_numpy(),
                TR=TR,
                motion_filter_type=motion_filter_type,
                band_stop_min=band_stop_min,
                band_stop_max=band_stop_max,
                motion_filter_order=motion_filter_order,
            )
            motion_df = volterra(motion_df)
            overlapping_columns = [c for c in new_confound_df.columns if c in motion_df.columns]
            motion_unfiltered = [c for c in motion_based_params if c not in overlapping_columns]
            if motion_unfiltered:
                raise ValueError(f"Motion-based regressors {motion_unfiltered} were not filtered.")

            # Select the relevant filtered motion parameter columns
            motion_df = motion_df[overlapping_columns]
            motion_df.columns = [f"{c}_filtered" for c in motion_df.columns]

            # Replace the original motion columns with the filtered versions
            new_confound_df.drop(columns=overlapping_columns, inplace=True)
            new_confound_df = pd.concat([new_confound_df, motion_df], axis=1)

    confounds_tsv = os.path.join(os.getcwd(), "confounds.tsv")
    new_confound_df.to_csv(confounds_tsv, sep="\t", index=False)
    return confounds_tsv, confounds_images


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
    The original paper also used zero-padding with a padding size of 100.
    We use constant-padding, with the default padding size determined by
    :func:`scipy.signal.filtfilt`.

    Band-stop filtering (``motion_filter_type = "notch"``) is performed with a notch filter,
    as in :footcite:t:`fair2020correction`.
    This filter uses the mean of the stopband frequencies as the target frequency,
    and the range between the two frequencies as the bandwidth.
    The filter is applied with constant-padding, using the default padding size determined by
    :func:`scipy.signal.filtfilt`.

    References
    ----------
    .. footbibliography::
    """
    if motion_filter_type not in ("lp", "notch"):
        raise ValueError(f"Motion filter type '{motion_filter_type}' not supported.")

    lowpass_hz = band_stop_min / 60

    sampling_frequency = 1 / TR

    if motion_filter_type == "lp":  # low-pass filter
        b, a = butter(
            motion_filter_order / 2,
            lowpass_hz,
            btype="lowpass",
            output="ba",
            fs=sampling_frequency,
        )
        filtered_data = filtfilt(b, a, data, axis=0, padtype="constant", padlen=data.shape[0] - 1)

    else:  # notch filter
        highpass_hz = band_stop_max / 60
        stopband_hz = np.array([lowpass_hz, highpass_hz])
        # Convert stopband to a single notch frequency.
        freq_to_remove = np.mean(stopband_hz)
        bandwidth = np.abs(np.diff(stopband_hz))

        # Create filter coefficients.
        b, a = iirnotch(freq_to_remove, freq_to_remove / bandwidth, fs=sampling_frequency)
        n_filter_applications = int(np.floor(motion_filter_order / 2))

        filtered_data = data.copy()
        for _ in range(n_filter_applications):
            filtered_data = filtfilt(
                b,
                a,
                filtered_data,
                axis=0,
                padtype="constant",
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
    if motion_filter_type == "lp":  # low-pass filter
        # Remove any frequencies above band_stop_min.
        assert band_stop_min is not None
        assert band_stop_min > 0
        if band_stop_max:
            warnings.warn("The parameter 'band_stop_max' will be ignored.")

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
                f"Low-pass filter frequency is above Nyquist frequency ({nyquist_bpm} BPM), "
                f"so it has been changed ({band_stop_min} --> {band_stop_min_adjusted} BPM)."
            )
            is_modified = True

        band_stop_max_adjusted = None

    elif motion_filter_type == "notch":  # notch filter
        # Retain any frequencies *outside* the band_stop_min-band_stop_max range.
        assert band_stop_max is not None
        assert band_stop_min is not None
        assert band_stop_max > 0
        assert band_stop_min > 0
        assert band_stop_min < band_stop_max, f"{band_stop_min} >= {band_stop_max}"

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
                f"One or both filter frequencies are above Nyquist frequency ({nyquist_bpm} BPM), "
                "so they have been changed "
                f"({stopband[0]} --> {stopband_adjusted[0]}, "
                f"{stopband[1]} --> {stopband_adjusted[1]} BPM)."
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
    if dummy_scans == "auto":
        confounds_df = pd.read_table(confounds_file)

        nss_cols = [c for c in confounds_df.columns if c.startswith("non_steady_state_outlier")]

        if nss_cols:
            initial_volumes_df = confounds_df[nss_cols]
            dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
            dummy_scans = np.where(dummy_scans)[0]

            # reasonably assumes all NSS volumes are contiguous
            dummy_scans = int(dummy_scans[-1] + 1)
            LOGGER.info(f"Found {dummy_scans} dummy scans in {os.path.basename(confounds_file)}")

        else:
            LOGGER.warning(f"No non-steady-state outliers found in {confounds_file}")
            dummy_scans = 0

    return dummy_scans
