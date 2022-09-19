# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for interpolating over high-motion volumes."""
import numpy as np


def compute_FD(confound, head_radius=50):
    """Compute framewise displacement.

    NOTE: TS- Which kind of FD? Power?
    NOTE: TS- What units must rotation parameters be in?

    Parameters
    ----------
    confound : pandas.DataFrame
        DataFrame with six motion parameters.
    head_radius : int, optional
        The head radius, in millimeters. Default is 50.

    Returns
    -------
    fdres : numpy.ndarray
        The framewise displacement time series.
    """
    confound = confound.replace(np.nan, 0)
    mpars = confound[[
        "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"
    ]].to_numpy()
    diff = mpars[:-1, :6] - mpars[1:, :6]
    diff[:, 3:6] *= head_radius
    fd_res = np.abs(diff).sum(axis=1)
    fdres = np.hstack([0, fd_res])

    return fdres


def generate_mask(fd_res, fd_thresh):
    """Create binary temporal mask flagging high-motion volumes.

    Parameters
    ----------
    fd_res : numpy.ndarray of shape (T)
        Framewise displacement time series.
        T = time.
    fd_thresh : float
        Threshold used to identify high-motion volumes.
        Any ``fd_res`` values greater than the threshold will be flagged.

    Returns
    -------
    tmask : numpy.ndarray of shape (T)
        The temporal mask. Zeros are low-motion volumes. Ones are high-motion volumes.
    """
    tmask = np.zeros(len(fd_res))
    tmask[fd_res > fd_thresh] = 1

    return tmask


def interpolate_masked_data(bold_data, tmask, TR=1):
    """Interpolate masked data.

    No interpolation will be performed if more than 50% of the volumes in the BOLD data are
    flagged by the temporal mask.

    NOTE: TS- Why are slice times being inferred from the number of volumes?
    Am I missing somehting?

    Parameters
    ----------
    bold_data : numpy.ndarray of shape (S, T)
        The BOLD data to interpolate.
        T = time, S = samples.
    tmask : numpy.ndarray of shape (T)
        A temporal mask in which ones indicate volumes to be flagged and interpolated across.
    TR : float, optional
        The repetition time of the BOLD data, in seconds. Default is 1.

    Returns
    -------
    bold_data_interpolated : numpy.ndarray of shape (S, T)
        The interpolated BOLD data.
    """
    # import interpolate functionality
    from scipy.interpolate import interp1d

    # Confirm that interpolation can be correctly performed
    bold_data_interpolated = bold_data
    if np.mean(tmask) == 0:
        print('No flagged volume, interpolation will not be done.')
    elif np.mean(tmask) > 0.5:
        print('More than 50% of volumes are flagged, interpolation will not be done.')
    else:
        # Create slice time array
        slice_times = TR * np.arange(0, (bold_data.shape[1]), 1)
        # then add all the times which were not scrubbed. Append the last
        # time to the end
        slice_times_extended = np.append(slice_times[tmask == 0], slice_times[-1])
        # Stack bold data: all timepoints not scrubbed are appended to
        # the last timepoint
        clean_volume = np.hstack(
            (bold_data[:,
                       (tmask == 0)], np.reshape(bold_data[:, -1],
                                                 [bold_data.shape[0], 1])))

        # looping through each voxel
        for voxel in range(0, bold_data.shape[0]):
            # create interpolation function
            interpolation_function = interp1d(slice_times_extended, clean_volume[voxel, :])
            # create data
            interpolated_data = interpolation_function(slice_times)
            # if the data was scrubbed, replace it with the interpolated data
            bold_data_interpolated[voxel, (tmask == 1)] = interpolated_data[tmask == 1]

    return bold_data_interpolated
