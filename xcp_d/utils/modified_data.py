# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
""" compute FD, genetrate mask  """

import numpy as np


def compute_FD(confound, head_radius=50):
    """
    computes FD
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


def generate_mask(
    fd_res,
    fd_thresh,
):

    tmask = np.zeros(len(fd_res))
    tmask[fd_res > fd_thresh] = 1

    return tmask


def interpolate_masked_data(bold_data, tmask, TR=1):
    # import interpolate functionality
    from scipy.interpolate import interp1d
    # Confirm that interpolation can be correctly performed
    bold_data_interpolated = bold_data
    if np.mean(tmask) == 0:
        print('No flagged volume, interpolation will not be done .')
    elif np.mean(tmask) > 0.5:
        print(
            'More than 50% of volumes are flagged, interpolation will not be done '
        )
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
