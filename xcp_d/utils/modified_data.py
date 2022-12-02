# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for interpolating over high-motion volumes."""
import os

import nibabel as nb
import numpy as np
from nipype import logging

from xcp_d.utils.doc import fill_doc
from xcp_d.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger("nipype.utils")


@fill_doc
def compute_fd(confound, head_radius=50):
    """Compute framewise displacement.

    NOTE: TS- Which kind of FD? Power?
    NOTE: TS- What units must rotation parameters be in?

    Parameters
    ----------
    confound : pandas.DataFrame
        DataFrame with six motion parameters.
    %(head_radius)s

    Returns
    -------
    fdres : numpy.ndarray
        The framewise displacement time series.
    """
    confound = confound.replace(np.nan, 0)
    mpars = confound[["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]].to_numpy()
    diff = mpars[:-1, :6] - mpars[1:, :6]
    diff[:, 3:6] *= head_radius
    fd_res = np.abs(diff).sum(axis=1)
    fdres = np.hstack([0, fd_res])

    return fdres


@fill_doc
def generate_mask(fd_res, fd_thresh):
    """Create binary temporal mask flagging high-motion volumes.

    Parameters
    ----------
    fd_res : numpy.ndarray of shape (T)
        Framewise displacement time series.
        T = time.
    %(fd_thresh)s

    Returns
    -------
    tmask : numpy.ndarray of shape (T)
        The temporal mask. Zeros are low-motion volumes. Ones are high-motion volumes.
    """
    tmask = np.zeros(len(fd_res), dtype=int)
    tmask[fd_res > fd_thresh] = 1

    return tmask


def interpolate_masked_data(bold_data, tmask, TR=1):
    """Interpolate masked data.

    No interpolation will be performed if more than 50% of the volumes in the BOLD data are
    flagged by the temporal mask.

    NOTE: TS- Why are slice times being inferred from the number of volumes?
    Am I missing something?

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
        print("No flagged volume, interpolation will not be done.")
    elif np.mean(tmask) > 0.5:
        print("More than 50% of volumes are flagged, interpolation will not be done.")
    else:
        # Create slice time array
        slice_times = TR * np.arange(0, (bold_data.shape[1]), 1)
        # then add all the times which were not scrubbed. Append the last
        # time to the end
        slice_times_extended = np.append(slice_times[tmask == 0], slice_times[-1])
        # Stack bold data: all timepoints not scrubbed are appended to
        # the last timepoint
        clean_volume = np.hstack(
            (bold_data[:, (tmask == 0)], np.reshape(bold_data[:, -1], [bold_data.shape[0], 1]))
        )

        # looping through each voxel
        for voxel in range(0, bold_data.shape[0]):
            # create interpolation function
            interpolation_function = interp1d(slice_times_extended, clean_volume[voxel, :])
            # create data
            interpolated_data = interpolation_function(slice_times)
            # if the data was scrubbed, replace it with the interpolated data
            bold_data_interpolated[voxel, (tmask == 1)] = interpolated_data[tmask == 1]

    return bold_data_interpolated


def _drop_dummy_scans(bold_file, dummy_scans):
    """Remove the first X volumes from a BOLD file.

    Parameters
    ----------
    bold_file : str
        Path to a nifti or cifti file.
    dummy_scans : int
        If an integer, the first ``dummy_scans`` volumes will be removed.

    Returns
    -------
    dropped_image : img_like
        The BOLD image, with the first X volumes removed.
    """
    # read the bold file
    bold_image = nb.load(bold_file)
    ndim = bold_image.ndim

    if ndim == 2:  # cifti
        data = bold_image.get_fdata()
        data = data[dummy_scans:, ...]  # time series is the first element
        time_axis, brain_model_axis = [
            bold_image.header.get_axis(i) for i in range(bold_image.ndim)
        ]
        new_total_volumes = data.shape[0]
        dropped_time_axis = time_axis[:new_total_volumes]
        dropped_header = nb.cifti2.Cifti2Header.from_axes((dropped_time_axis, brain_model_axis))
        bold_image = nb.Cifti2Image(
            data,
            header=dropped_header,
            nifti_header=bold_image.nifti_header,
        )

    elif ndim == 4:  # nifti
        data = bold_image.get_fdata()
        data = data[..., dummy_scans:]  # time is fourth dim
        bold_image = nb.Nifti1Image(data, bold_image.affine, bold_image.header)

    else:
        raise ValueError(f"Image dimensionality ({ndim}) not supported for {bold_file}")

    return bold_image


def downcast_to_32(in_file):
    """Downcast a file from >32-bit to 32-bit if necessary.

    Parameters
    ----------
    in_file : None or str
        Path to a file to downcast.
        If None, None will be returned.
        If the file is lower-precision than 32-bit,
        then it will be returned without modification.

    Returns
    -------
    None or str
        Path to the downcast file.
        If in_file is None, None will be returned.
        If in_file is a file with lower than 32-bit precision,
        then it will be returned without modification.
        Otherwise, a new path will be returned.
    """
    if in_file is None:
        return in_file

    elif not os.path.isfile(in_file):
        raise FileNotFoundError(f"File not found: {in_file}")

    img = nb.load(in_file)
    if hasattr(img, "nifti_header"):
        header = img.nifti_header
    else:
        header = img.header

    SIZE32 = 4  # number of bytes in float32/int32
    dtype = header.get_data_dtype()
    if dtype.itemsize > SIZE32:
        LOGGER.warning(f"Downcasting {in_file} to 32-bit.")
        if np.issubdtype(dtype, np.integer):
            header.set_data_dtype(np.int32)
        elif np.issubdtype(dtype, np.floating):
            header.set_data_dtype(np.float32)
        else:
            raise TypeError(f"Unknown datatype '{dtype}'.")

        out_file = fname_presuffix(in_file, newpath=os.getcwd(), suffix="_downcast", use_ext=True)
        img.to_filename(out_file)
    else:
        out_file = in_file

    return out_file
