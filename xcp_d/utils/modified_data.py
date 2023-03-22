# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for interpolating over high-motion volumes."""
import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype import logging

from xcp_d.utils.confounds import _infer_dummy_scans, load_motion
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


def _drop_dummy_scans(bold_file, dummy_scans):
    """Remove the first X volumes from a BOLD file.

    Parameters
    ----------
    bold_file : :obj:`str`
        Path to a nifti or cifti file.
    dummy_scans : :obj:`int`
        If an integer, the first ``dummy_scans`` volumes will be removed.

    Returns
    -------
    dropped_image : img_like
        The BOLD image, with the first X volumes removed.
    """
    # read the bold file
    bold_image = nb.load(bold_file)

    data = bold_image.get_fdata()

    if bold_image.ndim == 2:  # cifti
        dropped_data = data[dummy_scans:, ...]  # time series is the first element
        time_axis, brain_model_axis = [
            bold_image.header.get_axis(i) for i in range(bold_image.ndim)
        ]
        new_total_volumes = dropped_data.shape[0]
        dropped_time_axis = time_axis[:new_total_volumes]
        dropped_header = nb.cifti2.Cifti2Header.from_axes((dropped_time_axis, brain_model_axis))
        dropped_image = nb.Cifti2Image(
            dropped_data, header=dropped_header, nifti_header=bold_image.nifti_header
        )

    else:  # nifti
        dropped_data = data[..., dummy_scans:]
        dropped_image = nb.Nifti1Image(
            dropped_data, affine=bold_image.affine, header=bold_image.header
        )

    return dropped_image


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


def cast_cifti_to_int16(in_file):
    """Cast a CIFTI file to int16 data.

    This function serves as a temporary workaround for a bug in the
    DerivativesDataSink class from niworkflows version 1.7.1.
    For more information, see https://github.com/nipreps/niworkflows/issues/778.

    NOTE: This is a Node function.

    Parameters
    ----------
    in_file : :obj:`str`
        Path to input CIFTI file.

    Returns
    -------
    out_file : :obj:`str`
        Path to output CIFTI file.
        The file will have the same filename, but will be written to the working directory.
    """
    import os

    import nibabel as nb
    import numpy as np

    filename = os.path.basename(in_file)
    out_file = os.path.abspath(filename)
    if os.path.abspath(in_file) == out_file:
        raise ValueError("This function must be run in a separate working directory!")

    img = nb.load(in_file)
    img.nifti_header.set_data_dtype(np.int16)
    img.to_filename(out_file)

    return out_file


def scale_to_min_max(X, x_min, x_max):
    """Scale data to between minimum and maximum values."""
    nom = (X - X.min()) * (x_max - x_min)
    denom = X.max() - X.min()

    if denom == 0:
        denom = 1

    return x_min + (nom / denom)


@fill_doc
def flag_bad_run(
    fmriprep_confounds_file,
    dummy_scans,
    TR,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    head_radius,
    fd_thresh,
):
    """Determine if a run has too many high-motion volumes to continue processing.

    Parameters
    ----------
    %(fmriprep_confounds_file)s
    %(dummy_scans)s
    %(TR)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(head_radius)s
    %(fd_thresh)s
    brain_mask

    Returns
    -------
    post_scrubbing_duration : :obj:`float`
        Amount of time remaining in the run after dummy scan removal, in seconds.
    """
    if fd_thresh <= 0:
        # No scrubbing will be performed, so there's no point is calculating amount of "good time".
        return np.inf

    dummy_scans = _infer_dummy_scans(
        dummy_scans=dummy_scans,
        confounds_file=fmriprep_confounds_file,
    )

    # Read in fmriprep confounds tsv to calculate FD
    fmriprep_confounds_df = pd.read_table(fmriprep_confounds_file)

    # Remove dummy volumes
    fmriprep_confounds_df = fmriprep_confounds_df.drop(np.arange(dummy_scans))

    # Calculate filtered FD
    motion_df = load_motion(
        fmriprep_confounds_df,
        TR=TR,
        motion_filter_type=motion_filter_type,
        motion_filter_order=motion_filter_order,
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
    )
    fd_arr = compute_fd(confound=motion_df, head_radius=head_radius)
    return np.sum(fd_arr <= fd_thresh) * TR
