"""Quality control metrics."""
import h5py
import nibabel as nb
import numpy as np
import pandas as pd
from nipype import logging

LOGGER = logging.getLogger("nipype.utils")


def compute_registration_qc(bold2t1w_mask, t1w_mask, bold2template_mask, template_mask):
    """Compute quality of registration metrics.

    This function will calculate a series of metrics, including Dice's similarity index,
    Jaccard's coefficient, cross-correlation, and coverage, between the BOLD-to-T1w brain mask
    and the T1w mask, as well as between the BOLD-to-template brain mask and the template mask.

    Parameters
    ----------
    bold2t1w_mask : str
        Path to the BOLD mask in T1w space.
    t1w_mask : str
        Path to the T1w mask.
    bold2template_mask : str
        Path to the BOLD mask in template space.
    template_mask : str
        Path to the template mask.

    Returns
    -------
    reg_qc : dict
        Quality control measures between different inputs.
    """
    reg_qc = {
        "coregDice": [dc(bold2t1w_mask, t1w_mask)],
        "coregJaccard": [jc(bold2t1w_mask, t1w_mask)],
        "coregCrossCorr": [crosscorr(bold2t1w_mask, t1w_mask)],
        "coregCoverag": [coverage(bold2t1w_mask, t1w_mask)],
        "normDice": [dc(bold2template_mask, template_mask)],
        "normJaccard": [jc(bold2template_mask, template_mask)],
        "normCrossCorr": [crosscorr(bold2template_mask, template_mask)],
        "normCoverage": [coverage(bold2template_mask, template_mask)],
    }
    return reg_qc


def dc(input1, input2):
    r"""Calculate Dice coefficient between two images.

    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in twom j images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    Parameters
    ----------
    input1/input2 : str
        Path to a NIFTI image.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```input1``` and the
        object(s) in ```input2```. It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric.
    """
    input1 = nb.load(input1).get_fdata()
    input2 = nb.load(input2).get_fdata()
    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))

    intersection = np.count_nonzero(input1 & input2)

    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)

    try:
        dc = 2.0 * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def jc(input1, input2):
    r"""Calculate Jaccard coefficient between two images.

    Computes the Jaccard coefficient between the binary objects in two images.

    Parameters
    ----------
    input1/input2 : str
        Path to a NIFTI image.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    jc : float
        The Jaccard coefficient between the object(s) in ``input1`` and the
        object(s) in ``input2``.
        It ranges from 0 (no overlap) to 1 (perfect overlap).

    Notes
    -----
    This is a real metric.
    """
    input1 = nb.load(input1).get_fdata()
    input2 = nb.load(input2).get_fdata()
    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))

    intersection = np.count_nonzero(input1 & input2)
    union = np.count_nonzero(input1 | input2)

    jc = float(intersection) / float(union)

    return jc


def crosscorr(input1, input2):
    """Calculate cross correlation between two images.

    NOTE: TS- This appears to be Pearson's correlation, not cross-correlation.

    Parameters
    ----------
    input1/input2 : str
        Path to a NIFTI image.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    cc : float
        Correlation between the two images.
    """
    input1 = nb.load(input1).get_fdata()
    input2 = nb.load(input2).get_fdata()
    input1 = np.atleast_1d(input1.astype(np.bool)).flatten()
    input2 = np.atleast_1d(input2.astype(np.bool)).flatten()
    cc = np.corrcoef(input1, input2)[0][1]
    return cc


def coverage(input1, input2):
    """Estimate the coverage between two masks.

    Parameters
    ----------
    input1/input2 : str
        Path to a NIFTI image.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    cov : float
        Coverage between two images.
    """
    input1 = nb.load(input1).get_fdata()
    input2 = nb.load(input2).get_fdata()
    input1 = np.atleast_1d(input1.astype(np.bool))
    input2 = np.atleast_1d(input2.astype(np.bool))
    intsec = np.count_nonzero(input1 & input2)
    if np.sum(input1) > np.sum(input2):
        smallv = np.sum(input2)
    else:
        smallv = np.sum(input1)
    cov = float(intsec) / float(smallv)
    return cov


def compute_dvars(datat):
    """Compute standard DVARS.

    Parameters
    ----------
    datat : numpy.ndarray
        The data matrix from which to calculate DVARS.
        Ordered as vertices by timepoints.

    Returns
    -------
    numpy.ndarray
        The calculated DVARS array.
        A (timepoints,) array.
    """
    firstcolumn = np.zeros((datat.shape[0]))[..., None]
    datax = np.hstack((firstcolumn, np.diff(datat)))
    datax_ss = np.sum(np.square(datax), axis=0) / datat.shape[0]
    return np.sqrt(datax_ss)


def _make_dcan_qc_file(filtered_motion, TR):
    """Make DCAN HDF5 file from single motion file."""
    import os

    from xcp_d.utils.qcmetrics import make_dcan_df

    dcan_df_file = os.path.abspath("desc-dcan_qc.hdf5")

    make_dcan_df(filtered_motion, dcan_df_file, TR)
    return dcan_df_file


def make_dcan_df(filtered_motion_file, name, TR):
    """Create an HDF5-format file containing a DCAN-format dataset.

    Parameters
    ----------
    filtered_motion_file : str
        File from which to extract information.
    name : str
        Name of the HDF5-format file to be created.
    TR : float
        Repetition time.

    Notes
    -----
    FD_threshold: a number >= 0 that represents the FD threshold used to calculate
    the metrics in this list.
    frame_removal: a binary vector/array the same length as the number of frames
    in the concatenated time series, indicates whether a frame is removed (1) or
    not (0)
    format_string (legacy): a string that denotes how the frames were excluded
    -- uses a notation devised by Avi Snyder
    total_frame_count: a whole number that represents the total number of frames
    in the concatenated series
    remaining_frame_count: a whole number that represents the number of remaining
    frames in the concatenated series
    remaining_seconds: a whole number that represents the amount of time remaining
    after thresholding
    remaining_frame_mean_FD: a number >= 0 that represents the mean FD of the
    remaining frames
    """
    LOGGER.debug(f"Generating DCAN file: {name}")

    # Load filtered framewise_displacement values from file
    filtered_motion_df = pd.read_table(filtered_motion_file)
    fd = filtered_motion_df["framewise_displacement"].values

    with h5py.File(name, "w") as dcan:
        for thresh in np.linspace(0, 1, 101):
            thresh = np.around(thresh, 2)

            dcan.create_dataset(f"/dcan_motion/fd_{thresh}/skip", data=0, dtype="float")
            dcan.create_dataset(
                f"/dcan_motion/fd_{thresh}/binary_mask",
                data=(fd > thresh).astype(int),
                dtype="float",
            )
            dcan.create_dataset(f"/dcan_motion/fd_{thresh}/threshold", data=thresh, dtype="float")
            dcan.create_dataset(
                f"/dcan_motion/fd_{thresh}/total_frame_count", data=len(fd), dtype="float"
            )
            dcan.create_dataset(
                f"/dcan_motion/fd_{thresh}/remaining_total_frame_count",
                data=len(fd[fd <= thresh]),
                dtype="float",
            )
            dcan.create_dataset(
                f"/dcan_motion/fd_{thresh}/remaining_seconds",
                data=len(fd[fd <= thresh]) * TR,
                dtype="float",
            )
            dcan.create_dataset(
                f"/dcan_motion/fd_{thresh}/remaining_frame_mean_FD",
                data=(fd[fd <= thresh]).mean(),
                dtype="float",
            )
