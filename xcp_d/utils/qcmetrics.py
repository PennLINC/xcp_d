"""Quality control metrics."""
import h5py
import nibabel as nb
import numpy as np
import pandas as pd
from nipype import logging

from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.utils")


def compute_registration_qc(bold2t1w_mask, anat_brainmask, bold2template_mask, template_mask):
    """Compute quality of registration metrics.

    This function will calculate a series of metrics, including:

    - Dice's similarity index,
    - Pearson correlation coefficient, and
    - Coverage

    between the BOLD-to-T1w brain mask and the T1w mask,
    as well as between the BOLD-to-template brain mask and the template mask.

    Parameters
    ----------
    bold2t1w_mask : :obj:`str`
        Path to the BOLD mask in T1w space.
    anat_brainmask : :obj:`str`
        Path to the T1w mask.
    bold2template_mask : :obj:`str`
        Path to the BOLD mask in template space.
    template_mask : :obj:`str`
        Path to the template mask.

    Returns
    -------
    reg_qc : dict
        Quality control measures between different inputs.
    """
    bold2t1w_mask_arr = nb.load(bold2t1w_mask).get_fdata()
    t1w_mask_arr = nb.load(anat_brainmask).get_fdata()
    bold2template_mask_arr = nb.load(bold2template_mask).get_fdata()
    template_mask_arr = nb.load(template_mask).get_fdata()

    reg_qc = {
        "coregDice": [dice(bold2t1w_mask_arr, t1w_mask_arr)],
        "coregPearson": [pearson(bold2t1w_mask_arr, t1w_mask_arr)],
        "coregCoverage": [coverage(bold2t1w_mask_arr, t1w_mask_arr)],
        "normDice": [dice(bold2template_mask_arr, template_mask_arr)],
        "normPearson": [pearson(bold2template_mask_arr, template_mask_arr)],
        "normCoverage": [coverage(bold2template_mask_arr, template_mask_arr)],
    }
    return reg_qc


def dice(input1, input2):
    r"""Calculate Dice coefficient between two arrays.

    Computes the Dice coefficient (also known as Sorensen index) between two binary images.

    The metric is defined as

    .. math::

        DC=\frac{2|A\cap B|}{|A|+|B|}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    This method was first proposed in :footcite:t:`dice1945measures` and
    :footcite:t:`sorensen1948method`.

    Parameters
    ----------
    input1/input2 : :obj:`numpy.ndarray`
        Numpy arrays to compare.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    dice : :obj:`float`
        The Dice coefficient between ``input1`` and ``input2``.
        It ranges from 0 (no overlap) to 1 (perfect overlap).

    References
    ----------
    .. footbibliography::
    """
    input1 = np.atleast_1d(input1.astype(bool))
    input2 = np.atleast_1d(input2.astype(bool))

    intersection = np.count_nonzero(input1 & input2)

    size_i1 = np.count_nonzero(input1)
    size_i2 = np.count_nonzero(input2)

    try:
        dsi = (2 * intersection) / (size_i1 + size_i2)
    except ZeroDivisionError:
        dsi = 0

    return dsi


def pearson(input1, input2):
    """Calculate Pearson product moment correlation between two images.

    Parameters
    ----------
    input1/input2 : :obj:`numpy.ndarray`
        Numpy arrays to compare.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    corr : :obj:`float`
        Correlation between the two images.
    """
    input1 = np.atleast_1d(input1.astype(bool)).flatten()
    input2 = np.atleast_1d(input2.astype(bool)).flatten()

    corr = np.corrcoef(input1, input2)[0][1]

    return corr


def coverage(input1, input2):
    """Estimate the coverage between two masks.

    Parameters
    ----------
    input1/input2 : :obj:`numpy.ndarray`
        Numpy arrays to compare.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    cov : :obj:`float`
        Coverage between two images.
    """
    input1 = np.atleast_1d(input1.astype(bool))
    input2 = np.atleast_1d(input2.astype(bool))

    intersection = np.count_nonzero(input1 & input2)

    smallv = np.minimum(np.sum(input1), np.sum(input2))

    cov = intersection / smallv

    return cov


def compute_dvars(datat):
    """Compute standard DVARS.

    Parameters
    ----------
    datat : :obj:`numpy.ndarray`
        The data matrix from which to calculate DVARS.
        Ordered as vertices by timepoints.

    Returns
    -------
    :obj:`numpy.ndarray`
        The calculated DVARS array.
        A (timepoints,) array.
    """
    firstcolumn = np.zeros((datat.shape[0]))[..., None]
    datax = np.hstack((firstcolumn, np.diff(datat)))
    datax_ss = np.sum(np.square(datax), axis=0) / datat.shape[0]
    return np.sqrt(datax_ss)


def _make_dcan_qc_file(filtered_motion, TR):
    """Make DCAN HDF5 file from single motion file.

    NOTE: This is a Node function.

    Parameters
    ----------
    filtered_motion_file : :obj:`str`
        File from which to extract information.
    TR : :obj:`float`
        Repetition time.

    Returns
    -------
    dcan_df_file : :obj:`str`
        Name of the HDF5-format file that is created.
    """
    import os

    from xcp_d.utils.qcmetrics import make_dcan_df

    dcan_df_file = os.path.abspath("desc-dcan_qc.hdf5")

    make_dcan_df(filtered_motion, dcan_df_file, TR)
    return dcan_df_file


@fill_doc
def make_dcan_df(filtered_motion, name, TR):
    """Create an HDF5-format file containing a DCAN-format dataset.

    Parameters
    ----------
    %(filtered_motion)s
    name : :obj:`str`
        Name of the HDF5-format file to be created.
    %(TR)s

    Notes
    -----
    The metrics in the file are:

    -   ``FD_threshold``: a number >= 0 that represents the FD threshold used to calculate
        the metrics in this list.
    -   ``frame_removal``: a binary vector/array the same length as the number of frames
        in the concatenated time series, indicates whether a frame is removed (1) or not (0)
    -   ``format_string`` (legacy): a string that denotes how the frames were excluded.
        This uses a notation devised by Avi Snyder.
    -   ``total_frame_count``: a whole number that represents the total number of frames
        in the concatenated series
    -   ``remaining_frame_count``: a whole number that represents the number of remaining
        frames in the concatenated series
    -   ``remaining_seconds``: a whole number that represents the amount of time remaining
        after thresholding
    -   ``remaining_frame_mean_FD``: a number >= 0 that represents the mean FD of the
        remaining frames
    """
    LOGGER.debug(f"Generating DCAN file: {name}")

    # Load filtered framewise_displacement values from file
    filtered_motion_df = pd.read_table(filtered_motion)
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
