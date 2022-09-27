"""Quality control metrics."""
import nibabel as nb
import numpy as np


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
        'coregDice': [dc(bold2t1w_mask, t1w_mask)],
        'coregJaccard': [jc(bold2t1w_mask, t1w_mask)],
        'coregCrossCorr': [crosscorr(bold2t1w_mask, t1w_mask)],
        'coregCoverag': [coverage(bold2t1w_mask, t1w_mask)],
        'normDice': [dc(bold2template_mask, template_mask)],
        'normJaccard': [jc(bold2template_mask, template_mask)],
        'normCrossCorr': [crosscorr(bold2template_mask, template_mask)],
        'normCoverage': [coverage(bold2template_mask, template_mask)],
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
        dc = 2. * intersection / float(size_i1 + size_i2)
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
        The data matrix fromw hich to calculate DVARS. #TODO: Fix typo
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
