import nibabel as nb
import numpy as np

def regisQ(bold2t1w_mask,t1w_mask,bold2template_mask,template_mask):
	reg_qc ={'coregDice': [dc(bold2t1w_mask,t1w_mask)], 'coregJaccard': [jc(bold2t1w_mask,t1w_mask)],
              'coregCrossCorr': [crosscorr(bold2t1w_mask,t1w_mask)],'coregCoverag': [coverage(bold2t1w_mask,t1w_mask)],
	      'normDice': [dc(bold2template_mask,template_mask)],'normJaccard': [jc(bold2template_mask,template_mask)], 
	      'normCrossCorr': [crosscorr(bold2template_mask,template_mask)], 'normCoverage': [coverage(bold2template_mask,template_mask)],
	      }
	return reg_qc



def dc(input1, input2):
    r"""
    Dice coefficient
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in twom j images.
    The metric is defined as
    .. math::
        DC=\frac{2|A\cap B|}{|A|+|B|}
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    Parameters
    ----------
    input1 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
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
    r"""
    Jaccard coefficient
    Computes the Jaccard coefficient between the binary objects in two images.
    Parameters
    ----------
    input1: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    input2: array_like
            Input data containing objects. Can be any type but will be converted
            into binary: background where 0, object everywhere else.
    Returns
    -------
    jc: float
        The Jaccard coefficient between the object(s) in `input1` and the
        object(s) in `input2`. It ranges from 0 (no overlap) to 1 (perfect overlap).
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
    r"""
    cross correlation
    computer compute cross correction bewteen input mask
    """
    input1 = nb.load(input1).get_fdata()
    input2 = nb.load(input2).get_fdata()
    input1 = np.atleast_1d(input1.astype(np.bool)).flatten()
    input2 = np.atleast_1d(input2.astype(np.bool)).flatten()
    cc = np.corrcoef(input1, input2)[0][1]
    return cc


def coverage(input1, input2):
    """
    estimate the coverage between  two mask
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
    cov = float(intsec)/float(smallv)
    return cov