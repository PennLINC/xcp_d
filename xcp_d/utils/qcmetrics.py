"""Quality control metrics."""

import nibabel as nb
import numpy as np
from nipype import logging

LOGGER = logging.getLogger("nipype.utils")


def compute_registration_qc(
    bold_mask_anatspace,
    anat_mask_anatspace,
    bold_mask_stdspace,
    template_mask,
):
    """Compute quality of registration metrics.

    This function will calculate a series of metrics, including:

    - Dice's similarity index,
    - Pearson correlation coefficient, and
    - Coverage

    between the BOLD-to-T1w brain mask and the T1w mask,
    as well as between the BOLD-to-template brain mask and the template mask.

    Parameters
    ----------
    bold_mask_anatspace : :obj:`str`
        Path to the BOLD brain mask in anatomical (T1w or T2w) space.
    anat_mask_anatspace : :obj:`str`
        Path to the anatomically-derived brain mask in anatomical space.
    bold_mask_stdspace : :obj:`str`
        Path to the BOLD brain mask in template space.
    template_mask : :obj:`str`
        Path to the template's official brain mask.

    Returns
    -------
    reg_qc : dict
        Quality control measures between different inputs.
    qc_metadata : dict
        Metadata describing the QC measures.
    """
    bold_mask_anatspace_arr = nb.load(bold_mask_anatspace).get_fdata()
    anat_mask_anatspace_arr = nb.load(anat_mask_anatspace).get_fdata()
    bold_mask_stdspace_arr = nb.load(bold_mask_stdspace).get_fdata()
    template_mask_arr = nb.load(template_mask).get_fdata()

    reg_qc = {
        "coreg_dice": [dice(bold_mask_anatspace_arr, anat_mask_anatspace_arr)],
        "coreg_correlation": [pearson(bold_mask_anatspace_arr, anat_mask_anatspace_arr)],
        "coreg_overlap": [overlap(bold_mask_anatspace_arr, anat_mask_anatspace_arr)],
        "norm_dice": [dice(bold_mask_stdspace_arr, template_mask_arr)],
        "norm_correlation": [pearson(bold_mask_stdspace_arr, template_mask_arr)],
        "norm_overlap": [overlap(bold_mask_stdspace_arr, template_mask_arr)],
    }
    qc_metadata = {
        "coreg_dice": {
            "LongName": "Coregistration Sørensen-Dice Coefficient",
            "Description": (
                "The Sørensen-Dice coefficient calculated between the binary brain masks from the "
                "coregistered anatomical and functional images. "
                "Values are bounded between 0 and 1, "
                "with higher values indicating better coregistration."
            ),
            "Term URL": "https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient",
        },
        "coreg_correlation": {
            "LongName": "Coregistration Pearson Correlation",
            "Description": (
                "The Pearson correlation coefficient calculated between the binary brain masks "
                "from the coregistered anatomical and functional images. "
                "Values are bounded between 0 and 1, "
                "with higher values indicating better coregistration."
            ),
            "Term URL": "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",
        },
        "coreg_overlap": {
            "LongName": "Coregistration Coverage Metric",
            "Description": (
                "The Szymkiewicz-Simpson overlap coefficient calculated between the binary brain "
                "masks from the normalized functional image and the associated template. "
                "Higher values indicate better normalization."
            ),
            "Term URL": "https://en.wikipedia.org/wiki/Overlap_coefficient",
        },
        "norm_dice": {
            "LongName": "Normalization Sørensen-Dice Coefficient",
            "Description": (
                "The Sørensen-Dice coefficient calculated between the binary brain masks from the "
                "normalized functional image and the associated template. "
                "Values are bounded between 0 and 1, "
                "with higher values indicating better normalization."
            ),
            "Term URL": "https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient",
        },
        "norm_correlation": {
            "LongName": "Normalization Pearson Correlation",
            "Description": (
                "The Pearson correlation coefficient calculated between the binary brain masks "
                "from the normalized functional image and the associated template. "
                "Values are bounded between 0 and 1, "
                "with higher values indicating better normalization."
            ),
            "Term URL": "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient",
        },
        "norm_overlap": {
            "LongName": "Normalization Overlap Coefficient",
            "Description": (
                "The Szymkiewicz-Simpson overlap coefficient calculated between the binary brain "
                "masks from the normalized functional image and the associated template. "
                "Higher values indicate better normalization."
            ),
            "Term URL": "https://en.wikipedia.org/wiki/Overlap_coefficient",
        },
    }
    return reg_qc, qc_metadata


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
    coef : :obj:`float`
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

    if (size_i1 + size_i2) == 0:
        coef = 0
    else:
        coef = (2 * intersection) / (size_i1 + size_i2)

    return coef


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
    coef : :obj:`float`
        Correlation between the two images.
    """
    input1 = np.atleast_1d(input1.astype(bool)).flatten()
    input2 = np.atleast_1d(input2.astype(bool)).flatten()

    return np.corrcoef(input1, input2)[0, 1]


def overlap(input1, input2):
    r"""Calculate overlap coefficient between two images.

    The metric is defined as

    .. math::

        DC=\frac{|A \cap B||}{min(|A|,|B|)}

    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).

    The overlap coefficient is also known as the Szymkiewicz-Simpson coefficient
    :footcite:p:`vijaymeena2016survey`.

    Parameters
    ----------
    input1/input2 : :obj:`numpy.ndarray`
        Numpy arrays to compare.
        Can be any type but will be converted into binary:
        False where 0, True everywhere else.

    Returns
    -------
    coef : :obj:`float`
        Coverage between two images.

    References
    ----------
    .. footbibliography::
    """
    input1 = np.atleast_1d(input1.astype(bool))
    input2 = np.atleast_1d(input2.astype(bool))

    intersection = np.count_nonzero(input1 & input2)
    smallv = np.minimum(np.sum(input1), np.sum(input2))

    return intersection / smallv


def compute_dvars(
    *,
    datat,
    remove_zerovariance=True,
    variance_tol=1e-7,
):
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
    :obj:`numpy.ndarray`
        The calculated standardized DVARS array.
        A (timepoints,) array.
    """
    from nipype.algorithms.confounds import _AR_est_YW, regress_poly

    # Robust standard deviation (we are using "lower" interpolation because this is what FSL does
    try:
        func_sd = (
            np.percentile(datat, 75, axis=1, method="lower")
            - np.percentile(datat, 25, axis=1, method="lower")
        ) / 1.349
    except TypeError:  # NP < 1.22
        func_sd = (
            np.percentile(datat, 75, axis=1, interpolation="lower")
            - np.percentile(datat, 25, axis=1, interpolation="lower")
        ) / 1.349

    if remove_zerovariance:
        zero_variance_voxels = func_sd > variance_tol
        datat = datat[zero_variance_voxels, :]
        func_sd = func_sd[zero_variance_voxels]

    # Compute (non-robust) estimate of lag-1 autocorrelation
    temp_data = regress_poly(0, datat, remove_mean=True)[0].astype(np.float32)

    ar1 = np.apply_along_axis(_AR_est_YW, 1, temp_data, 1)

    # Compute (predicted) standard deviation of temporal difference time series
    diff_sdhat = np.squeeze(np.sqrt(((1 - ar1) * 2).tolist())) * func_sd
    diff_sd_mean = diff_sdhat.mean()

    # Compute temporal difference time series
    func_diff = np.diff(datat, axis=1)

    # DVARS (no standardization)
    dvars_nstd = np.sqrt(np.square(func_diff).mean(axis=0))

    # standardization
    dvars_stdz = dvars_nstd / diff_sd_mean

    # Insert 0 at the beginning (fMRIPrep would add a NaN here)
    dvars_nstd = np.insert(dvars_nstd, 0, 0)
    dvars_stdz = np.insert(dvars_stdz, 0, 0)

    return dvars_nstd, dvars_stdz
