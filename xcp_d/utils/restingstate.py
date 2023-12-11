# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for calculating resting-state derivatives (ReHo and ALFF)."""
import nibabel as nb
import numpy as np
from nipype import logging
from scipy import signal
from scipy.stats import rankdata
from templateflow.api import get as get_template

LOGGER = logging.getLogger("nipype.utils")


def compute_2d_reho(datat, adjacency_matrix):
    """Calculate ReHo on 2D data.

    Parameters
    ----------
    datat : numpy.ndarray of shape (V, T)
        data matrix in vertices by timepoints
    adjacency_matrix : numpy.ndarray of shape (V, V)
        surface adjacency matrix

    Returns
    -------
    KCC : numpy.ndarray of shape (V,)
        ReHo values.

    Notes
    -----
    From https://www.sciencedirect.com/science/article/pii/S0165178119305384#bib0045.
    """
    KCC = np.zeros(datat.shape[0])  # a zero for each voxel

    for i in range(datat.shape[0]):  # loop through each voxel
        neigbor_index = np.where(adjacency_matrix[i, :] > 0)[0]  # the index of 4 neightbouts
        nn = np.hstack((neigbor_index, np.array(i)))  # stack those indexes with voxel number
        neidata = datat[nn, :]  # pull out data for relevant voxels

        rankeddata = np.zeros_like(neidata)  # TODO: Fix typos #create 0s in same shape
        # pull out index of voxel, timepoint
        neigbor, timepoint = neidata.shape[0], neidata.shape[1]

        for j in range(neidata.shape[0]):  # loop through each neighbour
            rankeddata[j, :] = rankdata(neidata[j, :])  # assign ranks to timepoints for each voxel
        rankmean = np.sum(rankeddata, axis=0)  # add up ranks
        # KC is the sum of the squared rankmean minus the timepoints into
        # the mean of the rankmean squared
        KC = np.sum(np.power(rankmean, 2)) - timepoint * np.power(np.mean(rankmean), 2)
        # square number of neighbours, multiply by (cubed timepoint - timepoint)
        denom = np.power(neigbor, 2) * (np.power(timepoint, 3) - timepoint)
        # the voxel value is 12*KC divided by denom
        KCC[i] = 12 * KC / (denom)

    return KCC


def mesh_adjacency(hemi):
    """Calculate adjacency matrix from mesh timeseries.

    Parameters
    ----------
    hemi : {"L", "R"}
        Surface sphere to be load from templateflow
        Either left or right hemisphere

    Returns
    -------
    numpy.ndarray
        Adjacency matrix.
    """
    surf = str(
        get_template("fsLR", space="fsaverage", hemi=hemi, suffix="sphere", density="32k")
    )  # Get relevant template

    surf = nb.load(surf)  # load via nibabel
    #  Aggregate GIFTI data arrays into an ndarray or tuple of ndarray
    # select the arrays in a specific order
    vertices_faces = surf.agg_data(("pointset", "triangle"))
    vertices = vertices_faces[0]  # the first array of the tuple
    faces = vertices_faces[1]  # the second array in the tuples
    # create an array of 0s = voxel*voxel
    data_array = np.zeros([len(vertices), len(vertices)], dtype=np.uint8)

    for i in range(1, len(faces)):  # looping thorugh each value in faces
        data_array[faces[i, 0], faces[i, 2]] = 1  # use to index into data_array and
        # turn select values to 1
        data_array[faces[i, 1], faces[i, 1]] = 1
        data_array[faces[i, 2], faces[i, 0]] = 1

    return data_array + data_array.T  # transpose data_array and add it to itself


def compute_alff(data_matrix, low_pass, high_pass, TR):
    """Compute amplitude of low-frequency fluctuation (ALFF).

    Parameters
    ----------
    data_matrix : numpy.ndarray
        data matrix points by timepoints
    low_pass : float
        low pass frequency in Hz
    high_pass : float
        high pass frequency in Hz
    TR : float
        repetition time in seconds

    Returns
    -------
    alff : numpy.ndarray
        ALFF values.

    Notes
    -----
    Implementation based on https://pubmed.ncbi.nlm.nih.gov/16919409/.
    """
    fs = 1 / TR  # sampling frequency
    n_voxels = data_matrix.shape[0]
    alff = np.zeros(n_voxels)
    for i_voxel in range(n_voxels):
        # get array of sample frequencies + power spectrum density
        array_of_sample_frequencies, power_spec_density = signal.periodogram(
            data_matrix[i_voxel, :],
            fs,
            scaling="density",
        )
        # square root of power spectrum density
        power_spec_density_sqrt = np.sqrt(power_spec_density)
        # get the position of the arguments closest to high_pass and low_pass, respectively
        ff_alff = [
            np.argmin(np.abs(array_of_sample_frequencies - high_pass)),
            np.argmin(np.abs(array_of_sample_frequencies - low_pass)),
        ]
        # alff for that voxel is 2 * the mean of the sqrt of the power spec density
        # from the value closest to the low pass cutoff, to the value closest
        # to the high pass pass cutoff
        alff[i_voxel] = len(ff_alff) * np.mean(power_spec_density_sqrt[ff_alff[0] : ff_alff[1]])

    # Add second dimension to array
    alff = alff[:, None]

    return alff


def calculate_psd_mean(bold_signal, valid_frames, sampling_interval):
    """Calculate the Power Spectral Density (PSD) mean from lagged autocovariance.

    Parameters
    ----------
    bold_signal : numpy.ndarray
        The input BOLD signal (voxels x time series).
    valid_frames : numpy.ndarray
        Binary array indicating valid frames (1s) and invalid frames (0s).
    sampling_interval : float
        Sampling interval, e.g., 2.2s.

    Returns
    -------
    numpy.ndarray
        PSD mean computed from the lagged autocovariance.

    Notes
    -----
    This function computes the lagged autocovariance for each voxel, then
    calculates the Power Spectral Density (PSD) mean from the autocovariance.

    The input arrays `bold_signal` and `valid_frames` must have the same number of time points.

    References
    ----------
    This function is adapted from Avi's interp_lag_dat.c.

    Examples
    --------
    >>> import numpy as np
    >>> bold_signal = np.random.randn(10, 100)  # Example BOLD signal
    >>> valid_frames = np.random.choice([0, 1], size=100)  # Example valid frames array
    >>> sampling_interval = 2.2  # Example sampling interval
    >>> psd_mean_result = calculate_psd_mean(bold_signal, valid_frames, sampling_interval)
    """
    n_voxels, n_vols = bold_signal.shape

    # Initialize variables
    autocovariance_mean = np.zeros((n_vols, n_voxels))
    psd_mean = np.zeros((n_voxels, n_vols))

    # Compute lagged autocovariance
    for i_voxel in range(n_voxels):
        for j_lag in range(1, (n_vols // 2) + 2):
            sum_term = 0
            num_valid_pairs = 0
            for k_vol in range(1, (n_vols - j_lag + 2)):
                if valid_frames[k_vol - 1] != 1 or valid_frames[j_lag + k_vol - 2] != 1:
                    continue

                sum_term += (
                    bold_signal[i_voxel, k_vol - 1] * bold_signal[i_voxel, k_vol + j_lag - 2]
                )
                num_valid_pairs += 1

            lag_time = (j_lag - 1) * sampling_interval / 20
            autocovariance_mean[j_lag - 1, i_voxel] = (
                sum_term * np.exp(-0.5 * lag_time * lag_time) / num_valid_pairs
            )

    # Mirror autocovariance
    for i_lag in range(2, (n_vols // 2) + 1):
        autocovariance_mean[n_vols - i_lag + 1, :] = autocovariance_mean[i_lag - 1, :]

    # Calculate PSD mean
    for i_voxel in range(n_voxels):
        autocovariance_mean_voxel = autocovariance_mean[:, i_voxel]
        for freq_index in range(1, (n_vols // 2) + 2):
            cos_values = np.cos(2 * np.pi * (freq_index - 1) * np.arange(n_vols) / n_vols)
            psd_mean[i_voxel, freq_index - 1] = np.dot(cos_values, autocovariance_mean_voxel)

    # Mirror PSD mean
    for freq_index in range(2, (n_vols // 2) + 1):
        psd_mean[:, n_vols - freq_index] = psd_mean[:, freq_index - 1]

    return psd_mean
