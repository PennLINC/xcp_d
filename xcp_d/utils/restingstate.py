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


def compute_alff(data_matrix, low_pass, high_pass, TR, sample_mask=None):
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
    sample_mask : numpy.ndarray
        (timepoints,) 1D array with 1s for good volumes and 0s for censored ones.

    Returns
    -------
    alff : numpy.ndarray
        ALFF values.

    Notes
    -----
    Implementation based on https://pubmed.ncbi.nlm.nih.gov/16919409/.
    """
    fs = 1 / TR  # sampling frequency
    n_voxels, n_volumes = data_matrix.shape
    if sample_mask is None:
        sample_mask = np.ones(n_volumes, dtype=int)
    else:
        LOGGER.warning(
            "Outlier volumes detected. ALFF will be calculated using Lomb-Scargle method."
        )

    sample_mask = sample_mask.astype(bool)
    assert sample_mask.size == n_volumes, f"{sample_mask.size} != {n_volumes}"

    alff = np.zeros(n_voxels)
    for i_voxel in range(n_voxels):
        voxel_data = data_matrix[i_voxel, :]
        # Check if the voxel's data are all the same value (esp. zeros).
        # Set ALFF to 0 in that case and move on to the next voxel.
        if np.std(voxel_data) == 0:
            alff[i_voxel] = 0
            continue

        # Normalize data matrix over time. This will ensure that the standard periodogram and
        # Lomb-Scargle periodogram will have the same scale.
        voxel_data -= np.mean(voxel_data)
        voxel_data /= np.std(voxel_data)

        if sample_mask.sum() != sample_mask.size:
            voxel_data_censored = voxel_data[sample_mask]
            time_arr = np.arange(0, n_volumes * TR, TR)
            assert sample_mask.size == time_arr.size, f"{sample_mask.size} != {time_arr.size}"
            time_arr = time_arr[sample_mask]
            frequencies_hz = np.linspace(0, 0.5 * fs, (n_volumes // 2) + 1)[1:]
            angular_frequencies = 2 * np.pi * frequencies_hz
            power_spectrum = signal.lombscargle(
                time_arr,
                voxel_data_censored,
                angular_frequencies,
                normalize=True,
            )
        else:
            # get array of sample frequencies + power spectrum density
            frequencies_hz, power_spectrum = signal.periodogram(
                voxel_data,
                fs,
                scaling="spectrum",
            )

        # square root of power spectrum
        power_spectrum_sqrt = np.sqrt(power_spectrum)
        # get the position of the arguments closest to high_pass and low_pass, respectively
        ff_alff = [
            np.argmin(np.abs(frequencies_hz - high_pass)),
            np.argmin(np.abs(frequencies_hz - low_pass)),
        ]
        # alff for that voxel is 2 * the mean of the sqrt of the power spec
        # from the value closest to the low pass cutoff, to the value closest
        # to the high pass pass cutoff
        alff[i_voxel] = len(ff_alff) * np.mean(power_spectrum_sqrt[ff_alff[0] : ff_alff[1]])

    assert alff.size == n_voxels, f"{alff.shape} != {n_voxels}"

    # Add second dimension to array
    alff = alff[:, None]

    return alff
