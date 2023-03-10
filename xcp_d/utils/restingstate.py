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
    alff = np.zeros(data_matrix.shape[0])  # Create a matrix of zeros in the shape of
    # number of voxels
    for ii in range(data_matrix.shape[0]):  # Loop through the voxels
        # get array of sample frequencies + power spectrum density
        array_of_sample_frequencies, power_spec_density = signal.periodogram(
            data_matrix[ii, :], fs, scaling="spectrum"
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
        alff[ii] = len(ff_alff) * np.mean(power_spec_density_sqrt[ff_alff[0] : ff_alff[1]])
    # reshape alff so it's no longer 1 dimensional, but a #ofvoxels by 1 matrix
    alff = np.reshape(alff, [len(alff), 1])
    return alff
