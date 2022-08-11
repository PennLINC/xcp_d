# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
nifti functional connectivity
"""
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
from scipy.stats import rankdata
from scipy import signal
import nibabel as nb
from templateflow.api import get as get_template


def extract_timeseries_funct(in_file, atlas, timeseries, fconmatrix):
    """
     This function used Nilearn *NiftiLabelsMasker*
     to extact timeseries
    in_file
       bold file timeseries
    atlas
       atlas in the same space with bold
    timeseries
      extracted timesries filename
    fconmatrix
      functional connectivity matrix filename

    """
    masker = NiftiLabelsMasker(labels_img=atlas,
                               smoothing_fwhm=None,
                               standardize=False)
    time_series = masker.fit_transform(in_file)
    correlation_matrices = np.corrcoef(time_series.T)

    np.savetxt(fconmatrix, correlation_matrices, delimiter=",")
    np.savetxt(timeseries, time_series, delimiter=",")

    return timeseries, fconmatrix


def compute_2d_reho(datat, adjacency_matrix):
    """
    https://www.sciencedirect.com/science/article/pii/S0165178119305384#bib0045
    this function compute 2d reho

    datat: numpy darray
       data matrix in vertices by timepoints
    adjacency_matrix : numpy matrix
       surface adjacency matrix

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
            rankeddata[j, :] = rankdata(neidata[j, ])  # assign ranks to timepoints for each voxel
        rankmean = np.sum(rankeddata, axis=0)  # add up ranks
        # KC is the sum of the squared rankmean minus the timepoints into the mean of the rankmean squared
        KC = np.sum(np.power(rankmean, 2)) - \
            timepoint * np.power(np.mean(rankmean), 2)
        # square number of neighbours, multiply by (cubed timepoint - timepoint)
        denom = np.power(neigbor, 2) * (np.power(timepoint, 3) - timepoint)
        # the voxel value is 12*KC divided by denom
        KCC[i] = 12 * KC / (denom)

    return KCC


def mesh_adjacency(hemi):
    # surface sphere to be load from templateflow
    # either left or right hemisphere

    surf = str(
        get_template("fsLR",
                     space='fsaverage',
                     hemi=hemi,
                     suffix='sphere',
                     density='32k'))  # Get relevant template

    surf = nb.load(surf)  # load via nibabel
    #  Aggregate GIFTI data arrays into an ndarray or tuple of ndarray
    # select the arrays in a specific order
    vertices_faces = surf.agg_data(('pointset', 'triangle'))
    vertices = vertices_faces[0]  # the first array of the tuple
    faces = vertices_faces[1]  # the second array in the tuples
    # create an array of 0s = voxel*voxel
    A = np.zeros([len(vertices), len(vertices)], dtype=np.uint8)

    for i in range(1, len(faces)): # looping thorugh each value in faces
        A[faces[i, 0], faces[i, 2]] = 1 # use to index into A and turn select values to 1
        A[faces[i, 1], faces[i, 1]] = 1
        A[faces[i, 2], faces[i, 0]] = 1

    return A + A.T # transpose A and add it to itself


def compute_alff(data_matrix, low_pass, high_pass, TR):
    """
     https://pubmed.ncbi.nlm.nih.gov/16919409/

     compute ALFF
    data_matrix: numpy darray
        data matrix points by timepoints
    lowpass: numpy float
        low pass frequency in Hz
    highpass : numpy float
        high pass frequency in Hz
    TR: numpy float
       repetition time in seconds
    """
    fs = 1 / TR
    alff = np.zeros(data_matrix.shape[0])
    for i in range(data_matrix.shape[0]):
        fx, Pxx_den = signal.periodogram(data_matrix[i, :],
                                         fs,
                                         scaling='spectrum')
        # fx, Pxx_den = signal.periodogram(data_matrix[i,:], fs,scaling='density')
        pxx_sqrt = np.sqrt(Pxx_den)
        ff_alff = [
            np.argmin(np.abs(fx - high_pass)),
            np.argmin(np.abs(fx - low_pass))
        ]
        alff[i] = len(ff_alff) * np.mean(pxx_sqrt[ff_alff[0]:ff_alff[1]])
    alff = np.reshape(alff, [len(alff), 1])
    return alff
