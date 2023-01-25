# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for calculating functional connectivity in NIFTI files."""
import nibabel as nb
import numpy as np
from scipy import signal
from scipy.stats import rankdata
from templateflow.api import get as get_template


def extract_timeseries_funct(in_file, mask, atlas, node_labels_file):
    """Use Nilearn NiftiLabelsMasker to extract timeseries.

    Parameters
    ----------
    in_file : str
        bold file timeseries
    mask : str
        BOLD file's associated brain mask file.
    atlas : str
        atlas in the same space with bold
    node_labels_file : str
        The name of each node in the atlas, in the same order as the values in the atlas file.

    Returns
    -------
    timeseries_file : str
        extracted timeseries filename
    """
    import os
    import warnings

    import numpy as np
    import pandas as pd
    from nilearn.input_data import NiftiLabelsMasker

    timeseries_file = os.path.abspath("timeseries.tsv")

    node_labels_df = pd.read_table(node_labels_file, index_col="index")

    # Explicitly remove label corresponding to background (index=0), if present.
    if 0 in node_labels_df.index:
        node_labels_df = node_labels_df.drop(index=[0])

    node_labels = node_labels_df["name"].tolist()

    # Extract time series with nilearn
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        labels=node_labels,
        mask_img=mask,
        smoothing_fwhm=None,
        standardize=False,
        resampling_target=None,  # they should be in the same space/resolution already
    )
    timeseries_arr = masker.fit_transform(in_file)

    # Region indices in the atlas may not be sequential, so we map them to sequential ints.
    seq_mapper = {idx: i for i, idx in enumerate(masker.labels_)}

    if timeseries_arr.shape[1] != len(node_labels):
        warnings.warn(
            f"The number of detected nodes ({timeseries_arr.shape[1]}) does not equal "
            f"the number of expected nodes ({len(node_labels)}) in {atlas}."
        )

        new_timeseries_arr = np.zeros(
            (timeseries_arr.shape[0], len(node_labels)),
            dtype=timeseries_arr.dtype,
        )
        for col in range(timeseries_arr.shape[1]):
            label_col = seq_mapper[masker.labels_[col]]
            new_timeseries_arr[:, label_col] = timeseries_arr[:, col]

        timeseries_arr = new_timeseries_arr

    # The time series file is tab-delimited, with node names included in the first row.
    timeseries_df = pd.DataFrame(data=timeseries_arr, columns=node_labels)
    timeseries_df.to_csv(timeseries_file, sep="\t", index=False)

    return timeseries_file


def extract_ptseries(in_file, node_labels_file):
    """Extract time series and parcel names from ptseries CIFTI file.

    Parameters
    ----------
    in_file : str
        Path to a ptseries (parcellated time series) CIFTI file.
    node_labels_file : str
        The name of each node in the atlas, in the same order as the values in the atlas file.

    Returns
    -------
    timeseries_file : str
        The saved tab-delimited time series file.
        Column headers are the names of the parcels from the CIFTI file.
    """
    import os
    import warnings

    import nibabel as nib
    import numpy as np
    import pandas as pd

    timeseries_file = os.path.abspath("timeseries.tsv")

    node_labels_df = pd.read_table(node_labels_file, index_col="index")

    # Explicitly remove label corresponding to background (index=0), if present.
    if 0 in node_labels_df.index:
        node_labels_df = node_labels_df.drop(index=[0])

    node_labels = node_labels_df["name"].tolist()

    img = nib.load(in_file)
    assert "ConnParcelSries" in img.nifti_header.get_intent(), img.nifti_header.get_intent()

    # Load node names from CIFTI file.
    # First axis should be time, second should be parcels
    ax = img.header.get_axis(1)
    detected_node_labels = ax.name

    # If there are nodes in the CIFTI that aren't in the node labels file, raise an error.
    found_but_not_expected = sorted(list(set(detected_node_labels) - set(node_labels)))
    if found_but_not_expected:
        raise ValueError(
            "Mismatch found between atlas nodes and node labels file: "
            f"{', '.join(found_but_not_expected)}"
        )

    # Load the extracted time series from the CIFTI file.
    timeseries_arr = np.array(img.get_fdata())

    # Region indices in the atlas may not be sequential, so we map them to sequential ints.
    seq_mapper = {label: i for i, label in enumerate(node_labels)}

    # Check if all of the nodes in the atlas node labels file are represented.
    if timeseries_arr.shape[1] != len(node_labels):
        warnings.warn(
            f"The number of detected nodes ({timeseries_arr.shape[1]}) does not equal "
            f"the number of expected nodes ({len(node_labels)}) in atlas."
        )

        new_timeseries_arr = np.zeros(
            (timeseries_arr.shape[0], len(node_labels)),
            dtype=timeseries_arr.dtype,
        )
        for col in range(timeseries_arr.shape[1]):
            label_col = seq_mapper[detected_node_labels[col]]
            new_timeseries_arr[:, label_col] = timeseries_arr[:, col]

        timeseries_arr = new_timeseries_arr

    # Place the data in a DataFrame and save to a TSV
    df = pd.DataFrame(columns=node_labels, data=timeseries_arr)
    df.to_csv(timeseries_file, index=False, sep="\t")

    return timeseries_file


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
            rankeddata[j, :] = rankdata(
                neidata[
                    j,
                ]
            )  # assign ranks to timepoints for each voxel
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
    lowpass : float
        low pass frequency in Hz
    highpass : float
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


def compute_functional_connectivity(in_file):
    """Compute pair-wise correlations between columns in a tab-delimited file.

    Parameters
    ----------
    in_file : str
        Path to a tab-delimited file with time series.
        Column headers should indicate the nodes/parcels of the atlas.

    Returns
    -------
    correlations_file : str
        The saved tab-delimited correlations file.
        The first column is named "Node", and it is the node names from the time series file.
        The remaining columns are the names of the nodes.
    """
    import os

    import pandas as pd

    df = pd.read_table(in_file)

    correlations_file = os.path.abspath("correlations.tsv")

    # Compute Pearson correlation
    df_corr = df.corr(method="pearson")
    df_corr.to_csv(correlations_file, index_label="Node", sep="\t")

    return correlations_file
