# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for calculating resting-state derivatives (ReHo and ALFF)."""

import nibabel as nb
import numpy as np
from nipype import logging
from scipy import signal
from scipy.stats import rankdata
from templateflow.api import get as get_template
from threadpoolctl import threadpool_limits

LOGGER = logging.getLogger('nipype.utils')


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
    kcc : numpy.ndarray of shape (V,)
        ReHo values.

    Notes
    -----
    From https://www.sciencedirect.com/science/article/pii/S0165178119305384#bib0045.
    """
    n_vertices = datat.shape[0]
    kcc = np.zeros(n_vertices)

    for i_vertex in range(n_vertices):  # loop through each voxel
        neighbor_idx = np.where(adjacency_matrix[i_vertex, :])[0]  # the index of neighbors
        neighborhood_idx = np.hstack((neighbor_idx, np.array(i_vertex)))

        neighborhood_data = datat[neighborhood_idx, :]

        # The ranks are summed and squared in float64, even if the data are float32,
        # because kc is a small difference between large sums of squares.
        rankeddata = np.zeros(neighborhood_data.shape, dtype=np.float64)
        # pull out index of voxel, timepoint
        n_neighbors, n_volumes = neighborhood_data.shape[0], neighborhood_data.shape[1]

        for j_neighbor in range(n_neighbors):
            # assign ranks to timepoints for each voxel
            rankeddata[j_neighbor, :] = rankdata(neighborhood_data[j_neighbor, :])

        rankmean = np.sum(rankeddata, axis=0)  # add up ranks
        # kc is the sum of the squared rankmean minus the timepoints into
        # the mean of the rankmean squared
        kc = np.sum(np.power(rankmean, 2)) - n_volumes * np.power(np.mean(rankmean), 2)

        # square number of neighbours, multiply by (cubed timepoint - timepoint)
        denom = np.power(n_neighbors, 2) * (np.power(n_volumes, 3) - n_volumes)

        # the voxel value is 12*kc divided by denom
        kcc[i_vertex] = 12 * kc / (denom)

    return kcc


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

    Notes
    -----
    Modified by Taylor Salo to loop over all vertices in faces.
    """
    surf = str(
        get_template(
            'fsLR',
            space=None,
            hemi=hemi,
            suffix='sphere',
            density='32k',
            raise_empty=True,
        )
    )
    surf = nb.load(surf)  # load via nibabel

    # Aggregate GIFTI data arrays into an ndarray or tuple of ndarray select the arrays in a
    # specific order
    vertices_faces = surf.agg_data(('pointset', 'triangle'))
    vertices = vertices_faces[0]
    faces = vertices_faces[1]
    n_vertices = vertices.shape[0]

    adjacency_matrix = np.zeros([n_vertices, n_vertices], dtype=bool)
    for i_face in range(faces.shape[0]):
        face = faces[i_face, :]  # pull out the face
        for vertex1 in face:
            for vertex2 in face:
                if vertex1 != vertex2:  # don't include the vertex as its own neighbor
                    adjacency_matrix[vertex1, vertex2] = True

    if not np.array_equal(adjacency_matrix, adjacency_matrix.T):
        raise ValueError('Adjacency matrix must be symmetric')
    return adjacency_matrix


def compute_alff(*, data_matrix, low_pass, high_pass, TR, sample_mask):
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
    sample_mask : numpy.ndarray or None
        (timepoints,) 1D array with 1s for good volumes and 0s for censored ones.

    Returns
    -------
    alff : numpy.ndarray
        ALFF values.

    Notes
    -----
    Implementation based on :footcite:t:`yu2007altered`,
    although the ALFF values are not scaled by the mean ALFF value across the brain.

    If a ``sample_mask`` is provided, then the power spectrum will be estimated using a
    Lomb-Scargle periodogram
    :footcite:p:`lomb1976least,scargle1982studies,townsend2010fast,taylorlomb`.

    Voxels are processed in blocks rather than one at a time,
    but the result is identical to the per-voxel calculation in :func:`_compute_alff_voxel`,
    which is used directly for voxels with non-finite values.
    Each block is processed in float64, so float32 data are never copied to float64 in full.
    Every reduction over time is performed on C-contiguous rows,
    the data-dependent matrix products from :func:`scipy.signal.lombscargle` are separate
    per-voxel :func:`numpy.dot` calls,
    and :func:`scipy.signal.periodogram` is called once per voxel,
    because the summation order of each depends on memory layout and array shape.

    BLAS is limited to one thread here,
    because each voxel's matrix products are small, so BLAS threads add overhead rather than
    speed.
    This also makes the result independent of the number of CPUs,
    since some OpenBLAS builds split these products differently across different numbers of
    threads, which changes the last bits of the float64 result.

    References
    ----------
    .. footbibliography::
    """
    with threadpool_limits(limits=1, user_api='blas'):
        return _compute_alff_blocks(
            data_matrix=data_matrix,
            low_pass=low_pass,
            high_pass=high_pass,
            TR=TR,
            sample_mask=sample_mask,
        )


def _compute_alff_blocks(*, data_matrix, low_pass, high_pass, TR, sample_mask):
    """Compute ALFF in blocks of voxels. See :func:`compute_alff`."""
    fs = 1 / TR  # sampling frequency
    n_voxels, n_volumes = data_matrix.shape

    if sample_mask is not None:
        time_arr = np.arange(n_volumes) * TR
        sample_mask_size_ok = sample_mask.size == time_arr.size
        if sample_mask_size_ok:
            time_arr = time_arr[sample_mask]
            frequencies_hz = np.linspace(0, 0.5 * fs, (n_volumes // 2) + 1)[1:]
            angular_frequencies = 2 * np.pi * frequencies_hz
            lombscargle_terms = _lombscargle_terms(time_arr, angular_frequencies)

    # Bound the memory used by the per-block copies and power spectra.
    block_size = max(1, (2**21) // max(n_volumes, 1))

    alff = np.zeros(n_voxels)
    for start in range(0, n_voxels, block_size):
        # A C-contiguous float64 copy, so that reductions along time match the per-voxel
        # calculation, so that float32 data are processed in float64,
        # and so that the caller's array is not modified.
        block = np.array(
            data_matrix[start : start + block_size],
            dtype=np.float64,
            order='C',
            copy=True,
        )

        # Voxels with data that are all the same value (esp. zeros) keep an ALFF of 0.
        sd_scale = np.nanstd(block, axis=1)
        nonconstant = sd_scale != 0
        if not nonconstant.any():
            continue

        if sample_mask is not None and not sample_mask_size_ok:
            raise ValueError(f'{sample_mask.size} != {n_volumes}')

        # Voxels with non-finite values use the per-voxel calculation,
        # so that NaN and inf propagate exactly as they always have.
        finite = np.isfinite(block).all(axis=1)
        fast_idx = np.flatnonzero(nonconstant & finite)
        slow_idx = np.flatnonzero(nonconstant & ~finite)

        # We will normalize data matrix over time.
        # This will ensure that the power spectra from the standard and Lomb-Scargle periodograms
        # have the same scale.
        # However, this also changes ALFF's scale, so we retain the SD to rescale ALFF.
        if sample_mask is not None:
            normalized = np.ascontiguousarray(block[fast_idx][:, sample_mask])
        else:
            normalized = block[fast_idx]

        normalized -= np.nanmean(normalized, axis=1, keepdims=True)
        normalized /= np.nanstd(normalized, axis=1, keepdims=True)

        # Voxels that are constant across the retained volumes become non-finite here.
        normalized_finite = np.isfinite(normalized).all(axis=1)
        slow_idx = np.sort(np.concatenate((slow_idx, fast_idx[~normalized_finite])))
        fast_idx = fast_idx[normalized_finite]
        normalized = normalized[normalized_finite]

        if fast_idx.size:
            if sample_mask is not None:
                power_spectrum = _lombscargle_normalized(normalized, *lombscargle_terms)
            else:
                # get array of sample frequencies + power spectrum density
                # The FFT is run one voxel at a time, because SciPy's FFT takes a different
                # (vectorized) code path for multiple rows, which changes the last bits.
                power_spectrum = []
                for row in normalized:
                    frequencies_hz, row_power_spectrum = signal.periodogram(
                        row,
                        fs,
                        scaling='spectrum',
                    )
                    power_spectrum.append(row_power_spectrum)

                power_spectrum = np.vstack(power_spectrum)

            ff_alff = _get_alff_band(frequencies_hz, low_pass=low_pass, high_pass=high_pass)
            # square root of power spectrum
            power_spectrum_sqrt = np.sqrt(power_spectrum)
            # alff for that voxel is 2 * the mean of the sqrt of the power spec
            # from the value closest to the low pass cutoff, to the value closest
            # to the high pass pass cutoff
            band = np.ascontiguousarray(power_spectrum_sqrt[:, ff_alff[0] : ff_alff[1]])
            block_alff = len(ff_alff) * np.nanmean(band, axis=1)
            # Rescale ALFF based on original BOLD scale
            block_alff *= sd_scale[fast_idx]
            alff[start + fast_idx] = block_alff

        for i_row in slow_idx:
            alff[start + i_row] = _compute_alff_voxel(
                voxel_data=block[i_row],
                low_pass=low_pass,
                high_pass=high_pass,
                TR=TR,
                sample_mask=sample_mask,
            )

    if alff.size != n_voxels:
        raise ValueError(f'{alff.shape} != {n_voxels}')
    return alff


def _get_alff_band(frequencies_hz, *, low_pass, high_pass):
    """Find the indices of the frequencies closest to the high-pass and low-pass cutoffs."""
    # get the position of the arguments closest to high_pass and low_pass, respectively
    if high_pass == 0:
        # If high_pass is 0, then we set it to the minimum frequency
        high_pass = frequencies_hz[0]

    if low_pass == 0:
        # If low_pass is 0, then we set it to the maximum frequency
        low_pass = frequencies_hz[-1]

    return [
        np.argmin(np.abs(frequencies_hz - high_pass)),
        np.argmin(np.abs(frequencies_hz - low_pass)),
    ]


def _lombscargle_terms(time_arr, angular_frequencies):
    """Compute the data-independent terms of :func:`scipy.signal.lombscargle`.

    These lines reproduce SciPy's implementation (SciPy >= 1.15, unweighted,
    ``floating_mean=False``), operation for operation,
    so that :func:`_lombscargle_normalized` returns exactly what
    ``scipy.signal.lombscargle(time_arr, y, angular_frequencies, normalize=True)`` returns.
    The equivalence is checked against SciPy in the test suite.
    """
    weights = np.ones_like(time_arr, dtype=np.float64)
    weights = weights * (1.0 / weights.sum())

    x = np.asarray(time_arr, dtype=np.float64).reshape(-1, 1)
    freqs = np.asarray(angular_frequencies, dtype=np.float64).reshape(1, -1)
    weights = weights.reshape(-1, 1)

    freqst = freqs * x
    coswt = np.cos(freqst)
    sinwt = np.sin(freqst)

    CC = np.dot(weights.T, coswt * coswt)
    SS = 1.0 - CC
    CS = np.dot(weights.T, coswt * sinwt)

    # phase offset to eliminate CS
    tau = 0.5 * np.arctan2(2.0 * CS, CC - SS)
    freqst_tau = freqst - tau
    coswt_tau = np.cos(freqst_tau)
    sinwt_tau = np.sin(freqst_tau)

    CC = np.dot(weights.T, coswt_tau * coswt_tau)
    SS = 1.0 - CC

    epsneg = np.finfo(dtype=np.float64).epsneg
    CC[CC < epsneg] = epsneg
    SS[SS < epsneg] = epsneg

    return weights, coswt_tau, sinwt_tau, CC, SS


def _lombscargle_normalized(data, weights, coswt_tau, sinwt_tau, CC, SS):
    """Compute normalized Lomb-Scargle periodograms for each row of ``data``.

    The terms that depend on the data are computed with one :func:`numpy.dot` call per row,
    with the same shapes SciPy uses, because a single matrix product over all rows
    sums in a different order and changes the last bits of the result.
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    n_rows, n_freqs = data.shape[0], coswt_tau.shape[1]
    YC = np.empty((n_rows, n_freqs))
    YS = np.empty((n_rows, n_freqs))
    YY = np.empty(n_rows)
    for i_row in range(n_rows):
        # Build the same arrays that SciPy builds: freshly allocated, with the same shapes and
        # strides. NumPy chooses how to call BLAS based on shapes and strides,
        # and some OpenBLAS kernels sum differently depending on memory alignment.
        y = data[i_row].copy().reshape(-1, 1)
        weights_y = weights * y
        YC[i_row] = np.dot(weights_y.T, coswt_tau)[0]
        YS[i_row] = np.dot(weights_y.T, sinwt_tau)[0]
        YY[i_row] = np.dot(weights_y.T, y)[0, 0]

    a = YC / CC
    b = YS / SS
    pgram = 2.0 * (a * YC + b * YS)
    pgram *= (0.5 / YY)[:, None]
    return pgram


def _compute_alff_voxel(*, voxel_data, low_pass, high_pass, TR, sample_mask):
    """Compute ALFF for a single voxel.

    This is the reference calculation that :func:`compute_alff` reproduces in blocks.
    ``voxel_data`` is modified in place.
    """
    fs = 1 / TR  # sampling frequency
    n_volumes = voxel_data.size

    # Check if the voxel's data are all the same value (esp. zeros).
    if np.nanstd(voxel_data) == 0:
        return 0

    sd_scale = np.nanstd(voxel_data)

    if sample_mask is not None:
        voxel_data_censored = voxel_data[sample_mask]
        voxel_data_censored -= np.nanmean(voxel_data_censored)
        voxel_data_censored /= np.nanstd(voxel_data_censored)

        time_arr = np.arange(n_volumes) * TR
        if sample_mask.size != time_arr.size:
            raise ValueError(f'{sample_mask.size} != {time_arr.size}')
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
        voxel_data -= np.nanmean(voxel_data)
        voxel_data /= np.nanstd(voxel_data)
        # get array of sample frequencies + power spectrum density
        frequencies_hz, power_spectrum = signal.periodogram(
            voxel_data,
            fs,
            scaling='spectrum',
        )

    # square root of power spectrum
    power_spectrum_sqrt = np.sqrt(power_spectrum)
    ff_alff = _get_alff_band(frequencies_hz, low_pass=low_pass, high_pass=high_pass)
    alff = np.float64(len(ff_alff) * np.nanmean(power_spectrum_sqrt[ff_alff[0] : ff_alff[1]]))
    # Rescale ALFF based on original BOLD scale
    alff *= sd_scale
    return alff
