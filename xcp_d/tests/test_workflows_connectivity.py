"""Tests for connectivity matrix calculation."""
import os
import sys

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker

from xcp_d.utils.bids import _get_tr
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflows.connectivity import (
    init_cifti_functional_connectivity_wf,
    init_nifti_functional_connectivity_wf,
)

np.set_printoptions(threshold=sys.maxsize)


def test_nifti_conn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the nifti workflow."""
    tmpdir = tmp_path_factory.mktemp("test_nifti_conn")

    bold_file = fmriprep_with_freesurfer_data["nifti_file"]
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]
    template_to_t1w_xform = fmriprep_with_freesurfer_data["template_to_t1w_xform"]
    boldref = fmriprep_with_freesurfer_data["boldref"]
    t1w_to_native_xform = fmriprep_with_freesurfer_data["t1w_to_native_xform"]

    # Generate fake signal
    bold_data = read_ndata(bold_file, bold_mask)
    fake_signal = np.random.randint(1, 500, size=bold_data.shape)
    fake_bold_file = os.path.join(tmpdir, "fake_signal_file.nii.gz")
    write_ndata(
        fake_signal,
        template=bold_file,
        mask=bold_mask,
        TR=_get_tr(nb.load(bold_file)),
        filename=fake_bold_file,
    )
    assert os.path.isfile(fake_bold_file)

    # Let's define the inputs and create the node
    connectivity_wf = init_nifti_functional_connectivity_wf(
        output_dir=tmpdir,
        mem_gb=4,
        name="connectivity_wf",
        omp_nthreads=2,
    )
    connectivity_wf.inputs.inputnode.template_to_t1w = template_to_t1w_xform
    connectivity_wf.inputs.inputnode.t1w_to_native = t1w_to_native_xform
    connectivity_wf.inputs.inputnode.clean_bold = fake_bold_file
    connectivity_wf.inputs.inputnode.bold_file = bold_file
    connectivity_wf.inputs.inputnode.bold_mask = bold_mask
    connectivity_wf.inputs.inputnode.ref_file = boldref
    connectivity_wf.base_dir = tmpdir
    connectivity_wf.run()

    # Let's find the correct time series file
    connect_dir = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/nifti_connect/mapflow/_nifti_connect9",
    )

    coverage = os.path.join(connect_dir, "coverage.tsv")
    assert os.path.isfile(coverage), os.listdir(connect_dir)
    timeseries = os.path.join(connect_dir, "timeseries.tsv")
    assert os.path.isfile(timeseries), os.listdir(connect_dir)
    correlations = os.path.join(connect_dir, "correlations.tsv")
    assert os.path.isfile(correlations), os.listdir(connect_dir)

    # Read that into a df
    coverage_arr = pd.read_table(coverage, index_col="Node").to_numpy()
    correlations_arr = pd.read_table(correlations, index_col="Node").to_numpy()
    assert correlations_arr.shape == (1000, 1000)
    available_parcels = np.where(np.squeeze(coverage_arr) > 0)[0]

    assert available_parcels.size == 973

    # Parcels with <50% coverage should have NaNs
    assert np.array_equal(np.squeeze(coverage_arr) < 0.5, np.isnan(np.diag(correlations_arr)))

    # Drop missing parcels (there are 34 for the 1000parcel )
    correlations_arr = correlations_arr[available_parcels, :]
    correlations_arr = correlations_arr[:, available_parcels]
    assert correlations_arr.shape == (973, 973)

    # Now let's get the ground truth. First, we should locate the atlas
    atlas_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/warp_atlases_to_bold_space/mapflow/_warp_atlases_to_bold_space9",
        "Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm_trans.nii.gz",
    )
    assert os.path.isfile(atlas_file)

    # Masking img
    masker = NiftiLabelsMasker(
        atlas_file,
        mask_img=bold_mask,
        smoothing_fwhm=None,
        standardize=False,
    )
    masker.fit(fake_bold_file)
    signals = masker.transform(fake_bold_file)

    # The "ground truth" matrix
    calculated_correlations = np.corrcoef(signals.T)
    assert calculated_correlations.shape == (998, 998)
    calculated_correlations = calculated_correlations[available_parcels, :]
    calculated_correlations = calculated_correlations[:, available_parcels]
    assert calculated_correlations.shape == (973, 973)

    # If we replace the bad parcels' results in the "ground truth" matrix with NaNs,
    # the resulting matrix should match the workflow-generated one.
    bad_parcel_idx = np.where(np.isnan(np.diag(correlations_arr)))[0]
    calculated_correlations[bad_parcel_idx, :] = np.nan
    calculated_correlations[:, bad_parcel_idx] = np.nan

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    assert np.allclose(correlations_arr, calculated_correlations, atol=0.01, equal_nan=True)


def test_cifti_conn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    tmpdir = tmp_path_factory.mktemp("test_cifti_conn")

    bold_file = fmriprep_with_freesurfer_data["cifti_file"]
    TR = _get_tr(nb.load(bold_file))

    # Generate fake signal
    bold_data = read_ndata(bold_file)
    fake_signal = np.random.randint(1, 500, size=bold_data.shape).astype(np.float32)
    # Make half the vertices all zeros
    fake_signal[:5000, :] = 0
    fake_bold_file = os.path.join(tmpdir, "fake_signal_file.dtseries.nii")
    write_ndata(
        fake_signal,
        template=bold_file,
        TR=TR,
        filename=fake_bold_file,
    )
    assert os.path.isfile(fake_bold_file)

    # Create the node and a tmpdir to write its results out to
    connectivity_wf = init_cifti_functional_connectivity_wf(
        output_dir=tmpdir,
        mem_gb=4,
        omp_nthreads=2,
        name="connectivity_wf",
    )
    connectivity_wf.inputs.inputnode.clean_bold = fake_bold_file
    connectivity_wf.inputs.inputnode.bold_file = bold_file
    connectivity_wf.base_dir = tmpdir
    connectivity_wf.run()

    # Let's find the correct parcellated file
    connect_dir = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/cifti_connect/mapflow/_cifti_connect9",
    )

    # Let's find the cifti files
    pscalar = os.path.join(connect_dir, "coverage.pscalar.nii")
    assert os.path.isfile(pscalar), os.listdir(connect_dir)

    ptseries = os.path.join(connect_dir, "timeseries.ptseries.nii")
    assert os.path.isfile(ptseries), os.listdir(connect_dir)

    pconn = os.path.join(connect_dir, "correlations.pconn.nii")
    assert os.path.isfile(pconn), os.listdir(connect_dir)

    # Let's find the tsv files
    coverage = os.path.join(connect_dir, "coverage.tsv")
    assert os.path.isfile(coverage), os.listdir(connect_dir)

    timeseries = os.path.join(connect_dir, "timeseries.tsv")
    assert os.path.isfile(timeseries), os.listdir(connect_dir)

    correlations = os.path.join(connect_dir, "correlations.tsv")
    assert os.path.isfile(correlations), os.listdir(connect_dir)

    # Let's read in the ciftis' data
    pscalar_arr = nb.load(pscalar).get_fdata().T
    assert pscalar_arr.shape == (1000, 1)
    ptseries_arr = nb.load(ptseries).get_fdata()
    assert ptseries_arr.shape == (60, 1000)
    pconn_arr = nb.load(pconn).get_fdata()
    assert pconn_arr.shape == (1000, 1000)

    # Read in the tsvs' data
    coverage_arr = pd.read_table(coverage, index_col="Node").to_numpy()
    timeseries_arr = pd.read_table(timeseries).to_numpy()
    correlations_arr = pd.read_table(correlations, index_col="Node").to_numpy()

    assert coverage_arr.shape == pscalar_arr.shape
    assert timeseries_arr.shape == ptseries_arr.shape
    assert correlations_arr.shape == pconn_arr.shape

    assert np.allclose(coverage_arr, pscalar_arr)
    assert np.allclose(timeseries_arr, ptseries_arr, equal_nan=True)
    assert np.allclose(correlations_arr, pconn_arr, equal_nan=True)

    # Calculate correlations from timeseries data
    calculated_correlations = np.corrcoef(ptseries_arr.T)
    assert calculated_correlations.shape == (1000, 1000)
    bad_parcels_idx = np.where(np.isnan(np.diag(calculated_correlations)))[0]
    good_parcels_idx = np.where(~np.isnan(np.diag(calculated_correlations)))[0]

    # Parcels with <50% coverage should have NaNs
    first_good_parcel_corrs = pconn_arr[good_parcels_idx[0], :]

    # The number of NaNs for a good parcel's correlations should match the number of bad parcels.
    assert np.sum(np.isnan(first_good_parcel_corrs)) == bad_parcels_idx.size

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    if not np.array_equal(np.isnan(pconn_arr), np.isnan(calculated_correlations)):
        mismatch_idx = np.vstack(
            np.where(np.isnan(pconn_arr) != np.isnan(calculated_correlations))
        ).T
        raise ValueError(f"{mismatch_idx}\n\n{np.where(np.isnan(pconn_arr))}")

    if not np.allclose(pconn_arr, calculated_correlations, atol=0.01, equal_nan=True):
        diff = pconn_arr - calculated_correlations
        raise ValueError(np.nanmax(np.abs(diff)))
