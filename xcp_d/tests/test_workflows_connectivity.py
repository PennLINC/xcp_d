"""Tests for connectivity matrix calculation."""
import os
import sys

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

from xcp_d.tests.utils import get_nodes
from xcp_d.utils.bids import _get_tr
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflows.connectivity import (
    init_functional_connectivity_cifti_wf,
    init_functional_connectivity_nifti_wf,
)

np.set_printoptions(threshold=sys.maxsize)


def test_nifti_conn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the nifti workflow."""
    tmpdir = tmp_path_factory.mktemp("test_nifti_conn")

    bold_file = fmriprep_with_freesurfer_data["nifti_file"]
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]
    template_to_anat_xfm = fmriprep_with_freesurfer_data["template_to_anat_xfm"]
    boldref = fmriprep_with_freesurfer_data["boldref"]
    anat_to_native_xfm = fmriprep_with_freesurfer_data["anat_to_native_xfm"]

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

    # Let's define the inputs and create the workflow
    connectivity_wf = init_functional_connectivity_nifti_wf(
        output_dir=tmpdir,
        min_coverage=0.5,
        mem_gb=4,
        name="connectivity_wf",
        omp_nthreads=2,
    )
    connectivity_wf.inputs.inputnode.template_to_anat_xfm = template_to_anat_xfm
    connectivity_wf.inputs.inputnode.anat_to_native_xfm = anat_to_native_xfm
    connectivity_wf.inputs.inputnode.denoised_bold = fake_bold_file
    connectivity_wf.inputs.inputnode.name_source = bold_file
    connectivity_wf.inputs.inputnode.bold_mask = bold_mask
    connectivity_wf.inputs.inputnode.boldref = boldref
    connectivity_wf.base_dir = tmpdir
    connectivity_wf_res = connectivity_wf.run()
    nodes = get_nodes(connectivity_wf_res)

    n_parcels, n_parcels_in_atlas = 1000, 1000

    # Let's find the correct workflow outputs
    atlas_file = nodes["connectivity_wf.warp_atlases_to_bold_space"].get_output("output_image")[9]
    assert os.path.isfile(atlas_file)
    coverage = nodes["connectivity_wf.nifti_connect"].get_output("coverage")[9]
    assert os.path.isfile(coverage)
    timeseries = nodes["connectivity_wf.nifti_connect"].get_output("timeseries")[9]
    assert os.path.isfile(timeseries)
    correlations = nodes["connectivity_wf.nifti_connect"].get_output("correlations")[9]
    assert os.path.isfile(correlations)

    # Read that into a df
    coverage_df = pd.read_table(coverage, index_col="Node")
    coverage_arr = coverage_df.to_numpy()
    assert coverage_arr.shape[0] == n_parcels
    correlations_arr = pd.read_table(correlations, index_col="Node").to_numpy()
    assert correlations_arr.shape == (n_parcels, n_parcels)

    # Parcels with <50% coverage should have NaNs
    assert np.array_equal(np.squeeze(coverage_arr) < 0.5, np.isnan(np.diag(correlations_arr)))

    # Now to get ground truth correlations
    # Masking img
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        labels=coverage_df.index.tolist(),
        smoothing_fwhm=None,
        standardize=False,
    )
    masker.fit(fake_bold_file)
    signals = masker.transform(fake_bold_file)

    atlas_idx = np.arange(len(coverage_df.index.tolist()), dtype=int)
    idx_not_in_atlas = np.setdiff1d(atlas_idx + 1, masker.labels_)
    idx_in_atlas = np.array(masker.labels_, dtype=int) - 1
    n_partial_parcels = np.where(coverage_df["coverage"] >= 0.5)[0].size

    # Drop missing parcels
    correlations_arr = correlations_arr[idx_in_atlas, :]
    correlations_arr = correlations_arr[:, idx_in_atlas]
    assert correlations_arr.shape == (n_parcels_in_atlas, n_parcels_in_atlas)

    # The masker.labels_ attribute only contains the labels that were found
    assert idx_not_in_atlas.size == 0
    assert idx_in_atlas.size == n_parcels_in_atlas

    # The "ground truth" matrix
    calculated_correlations = np.corrcoef(signals.T)
    assert calculated_correlations.shape == (n_parcels_in_atlas, n_parcels_in_atlas)

    # If we replace the bad parcels' results in the "ground truth" matrix with NaNs,
    # the resulting matrix should match the workflow-generated one.
    bad_parcel_idx = np.where(np.isnan(np.diag(correlations_arr)))[0]
    assert bad_parcel_idx.size == n_parcels_in_atlas - n_partial_parcels
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
    connectivity_wf = init_functional_connectivity_cifti_wf(
        output_dir=tmpdir,
        min_coverage=0.5,
        mem_gb=4,
        omp_nthreads=2,
        name="connectivity_wf",
    )
    connectivity_wf.inputs.inputnode.denoised_bold = fake_bold_file
    connectivity_wf.inputs.inputnode.name_source = bold_file
    connectivity_wf.base_dir = tmpdir
    connectivity_wf_res = connectivity_wf.run()
    nodes = get_nodes(connectivity_wf_res)

    # Let's find the cifti files
    pscalar = nodes["connectivity_wf.cifti_connect"].get_output("coverage_ciftis")[9]
    assert os.path.isfile(pscalar)
    timeseries_ciftis = nodes["connectivity_wf.cifti_connect"].get_output("timeseries_ciftis")[9]
    assert os.path.isfile(timeseries_ciftis)
    correlation_ciftis = nodes["connectivity_wf.cifti_connect"].get_output("correlation_ciftis")[9]
    assert os.path.isfile(correlation_ciftis)

    # Let's find the tsv files
    coverage = nodes["connectivity_wf.cifti_connect"].get_output("coverage")[9]
    assert os.path.isfile(coverage)
    timeseries = nodes["connectivity_wf.cifti_connect"].get_output("timeseries")[9]
    assert os.path.isfile(timeseries)
    correlations = nodes["connectivity_wf.cifti_connect"].get_output("correlations")[9]
    assert os.path.isfile(correlations)

    # Let's read in the ciftis' data
    pscalar_arr = nb.load(pscalar).get_fdata().T
    assert pscalar_arr.shape == (1000, 1)
    ptseries_arr = nb.load(timeseries_ciftis).get_fdata()
    assert ptseries_arr.shape == (60, 1000)
    pconn_arr = nb.load(correlation_ciftis).get_fdata()
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
