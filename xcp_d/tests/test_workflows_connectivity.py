"""Tests for connectivity matrix calculation."""

import os
import sys

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

from xcp_d import config
from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.connectivity import _sanitize_nifti_atlas
from xcp_d.tests.tests import mock_config
from xcp_d.tests.utils import get_nodes
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_nifti
from xcp_d.utils.bids import _get_tr
from xcp_d.utils.utils import _create_mem_gb, get_std2bold_xfms
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflows.bold.connectivity import (
    init_functional_connectivity_cifti_wf,
    init_functional_connectivity_nifti_wf,
)
from xcp_d.workflows.parcellation import init_load_atlases_wf

np.set_printoptions(threshold=sys.maxsize)


def test_init_load_atlases_wf_nifti(ds001419_data, tmp_path_factory):
    """Test init_load_atlases_wf with a nifti input."""
    tmpdir = tmp_path_factory.mktemp("test_init_load_atlases_wf_nifti")

    bold_file = ds001419_data["nifti_file"]

    with mock_config():
        config.execution.xcp_d_dir = tmpdir
        config.workflow.file_format = "nifti"
        config.execution.atlases = ["4S156Parcels", "Glasser"]
        config.nipype.omp_nthreads = 1

        load_atlases_wf = init_load_atlases_wf(name="load_atlases_wf")
        load_atlases_wf.inputs.inputnode.name_source = bold_file
        load_atlases_wf.inputs.inputnode.bold_file = bold_file
        load_atlases_wf.base_dir = tmpdir
        load_atlases_wf_res = load_atlases_wf.run()

        nodes = get_nodes(load_atlases_wf_res)
        atlas_names = nodes["load_atlases_wf.warp_atlases_to_bold_space"].get_output(
            "output_image"
        )
        assert len(atlas_names) == 2


def test_init_load_atlases_wf_cifti(ds001419_data, tmp_path_factory):
    """Test init_load_atlases_wf with a cifti input."""
    tmpdir = tmp_path_factory.mktemp("test_init_load_atlases_wf_cifti")

    bold_file = ds001419_data["cifti_file"]

    with mock_config():
        config.execution.xcp_d_dir = tmpdir
        config.workflow.file_format = "cifti"
        config.execution.atlases = ["4S156Parcels", "Glasser"]
        config.nipype.omp_nthreads = 1

        load_atlases_wf = init_load_atlases_wf(name="load_atlases_wf")
        load_atlases_wf.inputs.inputnode.name_source = bold_file
        load_atlases_wf.inputs.inputnode.bold_file = bold_file
        load_atlases_wf.base_dir = tmpdir
        load_atlases_wf_res = load_atlases_wf.run()

        nodes = get_nodes(load_atlases_wf_res)
        atlas_names = nodes["load_atlases_wf.ds_atlas"].get_output("out_file")
        assert len(atlas_names) == 2


def test_init_functional_connectivity_nifti_wf(ds001419_data, tmp_path_factory):
    """Test the nifti workflow."""
    tmpdir = tmp_path_factory.mktemp("test_init_functional_connectivity_nifti_wf")

    bold_file = ds001419_data["nifti_file"]
    boldref = ds001419_data["boldref"]
    bold_mask = ds001419_data["brain_mask_file"]

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
    mem_gbx = _create_mem_gb(bold_file)

    # Create a fake temporal mask to satisfy the workflow
    n_volumes = bold_data.shape[1]
    censoring_df = pd.DataFrame(
        columns=["framewise_displacement", "exact_10"],
        data=np.stack(
            (np.zeros(n_volumes), np.concatenate((np.ones(10), np.zeros(n_volumes - 10)))),
            axis=1,
        ),
    )
    temporal_mask = os.path.join(tmpdir, "temporal_mask.tsv")
    censoring_df.to_csv(temporal_mask, sep="\t", index=False)

    # Load atlases
    atlas_names = ["Gordon", "Glasser"]
    atlas_files = [get_atlas_nifti(atlas_name)[0] for atlas_name in atlas_names]
    atlas_labels_files = [get_atlas_nifti(atlas_name)[1] for atlas_name in atlas_names]

    # Perform the resampling and parcellation done by init_load_atlases_wf
    warped_atlases = []
    # Get transform(s) from MNI152NLin6Asym to BOLD file's space
    transforms_from_MNI152NLin6Asym = get_std2bold_xfms(bold_file)
    for atlas_file in atlas_files:
        # Using the generated transforms, apply them to get everything in the correct MNI form
        warp_atlases_to_bold_space = ApplyTransforms(
            reference_image=boldref,
            transforms=transforms_from_MNI152NLin6Asym,
            input_image=atlas_file,
            interpolation="GenericLabel",
            input_image_type=3,
            dimension=3,
        )
        warp_atlases_to_bold_space_results = warp_atlases_to_bold_space.run(cwd=tmpdir)

        warped_atlases.append(warp_atlases_to_bold_space_results.outputs.output_image)

    atlas_file = warped_atlases[0]
    atlas_labels_file = atlas_labels_files[0]
    n_parcels, n_parcels_in_atlas = 333, 333

    # Let's define the inputs and create the workflow
    with mock_config():
        config.execution.xcp_d_dir = tmpdir
        config.workflow.bandpass_filter = False
        config.workflow.min_coverage = 0.5
        config.nipype.omp_nthreads = 2
        config.execution.atlases = atlas_names
        config.workflow.output_correlations = True

        connectivity_wf = init_functional_connectivity_nifti_wf(
            mem_gb=mem_gbx,
            name="connectivity_wf",
        )
        connectivity_wf.inputs.inputnode.denoised_bold = fake_bold_file
        connectivity_wf.inputs.inputnode.temporal_mask = temporal_mask
        connectivity_wf.inputs.inputnode.name_source = bold_file
        connectivity_wf.inputs.inputnode.bold_mask = bold_mask
        connectivity_wf.inputs.inputnode.reho = fake_bold_file
        connectivity_wf.inputs.inputnode.atlases = atlas_names
        connectivity_wf.inputs.inputnode.atlas_files = warped_atlases
        connectivity_wf.inputs.inputnode.atlas_labels_files = atlas_labels_files
        connectivity_wf.base_dir = tmpdir
        connectivity_wf_res = connectivity_wf.run()

        nodes = get_nodes(connectivity_wf_res)

        # Let's find the correct workflow outputs
        assert os.path.isfile(atlas_file)
        coverage = nodes["connectivity_wf.parcellate_data"].get_output("coverage")[0]
        assert os.path.isfile(coverage)
        timeseries = nodes["connectivity_wf.parcellate_data"].get_output("timeseries")[0]
        assert os.path.isfile(timeseries)
        correlations = nodes["connectivity_wf.functional_connectivity"].get_output("correlations")[
            0
        ]
        assert os.path.isfile(correlations)

        # Read that into a df
        coverage_df = pd.read_table(coverage, index_col="Node")
        coverage_arr = coverage_df.to_numpy()
        assert coverage_arr.shape[0] == n_parcels
        correlations_arr = pd.read_table(correlations, index_col="Node").to_numpy()
        assert correlations_arr.shape == (n_parcels, n_parcels)

        # Now to get ground truth correlations
        labels_df = pd.read_table(atlas_labels_file, index_col="index")
        atlas_img, _ = _sanitize_nifti_atlas(atlas_file, labels_df)
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=["background"] + coverage_df.index.tolist(),
            smoothing_fwhm=None,
            standardize=False,
        )
        masker.fit(fake_bold_file)
        signals = masker.transform(fake_bold_file)

        # Parcels with <50% coverage should have NaNs
        assert np.array_equal(np.squeeze(coverage_arr) < 0.5, np.isnan(np.diag(correlations_arr)))

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

        # pnc data doesn't have complete coverage, so we must allow NaNs here.
        assert np.allclose(correlations_arr, calculated_correlations, atol=0.01, equal_nan=True)


def test_init_functional_connectivity_cifti_wf(ds001419_data, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    tmpdir = tmp_path_factory.mktemp("test_init_functional_connectivity_cifti_wf")

    bold_file = ds001419_data["cifti_file"]
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

    mem_gbx = _create_mem_gb(bold_file)

    # Create a fake temporal mask to satisfy the workflow
    n_volumes = bold_data.shape[1]
    censoring_df = pd.DataFrame(
        columns=["framewise_displacement", "exact_10"],
        data=np.stack(
            (np.zeros(n_volumes), np.concatenate((np.ones(10), np.zeros(n_volumes - 10)))),
            axis=1,
        ),
    )
    temporal_mask = os.path.join(tmpdir, "temporal_mask.tsv")
    censoring_df.to_csv(temporal_mask, sep="\t", index=False)

    # Load atlases
    atlas_names = ["4S1056Parcels", "4S156Parcels", "4S456Parcels", "Gordon", "Glasser"]
    atlas_files = [get_atlas_cifti(atlas_name)[0] for atlas_name in atlas_names]
    atlas_labels_files = [get_atlas_cifti(atlas_name)[1] for atlas_name in atlas_names]

    # Create the node and a tmpdir to write its results out to
    with mock_config():
        config.execution.xcp_d_dir = tmpdir
        config.workflow.bandpass_filter = False
        config.workflow.min_coverage = 0.5
        config.nipype.omp_nthreads = 2
        config.execution.atlases = atlas_names
        config.workflow.output_correlations = True

        connectivity_wf = init_functional_connectivity_cifti_wf(
            mem_gb=mem_gbx,
            exact_scans=[],
            name="connectivity_wf",
        )
        connectivity_wf.inputs.inputnode.denoised_bold = fake_bold_file
        connectivity_wf.inputs.inputnode.temporal_mask = temporal_mask
        connectivity_wf.inputs.inputnode.name_source = bold_file
        connectivity_wf.inputs.inputnode.reho = fake_bold_file
        connectivity_wf.inputs.inputnode.atlases = atlas_names
        connectivity_wf.inputs.inputnode.atlas_files = atlas_files
        connectivity_wf.inputs.inputnode.atlas_labels_files = atlas_labels_files
        connectivity_wf.base_dir = tmpdir
        connectivity_wf_res = connectivity_wf.run()

        nodes = get_nodes(connectivity_wf_res)

        # Let's find the cifti files
        pscalar = nodes["connectivity_wf.parcellate_bold_wf.parcellate_coverage"].get_output(
            "out_file"
        )[0]
        assert os.path.isfile(pscalar)
        timeseries_ciftis = nodes[
            "connectivity_wf.parcellate_bold_wf.mask_parcellated_data"
        ].get_output("out_file")[0]
        assert os.path.isfile(timeseries_ciftis)
        correlation_ciftis = nodes["connectivity_wf.correlate_bold"].get_output("out_file")[0]
        assert os.path.isfile(correlation_ciftis)

        # Let's find the tsv files
        coverage = nodes["connectivity_wf.parcellate_bold_wf.coverage_to_tsv"].get_output(
            "out_file"
        )[0]
        assert os.path.isfile(coverage)
        timeseries = nodes["connectivity_wf.parcellate_bold_wf.cifti_to_tsv"].get_output(
            "out_file"
        )[0]
        assert os.path.isfile(timeseries)
        correlations = nodes["connectivity_wf.dconn_to_tsv"].get_output("out_file")[0]
        assert os.path.isfile(correlations)

        # Let's read in the ciftis' data
        pscalar_arr = nb.load(pscalar).get_fdata().T
        assert pscalar_arr.shape == (1056, 1)
        ptseries_arr = nb.load(timeseries_ciftis).get_fdata()
        assert ptseries_arr.shape == (60, 1056)
        pconn_arr = nb.load(correlation_ciftis).get_fdata()
        assert pconn_arr.shape == (1056, 1056)

        # Read in the tsvs' data
        coverage_arr = pd.read_table(coverage).to_numpy().T
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
        assert calculated_correlations.shape == (1056, 1056)
        bad_parcels_idx = np.where(np.isnan(np.diag(calculated_correlations)))[0]
        good_parcels_idx = np.where(~np.isnan(np.diag(calculated_correlations)))[0]

        # Parcels with <50% coverage should have NaNs
        first_good_parcel_corrs = pconn_arr[good_parcels_idx[0], :]

        # The number of NaNs for a good parcel's correlations should match the number of bad
        # parcels.
        assert np.sum(np.isnan(first_good_parcel_corrs)) == bad_parcels_idx.size

        # pnc data doesn't have complete coverage, so we must allow NaNs here.
        # First set diagonals to NaNs
        np.fill_diagonal(pconn_arr, np.nan)
        np.fill_diagonal(calculated_correlations, np.nan)
        if not np.array_equal(np.isnan(pconn_arr), np.isnan(calculated_correlations)):
            mismatch_idx = np.vstack(
                np.where(np.isnan(pconn_arr) != np.isnan(calculated_correlations))
            ).T
            raise ValueError(
                f"{mismatch_idx.shape} mismatches\n\n"
                f"{mismatch_idx}\n\n"
                f"{pconn_arr[mismatch_idx[:, 0], mismatch_idx[:, 1]]}\n\n"
                f"{calculated_correlations[mismatch_idx[:, 0], mismatch_idx[:, 1]]}"
            )

        if not np.allclose(pconn_arr, calculated_correlations, atol=0.01, equal_nan=True):
            diff = pconn_arr - calculated_correlations
            raise ValueError(np.nanmax(np.abs(diff)))
