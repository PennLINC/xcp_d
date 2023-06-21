"""Tests for connectivity matrix calculation."""
import os
import sys

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.workbench import CiftiCreateDenseFromTemplate, CiftiParcellate
from xcp_d.tests.utils import get_nodes
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_nifti
from xcp_d.utils.bids import _get_tr
from xcp_d.utils.utils import get_std2bold_xfms
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflows.connectivity import (
    init_functional_connectivity_cifti_wf,
    init_functional_connectivity_nifti_wf,
    init_load_atlases_wf,
)

np.set_printoptions(threshold=sys.maxsize)


def test_init_load_atlases_wf_nifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test init_load_atlases_wf with a nifti input."""
    tmpdir = tmp_path_factory.mktemp("test_init_functional_connectivity_nifti_wf")

    bold_file = fmriprep_with_freesurfer_data["nifti_file"]

    load_atlases_wf = init_load_atlases_wf(
        output_dir=tmpdir,
        cifti=False,
        mem_gb=1,
        omp_nthreads=1,
        name="load_atlases_wf",
    )
    load_atlases_wf.inputs.inputnode.name_source = bold_file
    load_atlases_wf.inputs.inputnode.bold_file = bold_file
    load_atlases_wf.base_dir = tmpdir
    load_atlases_wf_res = load_atlases_wf.run()
    nodes = get_nodes(load_atlases_wf_res)
    atlas_names = nodes["load_atlases_wf.warp_atlases_to_bold_space"].get_output("output_image")
    assert len(atlas_names) == 14


def test_init_load_atlases_wf_cifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test init_load_atlases_wf with a cifti input."""
    tmpdir = tmp_path_factory.mktemp("test_init_load_atlases_wf_cifti")

    bold_file = fmriprep_with_freesurfer_data["cifti_file"]

    load_atlases_wf = init_load_atlases_wf(
        output_dir=tmpdir,
        cifti=True,
        mem_gb=1,
        omp_nthreads=1,
        name="load_atlases_wf",
    )
    load_atlases_wf.inputs.inputnode.name_source = bold_file
    load_atlases_wf.inputs.inputnode.bold_file = bold_file
    load_atlases_wf.base_dir = tmpdir
    load_atlases_wf_res = load_atlases_wf.run()
    nodes = get_nodes(load_atlases_wf_res)
    atlas_names = nodes["load_atlases_wf.cast_atlas_to_int16"].get_output("out_file")
    assert len(atlas_names) == 14


def test_init_functional_connectivity_nifti_wf(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the nifti workflow."""
    tmpdir = tmp_path_factory.mktemp("test_init_functional_connectivity_nifti_wf")

    bold_file = fmriprep_with_freesurfer_data["nifti_file"]
    boldref = fmriprep_with_freesurfer_data["boldref"]
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]

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

    atlas_names = ["Schaefer1017", "Schaefer217", "Schaefer417", "Gordon", "Glasser"]
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

    # Let's define the inputs and create the workflow
    connectivity_wf = init_functional_connectivity_nifti_wf(
        output_dir=tmpdir,
        min_coverage=0.5,
        alff_available=False,
        mem_gb=4,
        name="connectivity_wf",
    )
    connectivity_wf.inputs.inputnode.denoised_bold = fake_bold_file
    connectivity_wf.inputs.inputnode.name_source = bold_file
    connectivity_wf.inputs.inputnode.bold_mask = bold_mask
    connectivity_wf.inputs.inputnode.reho = fake_bold_file
    connectivity_wf.inputs.inputnode.atlas_names = atlas_names
    connectivity_wf.inputs.inputnode.atlas_files = warped_atlases
    connectivity_wf.inputs.inputnode.atlas_labels_files = atlas_labels_files
    connectivity_wf.base_dir = tmpdir
    connectivity_wf_res = connectivity_wf.run()
    nodes = get_nodes(connectivity_wf_res)

    n_parcels, n_parcels_in_atlas = 1000, 1000

    # Let's find the correct workflow outputs
    atlas_file = warped_atlases[0]
    assert os.path.isfile(atlas_file)
    coverage = nodes["connectivity_wf.functional_connectivity"].get_output("coverage")[0]
    assert os.path.isfile(coverage)
    timeseries = nodes["connectivity_wf.functional_connectivity"].get_output("timeseries")[0]
    assert os.path.isfile(timeseries)
    correlations = nodes["connectivity_wf.functional_connectivity"].get_output("correlations")[0]
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
    calculated_correlations_r = np.corrcoef(signals.T)
    calculated_correlations_z = np.arctanh(calculated_correlations_r)
    np.fill_diagonal(calculated_correlations_z, 0)
    assert calculated_correlations_z.shape == (n_parcels_in_atlas, n_parcels_in_atlas)

    # If we replace the bad parcels' results in the "ground truth" matrix with NaNs,
    # the resulting matrix should match the workflow-generated one.
    bad_parcel_idx = np.where(np.isnan(np.diag(correlations_arr)))[0]
    assert bad_parcel_idx.size == n_parcels_in_atlas - n_partial_parcels
    calculated_correlations_z[bad_parcel_idx, :] = np.nan
    calculated_correlations_z[:, bad_parcel_idx] = np.nan

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    assert np.allclose(correlations_arr, calculated_correlations_z, atol=0.01, equal_nan=True)


def test_init_functional_connectivity_cifti_wf(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    tmpdir = tmp_path_factory.mktemp("test_init_functional_connectivity_cifti_wf")

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

    atlas_names = ["Schaefer1017", "Schaefer217", "Schaefer417", "Gordon", "Glasser"]
    atlas_files = [get_atlas_cifti(atlas_name)[0] for atlas_name in atlas_names]
    atlas_labels_files = [get_atlas_cifti(atlas_name)[1] for atlas_name in atlas_names]

    # Perform the resampling and parcellation done by init_load_atlases_wf
    parcellated_atlases = []
    for i_file, atlas_file in enumerate(atlas_files):
        resample_atlas_to_data = CiftiCreateDenseFromTemplate(
            template_cifti=bold_file,
            label=atlas_file,
        )
        resample_results = resample_atlas_to_data.run(cwd=tmpdir)

        parcellate_atlas = CiftiParcellate(
            direction="COLUMN",
            only_numeric=True,
            out_file=f"parcellated_atlas_{i_file}.pscalar.nii",
            atlas_label=atlas_file,
            in_file=resample_results.outputs.cifti_out,
        )
        parcellate_atlas_results = parcellate_atlas.run(cwd=tmpdir)

        parcellated_atlases.append(parcellate_atlas_results.outputs.out_file)

    # Create the node and a tmpdir to write its results out to
    connectivity_wf = init_functional_connectivity_cifti_wf(
        output_dir=tmpdir,
        min_coverage=0.5,
        alff_available=False,
        mem_gb=4,
        omp_nthreads=2,
        name="connectivity_wf",
    )
    connectivity_wf.inputs.inputnode.denoised_bold = fake_bold_file
    connectivity_wf.inputs.inputnode.name_source = bold_file
    connectivity_wf.inputs.inputnode.reho = fake_bold_file
    connectivity_wf.inputs.inputnode.atlas_names = atlas_names
    connectivity_wf.inputs.inputnode.atlas_files = atlas_files
    connectivity_wf.inputs.inputnode.atlas_labels_files = atlas_labels_files
    connectivity_wf.inputs.inputnode.parcellated_atlas_files = parcellated_atlases
    connectivity_wf.base_dir = tmpdir
    connectivity_wf_res = connectivity_wf.run()
    nodes = get_nodes(connectivity_wf_res)

    # Let's find the cifti files
    pscalar = nodes["connectivity_wf.functional_connectivity"].get_output("coverage_ciftis")[0]
    assert os.path.isfile(pscalar)
    timeseries_ciftis = nodes["connectivity_wf.functional_connectivity"].get_output(
        "timeseries_ciftis"
    )[0]
    assert os.path.isfile(timeseries_ciftis)
    correlation_ciftis = nodes["connectivity_wf.functional_connectivity"].get_output(
        "correlation_ciftis"
    )[0]
    assert os.path.isfile(correlation_ciftis)

    # Let's find the tsv files
    coverage = nodes["connectivity_wf.functional_connectivity"].get_output("coverage")[0]
    assert os.path.isfile(coverage)
    timeseries = nodes["connectivity_wf.functional_connectivity"].get_output("timeseries")[0]
    assert os.path.isfile(timeseries)
    correlations = nodes["connectivity_wf.functional_connectivity"].get_output("correlations")[0]
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
    calculated_correlations_r = np.corrcoef(ptseries_arr.T)
    calculated_correlations_z = np.arctanh(calculated_correlations_r)
    np.fill_diagonal(calculated_correlations_z, 0)
    
    assert calculated_correlations_z.shape == (1000, 1000)
    bad_parcels_idx = np.where(np.isnan(np.diag(calculated_correlations_z)))[0]
    good_parcels_idx = np.where(~np.isnan(np.diag(calculated_correlations_z)))[0]

    # Parcels with <50% coverage should have NaNs
    first_good_parcel_corrs = pconn_arr[good_parcels_idx[0], :]

    # The number of NaNs for a good parcel's correlations should match the number of bad parcels.
    assert np.sum(np.isnan(first_good_parcel_corrs)) == bad_parcels_idx.size

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    if not np.array_equal(np.isnan(pconn_arr), np.isnan(calculated_correlations_z)):
        mismatch_idx = np.vstack(
            np.where(np.isnan(pconn_arr) != np.isnan(calculated_correlations_z))
        ).T
        raise ValueError(f"{mismatch_idx}\n\n{np.where(np.isnan(pconn_arr))}")

    if not np.allclose(pconn_arr, calculated_correlations_z, atol=0.01, equal_nan=True):
        diff = pconn_arr - calculated_correlations_z
        raise ValueError(np.nanmax(np.abs(diff)))
