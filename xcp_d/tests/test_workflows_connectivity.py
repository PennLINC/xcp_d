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
    timeseries_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/nifti_connect/mapflow/_nifti_connect9/fake_signal_filetime_series.tsv",
    )
    assert os.path.isfile(timeseries_file)

    # Let's find the correct correlation matrix file
    correlation_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/nifti_connect/mapflow/_nifti_connect9/fake_signal_filefcon_matrix.tsv",
    )
    assert os.path.isfile(correlation_file)

    coverage_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/nifti_connect/mapflow/_nifti_connect9",
        "fake_signal_fileparcel_coverage_file.tsv",
    )
    assert os.path.isfile(coverage_file)

    # Read that into a df
    timeseries_df = pd.read_table(timeseries_file, header=None)
    timeseries_arr = timeseries_df.to_numpy()
    assert timeseries_arr.shape == (60, 1000)

    # Read that into a df
    corr_df = pd.read_table(correlation_file, header=None)
    corr_arr = corr_df.to_numpy()
    assert corr_arr.shape == (1000, 1000)

    # Read that into a df
    coverage_df = pd.read_table(coverage_file, header=None)
    coverage_arr = coverage_df.to_numpy()
    assert coverage_arr.shape == (1000,)

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
    ground_truth = np.corrcoef(signals.T)
    assert ground_truth.shape == (1000, 1000)

    # Parcels with <50% coverage should have NaNs
    bad_parcel_idx = np.where(coverage_arr < 0.5)[0]
    assert np.array_equal(np.where(np.isnan(np.diag(corr_arr)))[0], bad_parcel_idx)

    # If we replace the bad parcels' results in the "ground truth" matrix with NaNs,
    # the resulting matrix should match the workflow-generated one.
    ground_truth[bad_parcel_idx, :] = np.nan
    ground_truth[:, bad_parcel_idx] = np.nan

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    assert np.allclose(corr_arr, ground_truth, atol=0.01, equal_nan=True)


def test_cifti_conn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    tmpdir = tmp_path_factory.mktemp("test_cifti_conn")

    bold_file = fmriprep_with_freesurfer_data["cifti_file"]
    TR = _get_tr(nb.load(bold_file))

    # Generate fake signal
    bold_data = read_ndata(bold_file)
    fake_signal = np.random.randint(1, 500, size=bold_data.shape).astype(np.float32)
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
        TR=TR,
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
    sanitize_dir = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/sanitize_parcellation_results/mapflow/_sanitize_parcellation_results9",
    )
    timeseries_file = os.path.join(
        sanitize_dir,
        "parcellated_prepared_timeseries.dtseries.ptseries.nii",
    )
    assert os.path.isfile(timeseries_file), os.listdir(sanitize_dir)

    correlation_file = os.path.join(
        sanitize_dir,
        "fake_signal_filefcon_matrix.pconn.nii",
    )
    assert os.path.isfile(correlation_file), os.listdir(sanitize_dir)

    coverage_file = os.path.join(
        sanitize_dir,
        "fake_signal_fileparcel_coverage_file.pscalar.nii",
    )
    assert os.path.isfile(coverage_file), os.listdir(sanitize_dir)

    # Let's read out the parcellated time series and get its corr coeff
    timeseries_arr = nb.load(timeseries_file).get_fdata().T
    assert timeseries_arr.shape == (60, 1000)

    corr_arr = nb.load(correlation_file).get_fdata().T
    assert corr_arr.shape == (1000, 1000)

    coverage_arr = nb.load(coverage_file).get_fdata().T
    assert coverage_arr.shape == (1000,)

    # Calculate correlations
    ground_truth = np.corrcoef(timeseries_arr)
    assert ground_truth.shape == (1000, 1000)

    # Parcels with <50% coverage should have NaNs
    bad_parcel_idx = np.where(coverage_arr < 0.5)[0]
    assert np.array_equal(np.where(np.isnan(np.diag(corr_arr)))[0], bad_parcel_idx)

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    if not np.array_equal(np.isnan(corr_arr), np.isnan(ground_truth)):
        mismatch_idx = np.vstack(np.where(np.isnan(corr_arr) != np.isnan(ground_truth))).T
        raise ValueError(f"{mismatch_idx}\n\n{np.where(np.isnan(corr_arr))}")

    if not np.allclose(corr_arr, ground_truth, atol=0.01, equal_nan=True):
        diff = corr_arr - ground_truth
        raise ValueError(np.nanmax(np.abs(diff)))
