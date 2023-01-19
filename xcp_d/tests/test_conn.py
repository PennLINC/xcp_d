"""Tests for connectivity matrix calculation."""
import os

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker

from xcp_d.utils.bids import _get_tr
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflow.connectivity import (
    init_cifti_functional_connectivity_wf,
    init_nifti_functional_connectivity_wf,
)


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
        "connectivity_wf/nifti_connect/mapflow/_nifti_connect3/fake_signal_filetime_series.tsv",
    )
    assert os.path.isfile(timeseries_file)

    # Let's find the correct correlation matrix file
    corr_mat_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/nifti_connect/mapflow/_nifti_connect3/fake_signal_filefcon_matrix.tsv",
    )
    assert os.path.isfile(corr_mat_file)

    # Read that into a df
    df = pd.read_table(corr_mat_file, header=None)
    xcp_array = df.to_numpy()
    assert xcp_array.shape == (400, 400)

    # Now let's get the ground truth. First, we should locate the atlas
    atlas_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/warp_atlases_to_bold_space/mapflow/_warp_atlases_to_bold_space3",
        "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm_trans.nii.gz",
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
    assert ground_truth.shape == (400, 400)

    # Parcels with <50% coverage should have NaNs
    # We know that 10 of the nodes in the 400-parcel Schaefer are flagged
    assert np.sum(np.isnan(np.diag(xcp_array))) == 10

    # If we replace the bad parcels' results in the "ground truth" matrix with NaNs,
    # the resulting matrix should match the workflow-generated one.
    bad_parcel_idx = np.where(np.isnan(np.diag(xcp_array)))[0]
    ground_truth[bad_parcel_idx, :] = np.nan
    ground_truth[:, bad_parcel_idx] = np.nan

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    assert np.allclose(xcp_array, ground_truth, atol=0.01, equal_nan=True)


def test_cifti_conn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    tmpdir = tmp_path_factory.mktemp("test_cifti_conn")

    boldfile = fmriprep_with_freesurfer_data["cifti_file"]

    # Generate fake signal
    bold_data = read_ndata(boldfile)
    fake_signal = np.random.randint(1, 500, size=bold_data.shape).astype(np.float32)
    # Make half the vertices all zeros
    fake_signal[:5000, :] = 0
    fake_bold_file = os.path.join(tmpdir, "fake_signal_file.dtseries.nii")
    write_ndata(
        fake_signal,
        template=boldfile,
        TR=_get_tr(nb.load(boldfile)),
        filename=fake_bold_file,
    )
    assert os.path.isfile(fake_bold_file)

    # Create the node and a tmpdir to write its results out to
    connectivity_wf = init_cifti_functional_connectivity_wf(
        mem_gb=4,
        name="connectivity_wf",
        omp_nthreads=2,
    )
    connectivity_wf.inputs.inputnode.clean_bold = fake_bold_file
    connectivity_wf.base_dir = tmpdir
    connectivity_wf.run()

    # Let's find the correct parcellated file
    parc_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/parcellate_data/mapflow/_parcellate_data3",
        "parcellated_modified_data.dtseries.ptseries.nii",
    )
    assert os.path.isfile(parc_file)

    # Let's read out the parcellated time series and get its corr coeff
    data = nb.load(parc_file).get_fdata().T
    ground_truth = np.corrcoef(data)
    assert ground_truth.shape == (400, 400)

    # Let's find the correct correlation matrix file
    files = os.listdir(
        os.path.join(
            connectivity_wf.base_dir, "connectivity_wf/correlate_data/mapflow/_correlate_data3"
        )
    )
    raise Exception(files)

    pconn_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/correlate_data/mapflow/_correlate_data3",
        "",
    )
    assert os.path.isfile(pconn_file)

    # Read it out
    xcp_array = nb.load(pconn_file).get_fdata().T
    assert xcp_array.shape == (400, 400)

    # Parcels with <50% coverage should have NaNs
    # We know that 10 of the nodes in the 400-parcel Schaefer are flagged
    assert np.sum(np.isnan(np.diag(xcp_array))) == 10

    # If we replace the bad parcels' results in the "ground truth" matrix with NaNs,
    # the resulting matrix should match the workflow-generated one.
    bad_parcel_idx = np.where(np.isnan(np.diag(xcp_array)))[0]
    ground_truth[bad_parcel_idx, :] = np.nan
    ground_truth[:, bad_parcel_idx] = np.nan

    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    assert np.allclose(xcp_array, ground_truth, atol=0.01, equal_nan=True)
