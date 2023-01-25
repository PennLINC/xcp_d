"""Tests for the xcp_d.workflow.connectivity module."""
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
    bold_file = fmriprep_with_freesurfer_data["nifti_file"]
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]
    template_to_t1w_xform = fmriprep_with_freesurfer_data["template_to_t1w_xform"]
    boldref = fmriprep_with_freesurfer_data["boldref"]
    t1w_to_native_xform = fmriprep_with_freesurfer_data["t1w_to_native_xform"]

    tempdir = tmp_path_factory.mktemp("test_nifti_conn")

    # Generate fake signal
    bold_data = read_ndata(bold_file, bold_mask)
    fake_signal = np.random.randint(bold_data.min(), bold_data.max(), size=bold_data.shape)

    # Let's write that out
    fake_bold_file = os.path.join(tempdir, "fake_signal_file.nii.gz")
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
        output_dir=tempdir,
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
    connectivity_wf.base_dir = tmp_path_factory.mktemp("fcon_nifti_test_2")
    connectivity_wf.run()

    # Let's find the correct FCON matrix file
    corr_mat_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/correlate_timeseries/mapflow/_correlate_timeseries3/correlations.tsv",
    )
    assert os.path.isfile(corr_mat_file)

    # Read that into a df
    df = pd.read_table(corr_mat_file, index_col="Node")
    xcp_array = df.to_numpy()
    assert xcp_array.shape == (400, 400)

    # Now let's get the ground truth. First, we should locate the atlas
    atlas_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/warp_atlases_to_bold_space/mapflow/_warp_atlases_to_bold_space3",
        "Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm_trans.nii.gz"
    )
    assert os.path.isfile(atlas_file)

    # Masking img
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        mask_img=bold_mask,
        smoothing_fwhm=None,
        standardize=False,
        resampling_target=None,  # they should be in the same space/resolution already
    )
    signals = masker.fit_transform(fake_bold_file)

    # The "ground truth" matrix
    ground_truth = np.corrcoef(signals.T)
    assert ground_truth.shape == (400, 400)

    # ds001491 data doesn't have complete coverage, so we must allow NaNs here.
    assert np.allclose(xcp_array, ground_truth, atol=0.01, equal_nan=True)


def test_cifti_conn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    tmpdir = tmp_path_factory.mktemp("fcon_cifti_test")

    # Define bold file
    boldfile = fmriprep_with_freesurfer_data["cifti_file"]

    # Generate fake signal
    bold_data = read_ndata(boldfile)
    shape = bold_data.shape
    fake_signal = np.random.randint(bold_data.min(), bold_data.max(), size=shape)
    fake_bold_file = os.path.join(tmpdir, "fake_signal_file.dtseries.nii")
    write_ndata(
        fake_signal,
        template=boldfile,
        TR=_get_tr(nb.load(boldfile)),
        filename=fake_bold_file,
    )
    assert os.path.isfile(fake_bold_file)

    # Run the cifti connectivity workflow
    connectivity_wf = init_cifti_functional_connectivity_wf(
        output_dir=tmpdir,
        mem_gb=4,
        name="connectivity_wf",
        omp_nthreads=2,
    )
    connectivity_wf.base_dir = tmpdir
    connectivity_wf.inputs.inputnode.bold_file = boldfile
    connectivity_wf.inputs.inputnode.clean_bold = fake_bold_file
    connectivity_wf.run()

    # Let's find the correct parcellated file
    parc_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/parcellate_data/mapflow/_parcellate_data3",
        "parcellated_modified_data.dtseries.ptseries.nii",
    )
    assert os.path.isfile(parc_file)

    # Let's read out the parcellated time series and get its corr coeff
    ground_truth_data = read_ndata(parc_file)
    ground_truth = np.corrcoef(ground_truth_data)

    # Let's find the correlation matrix generated by XCP
    corr_mat_file = os.path.join(
        connectivity_wf.base_dir,
        "connectivity_wf/correlate_timeseries/mapflow/_correlate_timeseries3/correlations.tsv",
    )
    assert os.path.isfile(corr_mat_file)

    # Read that into a df
    df = pd.read_table(corr_mat_file, index_col="Node")
    data = df.to_numpy()

    # Do the two match up?
    assert np.allclose(data, ground_truth, atol=0.01)
