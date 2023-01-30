"""Tests for connectivity matrix calculation."""

# Necessary imports
import fnmatch
import glob
import os

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
    fcon_ts_wf = init_nifti_functional_connectivity_wf(
        output_dir=tempdir,
        mem_gb=4,
        name="fcons_ts_wf",
        omp_nthreads=2,
    )
    fcon_ts_wf.inputs.inputnode.template_to_t1w = template_to_t1w_xform
    fcon_ts_wf.inputs.inputnode.t1w_to_native = t1w_to_native_xform
    fcon_ts_wf.inputs.inputnode.clean_bold = fake_bold_file
    fcon_ts_wf.inputs.inputnode.bold_file = bold_file
    fcon_ts_wf.inputs.inputnode.bold_mask = bold_mask
    fcon_ts_wf.inputs.inputnode.ref_file = boldref
    fcon_ts_wf.base_dir = tmp_path_factory.mktemp("fcon_nifti_test_2")
    fcon_ts_wf.run()

    # Let's find the correct FCON matrix file
    files = glob.glob(
        os.path.join(
            fcon_ts_wf.base_dir,
            "fcons_ts_wf/nifti_connect/mapflow/_nifti_connect3/*",
        ),
    )
    for file_ in files:
        if fnmatch.fnmatch(file_, "*matrix*"):
            corr_mat_file = file_

    # Read that into a df
    df = pd.read_table(corr_mat_file, header=None)
    xcp_array = df.to_numpy()

    # Now let's get the ground truth. First, we should locate the atlas
    files = glob.glob(
        os.path.join(
            fcon_ts_wf.base_dir,
            "fcons_ts_wf/warp_atlases_to_bold_space/mapflow/_warp_atlases_to_bold_space3/*",
        )
    )
    for file_ in files:
        if fnmatch.fnmatch(file_, "*.nii.gz*"):
            atlas = file_

    atlas = nb.load(atlas)

    # Masking img
    masker = NiftiLabelsMasker(atlas, mask_img=bold_mask, smoothing_fwhm=None, standardize=False)
    # Fitting mask
    masker.fit(fake_bold_file)
    signals = masker.transform(fake_bold_file)

    # The "ground truth" matrix
    ground_truth = np.corrcoef(signals.T)
    # ds001419 data doesn't have complete coverage, so we must allow NaNs here.
    assert np.allclose(xcp_array, ground_truth, atol=0.01, equal_nan=True)


def test_cifti_conn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    # Define bold file
    boldfile = fmriprep_with_freesurfer_data["cifti_file"]

    # Generate fake signal
    bold_data = read_ndata(boldfile)
    # get the shape so we can generate a matrix of random numbers with the same shape
    shape = bold_data.shape
    fake_signal = np.random.randint(bold_data.min(), bold_data.max(), size=shape)
    # Let's write that out
    tmpdir = tmp_path_factory.mktemp("fcon_cifti_test")
    filename = os.path.join(tmpdir, "fake_signal_file.dtseries.nii")
    write_ndata(fake_signal, template=boldfile, TR=_get_tr(nb.load(boldfile)), filename=filename)
    assert os.path.isfile(filename)
    fake_bold_file = filename
    # Create the node and a tempdir to write its results out to
    tmpdir = tmp_path_factory.mktemp("fcon_cifti_test_2")
    cifti_conts_wf = init_cifti_functional_connectivity_wf(
        output_dir=tmpdir,
        mem_gb=4,
        name="cifti_ts_con_wf",
        omp_nthreads=2,
    )
    cifti_conts_wf.base_dir = tmpdir
    # Run the node
    cifti_conts_wf.inputs.inputnode.clean_bold = fake_bold_file
    cifti_conts_wf.inputs.inputnode.bold_file = boldfile
    cifti_conts_wf.run()
    # Let's find the correct parcellated file
    for file_ in os.listdir(
        os.path.join(
            cifti_conts_wf.base_dir, "cifti_ts_con_wf/parcellate_data/mapflow/_parcellate_data3"
        )
    ):
        if fnmatch.fnmatch(file_, "*dtseries*"):
            out_file = file_
    out_file = os.path.join(
        cifti_conts_wf.base_dir,
        "cifti_ts_con_wf/parcellate_data/mapflow/_parcellate_data3",
        out_file,
    )
    # Let's read out the parcellated time series and get its corr coeff
    data = nb.load(out_file).get_fdata().T
    ground_truth = np.corrcoef(data)
    # Let's find the conn matt generated by XCP
    for file_ in os.listdir(
        os.path.join(
            cifti_conts_wf.base_dir, "cifti_ts_con_wf/correlate_data/mapflow/_correlate_data3"
        )
    ):
        if fnmatch.fnmatch(file_, "*matrix*"):
            out_file = file_
    out_file = os.path.join(
        cifti_conts_wf.base_dir,
        "cifti_ts_con_wf/correlate_data/mapflow/_correlate_data3",
        out_file,
    )

    # Read it out
    data = nb.load(out_file).get_fdata().T

    # Do the two match up?
    assert np.allclose(data, ground_truth, atol=0.01)
