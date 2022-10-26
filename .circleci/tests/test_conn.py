"""Tests for connectivity matrix calculation."""

# Necessary imports
import fnmatch
import os

import nibabel as nb
import nilearn
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker

from xcp_d.utils.plot import _get_tr
from xcp_d.utils.utils import _t12native
from xcp_d.utils.write_save import read_ndata, write_ndata
from xcp_d.workflow.connectivity import (
    init_nifti_functional_connectivity_wf,
    init_cifti_functional_connectivity_wf
)


def test_nifti_conn(data_dir, tmp_path_factory):
    """Test the nifti workflow."""
    bold_file = os.path.join(
        data_dir,
        (
            "fmriprep/sub-colornest001/ses-1/func/"
            "sub-colornest001_ses-1_task-rest_run-1"
            "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )
    bold_mask = os.path.join(
        data_dir,
        (
            "fmriprep/sub-colornest001/ses-1/func/"
            "sub-colornest001_ses-1_task-rest_run-1_"
            "space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
        )
    )
    # Generate fake signal
    bold_data = read_ndata(bold_file, bold_mask)
    shape = bold_data.shape  # Get the shape so we can generate a matrix of
    # random numbers with the same shape

    fake_signal = np.random.randint(bold_data.min(),
                                    bold_data.max(),
                                    size=shape)

    # Let's write that out
    tempdir = tmp_path_factory.mktemp('fcon_nifti_test')
    filename = os.path.join(tempdir, "fake_signal_file.nii.gz")
    write_ndata(
        fake_signal,
        template=bold_file,
        mask=bold_mask,
        TR=_get_tr(nb.load(bold_file)),
        filename=filename,
    )
    assert os.path.isfile(filename)
    fake_bold_file = filename

    # Let's define the inputs and create the node
    mni_to_t1w = os.path.join(
        data_dir,
        (
            "fmriprep/sub-colornest001/ses-1/anat/sub-colornest001_ses-1_rec-refaced_"
            "from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
        )
    )

    fcon_ts_wf = init_nifti_functional_connectivity_wf(
        mem_gb=4,
        name="fcons_ts_wf",
        omp_nthreads=2,
    )
    fcon_ts_wf.inputs.inputnode.mni_to_t1w = mni_to_t1w,
    fcon_ts_wf.inputs.inputnode.t1w_to_native = _t12native(bold_file),
    fcon_ts_wf.inputs.inputnode.clean_bold = fake_bold_file
    fcon_ts_wf.inputs.inputnode.bold_file = bold_file
    fcon_ts_wf.inputs.inputnode.ref_file = os.path.join(
        data_dir,
        (
            "fmriprep/sub-colornest001/ses-1/func/sub-colornest001_ses-1_task-rest"
            "_run-1_space-MNI152NLin2009cAsym_boldref.nii.gz"
        )
    )
    fcon_ts_wf.base_dir = tmp_path_factory.mktemp("fcon_nifti_test_2")
    fcon_ts_wf.run()

    # Let's find the correct FCON matrix file
    for file in os.listdir(os.path.join(
            fcon_ts_wf.base_dir, "fcons_ts_wf/nifti_connect/mapflow/_nifti_connect3")):
        if fnmatch.fnmatch(file, "*matrix*"):
            out_file = file
    out_file = os.path.join(fcon_ts_wf.base_dir,
                            "fcons_ts_wf/nifti_connect/mapflow/_nifti_connect3", out_file)

    # Read that into a df
    df = pd.read_table(out_file, header=None)
    # ... and then convert to an array
    xcp_array = np.array(df)

    # Now let's get the ground truth. First, we should locate the atlas
    for file in os.listdir(
        os.path.join(
            fcon_ts_wf.base_dir, "fcons_ts_wf/atlas_mni_to_native/mapflow/_atlas_mni_to_native3")
    ):
        if fnmatch.fnmatch(file, "*.nii.gz*"):
            atlas = file
    atlas = os.path.join(fcon_ts_wf.base_dir,
                         "fcons_ts_wf/atlas_mni_to_native/mapflow/_atlas_mni_to_native3",
                         atlas)
    atlas = nilearn.image.load_img(atlas)

    # Masking img
    masker = NiftiLabelsMasker(atlas, standardize=False)
    # Fitting mask
    masker.fit(fake_bold_file)
    signals = masker.transform(fake_bold_file)

    # The "ground truth" matrix
    ground_truth = np.corrcoef(signals.T)
    assert np.allclose(xcp_array, ground_truth, atol=0.01)


def test_cifti_conn(data_dir, tmp_path_factory):
    """Test the cifti workflow - only correlation, not parcellation."""
    # Define bold file
    boldfile = os.path.join(
        data_dir,
        (
            "fmriprep/sub-colornest001/ses-1/func/sub-colornest001_ses-1_task-rest_run-2_space-"
            "fsLR_den-91k_bold.dtseries.nii"
        )
    )
    # Generate fake signal
    bold_data = read_ndata(boldfile)
    shape = bold_data.shape  # get the shape so we can generate a
    # matrix of random numbers with the same shape
    fake_signal = np.random.randint(bold_data.min(),
                                    bold_data.max(),
                                    size=shape)
    # Let's write that out
    tmpdir = tmp_path_factory.mktemp("fcon_cifti_test")
    filename = os.path.join(tmpdir, "fake_signal_file.dtseries.nii")
    write_ndata(
        fake_signal,
        template=boldfile,
        TR=_get_tr(nb.load(boldfile)),
        filename=filename
    )
    assert os.path.isfile(filename)
    fake_bold_file = filename
    # Create the node and a tempdir to write its results out to
    tmpdir = tmp_path_factory.mktemp("fcon_cifti_test_2")
    cifti_conts_wf = init_cifti_functional_connectivity_wf(
        mem_gb=4, name="cifti_ts_con_wf", omp_nthreads=2
    )
    cifti_conts_wf.base_dir = tmpdir
    # Run the node
    cifti_conts_wf.inputs.inputnode.clean_bold = fake_bold_file
    cifti_conts_wf.run()
    # Let's find the correct parcellated file
    for file in os.listdir(os.path.join(
            cifti_conts_wf.base_dir, "cifti_ts_con_wf/parcellate_data/mapflow/_parcellate_data3")):
        if fnmatch.fnmatch(file, "*dtseries*"):
            out_file = file
    out_file = os.path.join(cifti_conts_wf.base_dir,
                            "cifti_ts_con_wf/parcellate_data/mapflow/_parcellate_data3", out_file)
    # Let's read out the parcellated time series and get its corr coeff
    data = read_ndata(out_file)
    ground_truth = np.corrcoef(data)
    # Let's find the conn matt generated by XCP
    for file in os.listdir(os.path.join(
            cifti_conts_wf.base_dir, "cifti_ts_con_wf/correlate_data/mapflow/_correlate_data3")):
        if fnmatch.fnmatch(file, "*matrix*"):
            out_file = file
    out_file = os.path.join(cifti_conts_wf.base_dir,
                            "cifti_ts_con_wf/correlate_data/mapflow/_correlate_data3", out_file)

    # Read it out
    data = read_ndata(out_file)

    # Do the two match up?
    assert np.allclose(data, ground_truth, atol=0.01)
