"""Tests for framewise displacement calculation."""
import os

import nibabel as nb
import numpy as np
import pandas as pd

from xcp_d.interfaces import censoring


def test_generate_confounds(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Check results."""
    tmpdir = tmp_path_factory.mktemp("test_generate_confounds")
    in_file = fmriprep_with_freesurfer_data["nifti_file"]
    confounds_file = fmriprep_with_freesurfer_data["confounds_file"]
    confounds_json = fmriprep_with_freesurfer_data["confounds_json"]

    df = pd.read_table(confounds_file)

    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]
    df.loc[4:6, "trans_y"] = [7, 8, 9]
    df.loc[7:9, "trans_z"] = [12, 8, 9]

    # Rename with same convention as initial confounds tsv
    confounds_tsv = os.path.join(tmpdir, "edited_confounds.tsv")
    df.to_csv(confounds_tsv, sep="\t", index=False, header=True)

    # Run workflow
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        params="24P",
        TR=0.8,
        fd_thresh=0.3,
        head_radius=50,
        fmriprep_confounds_file=confounds_tsv,
        fmriprep_confounds_json=confounds_json,
        custom_confounds_file=None,
        motion_filter_type=None,
        motion_filter_order=4,
        band_stop_min=0,
        band_stop_max=0,
    )
    results = interface.run(cwd=tmpdir)

    assert os.path.isfile(results.outputs.filtered_confounds_file)
    assert os.path.isfile(results.outputs.confounds_file)
    assert os.path.isfile(results.outputs.motion_file)
    assert os.path.isfile(results.outputs.temporal_mask)


def test_censor(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test Censor interface."""
    tmpdir = tmp_path_factory.mktemp("test_generate_confounds")
    nifti_file = fmriprep_with_freesurfer_data["nifti_file"]
    cifti_file = fmriprep_with_freesurfer_data["cifti_file"]
    in_img = nb.load(nifti_file)
    n_volumes = in_img.shape[3]
    censoring_df = pd.DataFrame(columns=["framewise_displacement"], data=np.zeros(n_volumes))
    temporal_mask = os.path.join(tmpdir, "temporal_mask.tsv")
    censoring_df.to_csv(temporal_mask, sep="\t", index=False)

    # Test with a NIfTI file, with no censored volumes
    interface = censoring.Censor(
        in_file=nifti_file,
        temporal_mask=temporal_mask,
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.censored_denoised_bold
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[3] == n_volumes

    # Test with a CIFTI file, with no censored volumes
    interface = censoring.Censor(
        in_file=cifti_file,
        temporal_mask=temporal_mask,
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.censored_denoised_bold
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[3] == n_volumes

    # Test with a NIfTI file, with some censored volumes
    n_censored_volumes = 10
    n_retained_volumes = n_volumes - n_censored_volumes
    censoring_df.loc[:n_censored_volumes, "framewise_displacement"] = 1
    censoring_df.to_csv(temporal_mask, sep="\t", index=False)
    interface = censoring.Censor(
        in_file=nifti_file,
        temporal_mask=temporal_mask,
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.censored_denoised_bold
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[3] == n_retained_volumes

    # Test with a CIFTI file, with some censored volumes
    interface = censoring.Censor(
        in_file=cifti_file,
        temporal_mask=temporal_mask,
    )
    results = interface.run(cwd=tmpdir)
    out_file = results.outputs.censored_denoised_bold
    assert os.path.isfile(out_file)
    out_img = nb.load(out_file)
    assert out_img.shape[3] == n_retained_volumes
