"""Tests for the xcp_d.utils.concatenation module."""
import os

import nibabel as nb
import numpy as np
import pandas as pd

from xcp_d.utils import concatenation


def test_concatenate_tsvs(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.utils.concatenation.concatenate_tsvs."""
    tmpdir = tmp_path_factory.mktemp("test_concatenate_tsvs")

    n_repeats = 3

    # First, concatenate TSVs with headers
    tsv_file_with_header = fmriprep_with_freesurfer_data["confounds_file"]
    concat_tsv_file_with_header = os.path.join(tmpdir, "concat_with_header.tsv")
    concatenation.concatenate_tsvs(
        [tsv_file_with_header] * n_repeats,
        out_file=concat_tsv_file_with_header,
    )
    assert os.path.isfile(concat_tsv_file_with_header)
    tsv_df = pd.read_table(tsv_file_with_header)
    concat_tsv_df = pd.read_table(concat_tsv_file_with_header)
    assert concat_tsv_df.shape[0] == tsv_df.shape[0] * n_repeats
    assert concat_tsv_df.shape[1] == tsv_df.shape[1]

    # Now, concatenate TSVs without headers
    tsv_file_without_header = os.path.join(tmpdir, "without_header.tsv")
    data = pd.read_table(tsv_file_with_header).to_numpy()
    np.savetxt(tsv_file_without_header, data, fmt="%.5f", delimiter="\t")

    concat_tsv_file_without_header = os.path.join(tmpdir, "concat_without_header.tsv")
    concatenation.concatenate_tsvs(
        [tsv_file_without_header] * n_repeats,
        out_file=concat_tsv_file_without_header,
    )
    assert os.path.isfile(concat_tsv_file_without_header)
    tsv_arr = np.loadtxt(tsv_file_without_header)
    concat_tsv_arr = np.loadtxt(concat_tsv_file_without_header)
    assert concat_tsv_arr.shape[0] == tsv_arr.shape[0] * n_repeats
    assert concat_tsv_arr.shape[1] == tsv_arr.shape[1]


def test_concatenate_niimgs(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.utils.concatenation.concatenate_niimgs.

    We don't have non-dtseries CIFTIs to test, so this test is a little limited.
    """
    tmpdir = tmp_path_factory.mktemp("test_concatenate_niimgs")

    n_repeats = 3

    # First, concatenate niftis
    nifti_file = fmriprep_with_freesurfer_data["nifti_file"]
    concat_nifti_file = os.path.join(tmpdir, "concat_nifti.nii.gz")
    concatenation.concatenate_niimgs(
        [nifti_file] * n_repeats,
        out_file=concat_nifti_file,
    )
    assert os.path.isfile(concat_nifti_file)
    nifti_img = nb.load(nifti_file)
    concat_nifti_img = nb.load(concat_nifti_file)
    assert concat_nifti_img.shape[:3] == nifti_img.shape[:3]
    assert concat_nifti_img.shape[3] == nifti_img.shape[3] * n_repeats

    # Now, concatenate dtseries ciftis
    cifti_file = fmriprep_with_freesurfer_data["cifti_file"]
    concat_cifti_file = os.path.join(tmpdir, "concat_cifti.dtseries.nii")
    concatenation.concatenate_niimgs(
        [cifti_file] * n_repeats,
        out_file=concat_cifti_file,
    )
    assert os.path.isfile(concat_cifti_file)
    cifti_img = nb.load(cifti_file)
    concat_cifti_img = nb.load(concat_cifti_file)
    assert concat_cifti_img.shape[0] == cifti_img.shape[0] * n_repeats
    assert concat_cifti_img.shape[1] == cifti_img.shape[1]
