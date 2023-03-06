"""Tests for the xcp_d.interfaces.nilearn module."""
import os

import nibabel as nb
import numpy as np
import pandas as pd

from xcp_d.interfaces import nilearn


def test_nilearn_merge(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.Merge."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_merge")

    in_file = fmriprep_with_freesurfer_data["boldref"]
    interface = nilearn.Merge(
        in_files=[in_file, in_file],
        out_file="merged.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 4
    assert out_img.shape[3] == 2


def test_nilearn_smooth(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.Smooth."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_smooth")

    in_file = fmriprep_with_freesurfer_data["boldref"]
    interface = nilearn.Smooth(
        in_file=in_file,
        fwhm=6,
        out_file="smoothed_1len.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3

    interface = nilearn.Smooth(
        in_file=in_file,
        fwhm=[2, 3, 4],
        out_file="smoothed_3len.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3


def test_nilearn_binarymath(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.BinaryMath."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_binarymath")

    in_file = fmriprep_with_freesurfer_data["brain_mask_file"]
    interface = nilearn.BinaryMath(
        in_file=in_file,
        expression="img * 5",
        out_file="mathed.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    out_data = out_img.get_fdata()
    assert np.max(out_data) == 5
    assert np.min(out_data) == 0


def test_nilearn_resampletoimage(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.ResampleToImage."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_meanimage")

    source_file = fmriprep_with_freesurfer_data["boldref_t1w"]
    target_file = fmriprep_with_freesurfer_data["t1w"]
    target_img = nb.load(target_file)
    source_img = nb.load(source_file)
    assert not np.array_equal(target_img.header.get_zooms(), source_img.header.get_zooms())
    interface = nilearn.ResampleToImage(
        in_file=source_file,
        target_file=target_file,
        out_file="resampled.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3
    assert np.array_equal(target_img.affine, out_img.affine)
    assert np.array_equal(target_img.header.get_zooms(), out_img.header.get_zooms())


def test_nilearn_denoisenifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.DenoiseNifti."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_denoisenifti")

    preprocessed_bold = fmriprep_with_freesurfer_data["nifti_file"]
    mask = fmriprep_with_freesurfer_data["brain_mask_file"]
    confounds_file = fmriprep_with_freesurfer_data["confounds_file"]

    # Select some confounds to use for denoising
    confounds_df = pd.read_table(confounds_file)
    reduced_confounds_df = confounds_df[["csf", "white_matter"]]
    reduced_confounds_df["linear_trend"] = np.arange(reduced_confounds_df.shape[0])
    reduced_confounds_df["intercept"] = np.ones(reduced_confounds_df.shape[0])
    reduced_confounds_file = os.path.join(tmpdir, "confounds.tsv")
    reduced_confounds_df.to_csv(reduced_confounds_file, sep="\t", index=False)

    # Create the censoring file
    censoring_df = confounds_df[["framewise_displacement"]]
    censoring_df["framewise_displacement"] = censoring_df["framewise_displacement"] > 0.2
    assert censoring_df["framewise_displacement"].sum() > 0
    temporal_mask = os.path.join(tmpdir, "censoring.tsv")
    censoring_df.to_csv(temporal_mask, sep="\t", index=False)

    preprocessed_img = nb.load(preprocessed_bold)

    interface = nilearn.DenoiseNifti(
        preprocessed_bold=preprocessed_bold,
        confounds_file=reduced_confounds_file,
        temporal_mask=temporal_mask,
        mask=mask,
        TR=2,
        bandpass_filter=True,
        high_pass=0.01,
        low_pass=0.08,
        filter_order=2,
    )
    results = interface.run(cwd=tmpdir)

    _check_denoising_outputs(preprocessed_img, results.outputs, cifti=False)


def test_nilearn_denoisecifti(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.DenoiseCifti."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_denoisecifti")

    preprocessed_bold = fmriprep_with_freesurfer_data["cifti_file"]
    confounds_file = fmriprep_with_freesurfer_data["confounds_file"]

    # Select some confounds to use for denoising
    confounds_df = pd.read_table(confounds_file)
    reduced_confounds_df = confounds_df[["csf", "white_matter"]]
    reduced_confounds_df["linear_trend"] = np.arange(reduced_confounds_df.shape[0])
    reduced_confounds_df["intercept"] = np.ones(reduced_confounds_df.shape[0])
    reduced_confounds_file = os.path.join(tmpdir, "confounds.tsv")
    reduced_confounds_df.to_csv(reduced_confounds_file, sep="\t", index=False)

    # Create the censoring file
    censoring_df = confounds_df[["framewise_displacement"]]
    censoring_df["framewise_displacement"] = censoring_df["framewise_displacement"] > 0.2
    assert censoring_df["framewise_displacement"].sum() > 0
    temporal_mask = os.path.join(tmpdir, "censoring.tsv")
    censoring_df.to_csv(temporal_mask, sep="\t", index=False)

    preprocessed_img = nb.load(preprocessed_bold)

    interface = nilearn.DenoiseCifti(
        preprocessed_bold=preprocessed_bold,
        confounds_file=reduced_confounds_file,
        temporal_mask=temporal_mask,
        TR=2,
        bandpass_filter=True,
        high_pass=0.01,
        low_pass=0.08,
        filter_order=2,
    )
    results = interface.run(cwd=tmpdir)

    _check_denoising_outputs(preprocessed_img, results.outputs, cifti=True)


def _check_denoising_outputs(preprocessed_img, outputs, cifti):
    if cifti:
        ndim = 2
        hdr_attr = "nifti_header"
    else:
        ndim = 4
        hdr_attr = "header"

    preprocessed_img_header = getattr(preprocessed_img, hdr_attr)

    # uncensored_denoised_bold is the size of the full run
    assert os.path.isfile(outputs.uncensored_denoised_bold)
    uncensored_denoised_img = nb.load(outputs.uncensored_denoised_bold)
    uncensored_denoised_img_header = getattr(uncensored_denoised_img, hdr_attr)
    assert uncensored_denoised_img.ndim == ndim
    assert uncensored_denoised_img.shape == preprocessed_img.shape
    assert np.array_equal(
        preprocessed_img_header.get_sform(),
        preprocessed_img_header.get_sform(),
    )
    assert np.array_equal(
        uncensored_denoised_img_header.get_zooms()[:-1],
        preprocessed_img_header.get_zooms()[:-1],
    )

    # interpolated_filtered_bold is the censored, denoised, interpolated, and filtered data
    assert os.path.isfile(outputs.interpolated_filtered_bold)
    filtered_denoised_img = nb.load(outputs.interpolated_filtered_bold)
    filtered_denoised_img_header = getattr(filtered_denoised_img, hdr_attr)
    assert filtered_denoised_img.ndim == ndim
    assert filtered_denoised_img.shape == preprocessed_img.shape
    assert np.array_equal(
        filtered_denoised_img_header.get_sform(),
        preprocessed_img_header.get_sform(),
    )
    assert np.array_equal(
        filtered_denoised_img_header.get_zooms()[:-1],
        preprocessed_img_header.get_zooms()[:-1],
    )
