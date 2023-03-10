"""Test functions in xcp_d.utils.utils."""
import os

import numpy as np
import pandas as pd
from nilearn import masking

from xcp_d.utils import utils


def test_estimate_brain_radius(fmriprep_with_freesurfer_data):
    """Ensure that the brain radius estimation function returns the right value."""
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]

    radius = utils.estimate_brain_radius(bold_mask)
    assert radius == 78.12350298308195


def test_denoise_with_nilearn(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.utils.utils.denoise_with_nilearn."""
    tmpdir = tmp_path_factory.mktemp("test_denoise_with_nilearn")

    high_pass, low_pass, filter_order, TR = 0.01, 0.08, 2, 2

    preprocessed_bold = fmriprep_with_freesurfer_data["nifti_file"]
    confounds_file = fmriprep_with_freesurfer_data["confounds_file"]
    bold_mask = fmriprep_with_freesurfer_data["brain_mask_file"]

    preprocessed_bold_arr = masking.apply_mask(preprocessed_bold, bold_mask)
    # Reduce the size of the data for the test
    preprocessed_bold_arr = preprocessed_bold_arr[:, ::3]
    n_volumes, n_voxels = preprocessed_bold_arr.shape

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
    n_censored_volumes = censoring_df["framewise_displacement"].sum()
    assert n_censored_volumes > 0
    temporal_mask = os.path.join(tmpdir, "censoring.tsv")
    censoring_df.to_csv(temporal_mask, sep="\t", index=False)

    # First, try out filtering
    (
        uncensored_denoised_bold,
        interpolated_filtered_bold,
    ) = utils.denoise_with_nilearn(
        preprocessed_bold=preprocessed_bold_arr,
        confounds_file=reduced_confounds_file,
        temporal_mask=temporal_mask,
        low_pass=low_pass,
        high_pass=high_pass,
        filter_order=filter_order,
        TR=TR,
    )

    assert uncensored_denoised_bold.shape == (n_volumes, n_voxels)
    assert interpolated_filtered_bold.shape == (n_volumes, n_voxels)

    # Now, no filtering
    (
        uncensored_denoised_bold,
        interpolated_filtered_bold,
    ) = utils.denoise_with_nilearn(
        preprocessed_bold=preprocessed_bold_arr,
        confounds_file=reduced_confounds_file,
        temporal_mask=temporal_mask,
        low_pass=None,
        high_pass=None,
        filter_order=None,
        TR=TR,
    )

    assert uncensored_denoised_bold.shape == (n_volumes, n_voxels)
    assert interpolated_filtered_bold.shape == (n_volumes, n_voxels)

    # Finally, do the orthogonalization
    reduced_confounds_df["signal__test"] = confounds_df["global_signal"]

    # Move intercept to end of dataframe
    reduced_confounds_df = reduced_confounds_df[
        [c for c in reduced_confounds_df.columns if c not in ["intercept"]] + ["intercept"]
    ]
    orth_confounds_file = os.path.join(tmpdir, "orth_confounds.tsv")
    reduced_confounds_df.to_csv(orth_confounds_file, sep="\t", index=False)
    (
        uncensored_denoised_bold,
        interpolated_filtered_bold,
    ) = utils.denoise_with_nilearn(
        preprocessed_bold=preprocessed_bold_arr,
        confounds_file=orth_confounds_file,
        temporal_mask=temporal_mask,
        low_pass=low_pass,
        high_pass=high_pass,
        filter_order=filter_order,
        TR=TR,
    )

    assert uncensored_denoised_bold.shape == (n_volumes, n_voxels)
    assert interpolated_filtered_bold.shape == (n_volumes, n_voxels)
