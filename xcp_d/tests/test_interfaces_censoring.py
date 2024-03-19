"""Tests for framewise displacement calculation."""

import json
import os

import nibabel as nb
import numpy as np
import pandas as pd

from xcp_d.interfaces import censoring


def test_generate_confounds(ds001419_data, tmp_path_factory):
    """Check results."""
    tmpdir = tmp_path_factory.mktemp("test_generate_confounds")
    in_file = ds001419_data["nifti_file"]
    confounds_file = ds001419_data["confounds_file"]
    confounds_json = ds001419_data["confounds_json"]

    df = pd.read_table(confounds_file)
    with open(confounds_json, "r") as fo:
        metadata = json.load(fo)

    # Replace confounds tsv values with values that should be omitted
    df.loc[1:3, "trans_x"] = [6, 8, 9]
    df.loc[4:6, "trans_y"] = [7, 8, 9]
    df.loc[7:9, "trans_z"] = [12, 8, 9]

    # Modify JSON file
    metadata["trans_x"] = {"test": "hello"}
    confounds_json = os.path.join(tmpdir, "edited_confounds.json")
    with open(confounds_json, "w") as fo:
        json.dump(metadata, fo)

    # Rename with same convention as initial confounds tsv
    confounds_tsv = os.path.join(tmpdir, "edited_confounds.tsv")
    df.to_csv(confounds_tsv, sep="\t", index=False, header=True)

    custom_confounds_file = os.path.join(tmpdir, "custom_confounds.tsv")
    df2 = pd.DataFrame(columns=["signal__test"], data=np.random.random((df.shape[0], 1)))
    df2.to_csv(custom_confounds_file, sep="\t", index=False, header=True)

    # Run workflow
    interface = censoring.GenerateConfounds(
        in_file=in_file,
        params="24P",
        TR=0.8,
        fd_thresh=0.3,
        head_radius=50,
        fmriprep_confounds_file=confounds_tsv,
        fmriprep_confounds_json=confounds_json,
        custom_confounds_file=custom_confounds_file,
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
    out_confounds_file = results.outputs.confounds_file
    out_df = pd.read_table(out_confounds_file)
    assert out_df.shape[1] == 24  # 24(P)
    assert sum(out_df.columns.str.endswith("_orth")) == 24  # all 24(P)


def test_random_censor(tmp_path_factory):
    """Test RandomCensor."""
    tmpdir = tmp_path_factory.mktemp("test_random_censor")
    n_volumes, n_outliers = 500, 100
    exact_scans = [100, 200, 300, 400]

    outliers_arr = np.zeros(n_volumes, dtype=int)
    rng = np.random.default_rng(0)
    outlier_idx = rng.choice(np.arange(n_volumes, dtype=int), size=n_outliers, replace=False)
    outliers_arr[outlier_idx] = 1
    temporal_mask_df = pd.DataFrame(data=outliers_arr, columns=["framewise_displacement"])
    original_temporal_mask = os.path.join(tmpdir, "orig_tmask.tsv")
    temporal_mask_df.to_csv(original_temporal_mask, index=False, sep="\t")

    # Run the RandomCensor interface without any exact_scans.
    interface = censoring.RandomCensor(
        temporal_mask_metadata={},
        temporal_mask=original_temporal_mask,
        exact_scans=[],
        random_seed=0,
    )
    results = interface.run(cwd=tmpdir)
    assert results.outputs.temporal_mask == original_temporal_mask  # same file as input
    assert isinstance(results.outputs.temporal_mask_metadata, dict)

    # Run the interface with exact_scans
    interface = censoring.RandomCensor(
        temporal_mask_metadata={},
        temporal_mask=original_temporal_mask,
        exact_scans=exact_scans,
        random_seed=0,
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.temporal_mask)
    assert isinstance(results.outputs.temporal_mask_metadata, dict)
    new_temporal_mask_df = pd.read_table(results.outputs.temporal_mask)
    new_temporal_mask_df_no_outliers = new_temporal_mask_df.loc[
        new_temporal_mask_df["framewise_displacement"] == 0
    ]
    for exact_scan in exact_scans:
        exact_scan_col = f"exact_{exact_scan}"
        assert exact_scan_col in new_temporal_mask_df_no_outliers.columns
        # The column's values should sum to the number of volumes minus the number of retained.
        # Outliers don't show up here.
        assert new_temporal_mask_df_no_outliers[exact_scan_col].sum() == n_volumes - (
            exact_scan + n_outliers
        )
        # The outlier volumes and exact-scan censored volumes shouldn't overlap.
        assert all(
            new_temporal_mask_df_no_outliers[[exact_scan_col, "framewise_displacement"]].sum(
                axis=1
            )
            <= 1
        )


def test_censor(ds001419_data, tmp_path_factory):
    """Test Censor interface."""
    tmpdir = tmp_path_factory.mktemp("test_censor")
    nifti_file = ds001419_data["nifti_file"]
    cifti_file = ds001419_data["cifti_file"]
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
    assert out_img.shape[0] == n_volumes

    # Test with a NIfTI file, with some censored volumes
    n_censored_volumes = 10
    n_retained_volumes = n_volumes - n_censored_volumes
    censoring_df.loc[range(10), "framewise_displacement"] = 1
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
    assert out_img.shape[0] == n_retained_volumes
