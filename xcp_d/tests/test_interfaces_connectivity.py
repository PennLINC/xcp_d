"""Tests for xcp_d.interfaces.utils module."""

import os

import nibabel as nb
import numpy as np
import pandas as pd

from xcp_d.interfaces.connectivity import NiftiParcellate


def test_nifti_parcellate(tmp_path_factory):
    """Convert nifti files to 32-bit."""
    tmpdir = tmp_path_factory.mktemp('test_nifti_parcellate')

    arr = np.zeros((4, 4, 4))
    arr[0, 0, :] = 3
    arr[0, 1, :] = 4
    arr[0, 2, :] = 5
    atlas_img = nb.Nifti1Image(arr.astype(np.int32), np.eye(4))
    lut = pd.DataFrame(
        columns=['index', 'label'],
        data=[[1, 'Region A'], [2, 'Region B'], [3, 'Region C'], [4, 'Region D'], [5, 'Region E']],
    )

    mask = np.ones((4, 4, 4))
    mask_img = nb.Nifti1Image(mask.astype(np.int32), np.eye(4))
    atlas_file = os.path.join(tmpdir, 'atlas_01.nii.gz')
    atlas_img.to_filename(atlas_file)
    mask_file = os.path.join(tmpdir, 'mask_01.nii.gz')
    mask_img.to_filename(mask_file)
    lut_file = os.path.join(tmpdir, 'lut_01.tsv')
    lut.to_csv(lut_file, sep='\t', index=False)

    # Some parcels are not present, but none are masked out
    parcellator = NiftiParcellate(
        filtered_file=atlas_file,
        mask=mask_file,
        atlas=atlas_file,
        atlas_labels=lut_file,
        min_coverage=0.5,
    )
    results = parcellator.run()
    coverage = results.outputs.coverage
    timeseries = results.outputs.timeseries
    assert os.path.isfile(coverage)
    assert os.path.isfile(timeseries)
    coverage_df = pd.read_table(coverage, index_col='Node')
    timeseries_df = pd.read_table(timeseries)
    assert coverage_df.shape == (5, 1)
    assert timeseries_df.shape == (1, 5)
    assert np.array_equal(coverage_df['coverage'].to_numpy(), np.array([0, 0, 1, 1, 1]))
    assert np.array_equal(timeseries_df.to_numpy(), np.array([[np.nan, np.nan, 3, 4, 5]]))

    # Now let's mask out some voxels
    mask[0, 0, 0] = 0  # 1/4 of the third parcel
    mask[0, 1, :2] = 0  # 1/2 of the fourth parcel
    mask[0, 2, :3] = 0  # 3/4 of the fifth parcel
    mask_img = nb.Nifti1Image(mask.astype(np.int32), np.eye(4))
    mask_file = os.path.join(tmpdir, 'mask_02.nii.gz')
    mask_img.to_filename(mask_file)
    parcellator = NiftiParcellate(
        filtered_file=atlas_file,
        mask=mask_file,
        atlas=atlas_file,
        atlas_labels=lut_file,
        min_coverage=0.5,
    )
    results = parcellator.run()
    coverage = results.outputs.coverage
    timeseries = results.outputs.timeseries
    assert os.path.isfile(coverage)
    assert os.path.isfile(timeseries)
    coverage_df = pd.read_table(coverage, index_col='Node')
    timeseries_df = pd.read_table(timeseries)
    assert coverage_df.shape == (5, 1)
    assert timeseries_df.shape == (1, 5)
    assert np.array_equal(coverage_df['coverage'].to_numpy(), np.array([0, 0, 0.75, 0.5, 0.25]))
    assert np.array_equal(timeseries_df.to_numpy(), np.array([[np.nan, np.nan, 3, 4, np.nan]]))
