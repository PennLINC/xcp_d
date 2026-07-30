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
        columns=['index', 'name'],
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
    results = parcellator.run(cwd=tmpdir)
    coverage = results.outputs.coverage
    timeseries = results.outputs.timeseries
    assert os.path.isfile(coverage)
    assert os.path.isfile(timeseries)
    coverage_df = pd.read_table(coverage, index_col='Node')
    timeseries_df = pd.read_table(timeseries)
    assert coverage_df.shape == (5, 1)
    assert timeseries_df.shape == (1, 5)
    assert np.array_equal(coverage_df['coverage'].to_numpy(), np.array([0, 0, 1, 1, 1]))
    assert np.array_equal(
        timeseries_df.to_numpy(),
        np.array([[np.nan, np.nan, 3, 4, 5]]),
        equal_nan=True,
    )

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
    results = parcellator.run(cwd=tmpdir)
    coverage = results.outputs.coverage
    timeseries = results.outputs.timeseries
    assert os.path.isfile(coverage)
    assert os.path.isfile(timeseries)
    coverage_df = pd.read_table(coverage, index_col='Node')
    timeseries_df = pd.read_table(timeseries)
    assert coverage_df.shape == (5, 1)
    assert timeseries_df.shape == (1, 5)
    assert np.array_equal(coverage_df['coverage'].to_numpy(), np.array([0, 0, 0.75, 0.5, 0.25]))
    assert np.array_equal(
        timeseries_df.to_numpy(),
        np.array([[np.nan, np.nan, 3, 4, np.nan]]),
        equal_nan=True,
    )


def test_correlate_timeseries_uses_denoising_column(tmp_path_factory):
    """correlate_timeseries drops every volume flagged by the denoising column."""
    import os

    import pandas as pd

    from xcp_d.interfaces.connectivity import correlate_timeseries

    tmpdir = tmp_path_factory.mktemp('test_correlate_timeseries_denoising')
    n_volumes = 40

    rng = np.random.default_rng(0)
    timeseries_df = pd.DataFrame({'roi_a': rng.random(n_volumes), 'roi_b': rng.random(n_volumes)})
    timeseries = os.path.join(tmpdir, 'timeseries.tsv')
    timeseries_df.to_csv(timeseries, sep='\t', index=False)

    # FD flags volumes 0-1; the expansion additionally flags volumes 2-3.
    fd_arr = np.zeros(n_volumes, dtype=int)
    fd_arr[:2] = 1
    between_arr = np.zeros(n_volumes, dtype=int)
    between_arr[2:4] = 1
    censoring_df = pd.DataFrame(
        {
            'framewise_displacement': fd_arr,
            'censor_between': between_arr,
            'denoising': ((fd_arr + between_arr) > 0).astype(int),
        }
    )
    temporal_mask = os.path.join(tmpdir, 'tmask.tsv')
    censoring_df.to_csv(temporal_mask, sep='\t', index=False)

    correlations_df, _ = correlate_timeseries(timeseries, temporal_mask=temporal_mask)

    # Correlating on `denoising` drops 4 volumes; correlating on `framewise_displacement`
    # would drop only 2, giving a different value.
    expected = timeseries_df.iloc[4:].corr()
    wrong = timeseries_df.iloc[2:].corr()
    assert np.isclose(correlations_df.loc['roi_a', 'roi_b'], expected.loc['roi_a', 'roi_b'])
    assert not np.isclose(correlations_df.loc['roi_a', 'roi_b'], wrong.loc['roi_a', 'roi_b'])
