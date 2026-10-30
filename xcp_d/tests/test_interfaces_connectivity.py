"""Tests for xcp_d.interfaces.utils module."""

import os

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

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


def test_tsvconnect_uses_denoising_column(tmp_path_factory):
    """TSVConnect drops every volume flagged by the denoising column."""
    from xcp_d.interfaces.connectivity import TSVConnect

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

    results = TSVConnect(timeseries=timeseries, temporal_mask=temporal_mask).run(cwd=tmpdir)
    correlations_df = pd.read_table(results.outputs.r, index_col='Node')

    # Correlating on `denoising` drops 4 volumes; correlating on `framewise_displacement`
    # would drop only 2, giving a different value.
    expected = timeseries_df.iloc[4:].corr()
    wrong = timeseries_df.iloc[2:].corr()
    assert np.isclose(correlations_df.loc['roi_a', 'roi_b'], expected.loc['roi_a', 'roi_b'])
    assert not np.isclose(correlations_df.loc['roi_a', 'roi_b'], wrong.loc['roi_a', 'roi_b'])


def test_tsvconnect_without_temporal_mask(tmp_path_factory):
    """TSVConnect falls back to Pearson correlations, without variances, if there is no mask."""
    from xcp_d.interfaces.connectivity import TSVConnect

    tmpdir = tmp_path_factory.mktemp('test_tsvconnect_without_temporal_mask')
    rng = np.random.default_rng(0)
    timeseries_df = pd.DataFrame(rng.random((30, 3)), columns=['roi_a', 'roi_b', 'roi_c'])
    timeseries = os.path.join(tmpdir, 'timeseries.tsv')
    timeseries_df.to_csv(timeseries, sep='\t', index=False)

    results = TSVConnect(timeseries=timeseries).run(cwd=tmpdir)

    r_df = pd.read_table(results.outputs.r, index_col='Node')
    z_df = pd.read_table(results.outputs.z, index_col='Node')
    pd.testing.assert_frame_equal(r_df, timeseries_df.corr(), check_names=False)
    offdiag = ~np.eye(3, dtype=bool)
    assert np.allclose(z_df.to_numpy()[offdiag], np.arctanh(r_df.to_numpy()[offdiag]))
    for output in ['var_r', 'var_z', 'r_exact', 'z_exact', 'var_r_exact', 'var_z_exact']:
        assert getattr(results.outputs, output) is None


def test_tsvconnect_exact_scans(tmp_path_factory):
    """TSVConnect writes one set of matrices per exact-scan column."""
    from xcp_d.interfaces.connectivity import TSVConnect

    tmpdir = tmp_path_factory.mktemp('test_tsvconnect_exact_scans')
    n_volumes = 40
    rng = np.random.default_rng(0)
    timeseries_df = pd.DataFrame(rng.random((n_volumes, 2)), columns=['roi_a', 'roi_b'])
    timeseries = os.path.join(tmpdir, 'timeseries.tsv')
    timeseries_df.to_csv(timeseries, sep='\t', index=False)

    denoising = np.zeros(n_volumes, dtype=int)
    denoising[:4] = 1
    # Exact-scan columns flag the additional volumes to drop
    exact = np.zeros(n_volumes, dtype=int)
    exact[30:] = 1
    censoring_df = pd.DataFrame({'denoising': denoising, 'exact_26': exact})
    temporal_mask = os.path.join(tmpdir, 'tmask.tsv')
    censoring_df.to_csv(temporal_mask, sep='\t', index=False)

    results = TSVConnect(timeseries=timeseries, temporal_mask=temporal_mask).run(cwd=tmpdir)

    for measure in ['r', 'z', 'var_r', 'var_z']:
        exact_files = getattr(results.outputs, f'{measure}_exact')
        assert len(exact_files) == 1
        assert os.path.isfile(exact_files[0])

    r_df = pd.read_table(results.outputs.r, index_col='Node')
    r_exact_df = pd.read_table(results.outputs.r_exact[0], index_col='Node')
    assert np.isclose(
        r_df.loc['roi_a', 'roi_b'],
        timeseries_df.iloc[4:].corr().loc['roi_a', 'roi_b'],
    )
    assert np.isclose(
        r_exact_df.loc['roi_a', 'roi_b'],
        timeseries_df.iloc[4:30].corr().loc['roi_a', 'roi_b'],
    )


def _make_pconn_inputs(tmpdir, labels_df, tsv_names):
    """Write a 3-parcel ptseries CIFTI, an atlas labels file, and a TSV matrix.

    The TSV rows/columns are in reverse CIFTI order, and value (i, j) is 10 * i + j,
    where i and j are the CIFTI parcel indices.
    """
    brain_models = nb.cifti2.BrainModelAxis.from_surface(np.arange(6), 6, 'CortexLeft')
    parcels = nb.cifti2.ParcelsAxis.from_brain_models(
        [
            ('P_A', brain_models[:2]),
            ('P_B', brain_models[2:4]),
            ('P_C', brain_models[4:]),
        ]
    )
    series = nb.cifti2.SeriesAxis(start=0, step=1, size=5)
    source_cifti = os.path.join(tmpdir, 'source.ptseries.nii')
    nb.Cifti2Image(np.zeros((5, 3)), header=(series, parcels)).to_filename(source_cifti)

    atlas_labels = os.path.join(tmpdir, 'labels.tsv')
    labels_df.to_csv(atlas_labels, sep='\t', index=False)

    order = [2, 1, 0]
    matrix = 10 * np.array(order)[:, None] + np.array(order)[None, :]
    names = [tsv_names[i] for i in order]
    in_file = os.path.join(tmpdir, 'r.tsv')
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        in_file,
        sep='\t',
        index_label='Node',
    )
    return source_cifti, atlas_labels, in_file


@pytest.mark.parametrize(
    ('labels', 'tsv_names'),
    [
        # cifti_label column, with a background row that should be dropped
        (
            {
                'index': [0, 1, 2, 3],
                'name': ['background', 'a', 'b', 'c'],
                'cifti_label': ['???', 'P_A', 'P_B', 'P_C'],
            },
            ['a', 'b', 'c'],
        ),
        # label_7network column, falling back to the name when it is missing
        (
            {
                'index': [1, 2, 3],
                'name': ['a', 'b', 'P_C'],
                'label_7network': ['P_A', 'P_B', None],
            },
            ['a', 'b', 'P_C'],
        ),
        # Names alone, which must match the CIFTI parcel names
        ({'index': [1, 2, 3], 'name': ['P_A', 'P_B', 'P_C']}, ['P_A', 'P_B', 'P_C']),
    ],
)
def test_tsv_to_pconn(tmp_path_factory, labels, tsv_names):
    """TSVToPconn maps TSV node names to CIFTI parcels and reorders the matrix."""
    from xcp_d.interfaces.connectivity import TSVToPconn

    tmpdir = tmp_path_factory.mktemp('test_tsv_to_pconn')
    source_cifti, atlas_labels, in_file = _make_pconn_inputs(
        tmpdir,
        pd.DataFrame(labels),
        tsv_names,
    )

    results = TSVToPconn(
        in_file=in_file,
        source_cifti=source_cifti,
        atlas_labels=atlas_labels,
    ).run(cwd=tmpdir)

    out_img = nb.load(results.outputs.out_file)
    expected = 10 * np.arange(3)[:, None] + np.arange(3)[None, :]
    np.testing.assert_array_equal(out_img.get_fdata(), expected)
    for i_axis in range(2):
        assert list(out_img.header.get_axis(i_axis).name) == ['P_A', 'P_B', 'P_C']

    # A list of files gives a list of outputs
    results = TSVToPconn(
        in_file=[in_file],
        source_cifti=source_cifti,
        atlas_labels=atlas_labels,
    ).run(cwd=tmpdir)
    assert isinstance(results.outputs.out_file, list)
    assert len(results.outputs.out_file) == 1


@pytest.mark.parametrize(
    ('labels', 'match'),
    [
        ({'index': [1, 2], 'name': ['P_A', 'P_B']}, 'Missing CIFTI labels'),
        (
            {'index': [1, 2, 3, 4], 'name': ['P_A', 'P_B', 'P_C', 'P_D']},
            'Missing TSV labels',
        ),
    ],
)
def test_tsv_to_pconn_mismatched_labels(tmp_path_factory, labels, match):
    """TSVToPconn requires the atlas labels and CIFTI parcels to match exactly."""
    from xcp_d.interfaces.connectivity import TSVToPconn

    tmpdir = tmp_path_factory.mktemp('test_tsv_to_pconn_mismatched_labels')
    source_cifti, atlas_labels, in_file = _make_pconn_inputs(
        tmpdir,
        pd.DataFrame(labels),
        ['P_A', 'P_B', 'P_C'],
    )

    with pytest.raises(ValueError, match=match):
        TSVToPconn(
            in_file=in_file,
            source_cifti=source_cifti,
            atlas_labels=atlas_labels,
        ).run(cwd=tmpdir)
