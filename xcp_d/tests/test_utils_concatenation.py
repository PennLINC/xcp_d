"""Tests for the xcp_d.utils.concatenation module."""

import os

import nibabel as nb
import numpy as np
import pandas as pd

from xcp_d.utils import concatenation


def test_concatenate_tsvs(ds001419_data, tmp_path_factory):
    """Test xcp_d.utils.concatenation.concatenate_tsvs."""
    tmpdir = tmp_path_factory.mktemp('test_concatenate_tsvs')

    n_repeats = 3

    # First, concatenate TSVs with headers
    tsv_file_with_header = ds001419_data['confounds_file']
    concat_tsv_file_with_header = os.path.join(tmpdir, 'concat_with_header.tsv')
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
    tsv_file_without_header = os.path.join(tmpdir, 'without_header.tsv')
    data = pd.read_table(tsv_file_with_header).to_numpy()
    np.savetxt(tsv_file_without_header, data, fmt='%.5f', delimiter='\t')

    concat_tsv_file_without_header = os.path.join(tmpdir, 'concat_without_header.tsv')
    concatenation.concatenate_tsvs(
        [tsv_file_without_header] * n_repeats,
        out_file=concat_tsv_file_without_header,
    )
    assert os.path.isfile(concat_tsv_file_without_header)
    tsv_arr = np.loadtxt(tsv_file_without_header)
    concat_tsv_arr = np.loadtxt(concat_tsv_file_without_header)
    assert concat_tsv_arr.shape[0] == tsv_arr.shape[0] * n_repeats
    assert concat_tsv_arr.shape[1] == tsv_arr.shape[1]


def test_concatenate_niimgs(ds001419_data, tmp_path_factory):
    """Test xcp_d.utils.concatenation.concatenate_niimgs.

    We don't have non-dtseries CIFTIs to test, so this test is a little limited.
    """
    tmpdir = tmp_path_factory.mktemp('test_concatenate_niimgs')

    n_repeats = 3

    # First, concatenate niftis
    nifti_file = ds001419_data['nifti_file']
    concat_nifti_file = os.path.join(tmpdir, 'concat_nifti.nii.gz')
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
    cifti_file = ds001419_data['cifti_file']
    concat_cifti_file = os.path.join(tmpdir, 'concat_cifti.dtseries.nii')
    concatenation.concatenate_niimgs(
        [cifti_file] * n_repeats,
        out_file=concat_cifti_file,
    )
    assert os.path.isfile(concat_cifti_file)
    cifti_img = nb.load(cifti_file)
    concat_cifti_img = nb.load(concat_cifti_file)
    assert concat_cifti_img.shape[0] == cifti_img.shape[0] * n_repeats
    assert concat_cifti_img.shape[1] == cifti_img.shape[1]


def test_zscore_tsv(tmp_path_factory):
    """Test xcp_d.utils.concatenation.zscore_tsv with all-zero column."""
    tmpdir = tmp_path_factory.mktemp('test_zscore_tsv')
    rng = np.random.default_rng(42)
    n_timepoints = 100

    data = pd.DataFrame(
        {
            'normal_1': rng.normal(5, 2, n_timepoints),
            'normal_2': rng.normal(-3, 1, n_timepoints),
            'all_zero': np.zeros(n_timepoints),
        }
    )
    in_file = str(tmpdir / 'test.tsv')
    data.to_csv(in_file, sep='\t', index=False)
    out_file = str(tmpdir / 'test_zscored.tsv')

    result = concatenation.zscore_tsv(in_file, out_file=out_file)
    out_data = pd.read_table(result)

    assert not out_data.isnull().any().any(), 'NaNs found in z-scored TSV output'
    np.testing.assert_array_equal(out_data['all_zero'].values, 0.0)
    assert abs(out_data['normal_1'].mean()) < 1e-10
    assert abs(out_data['normal_2'].mean()) < 1e-10
    np.testing.assert_approx_equal(out_data['normal_1'].std(ddof=0), 1.0, significant=5)
    np.testing.assert_approx_equal(out_data['normal_2'].std(ddof=0), 1.0, significant=5)


def test_zscore_niimg(ds001419_data, tmp_path_factory):
    """Test xcp_d.utils.concatenation.zscore_niimg with all-zero voxel/vertex."""
    tmpdir = tmp_path_factory.mktemp('test_zscore_niimg')

    # --- NIfTI branch ---
    nifti_img = nb.load(ds001419_data['nifti_file'])
    data = nifti_img.get_fdata()
    data[0, 0, 0, :] = 0.0  # force one voxel to all-zero
    modified_nifti = nb.Nifti1Image(data, nifti_img.affine, nifti_img.header)
    nifti_in = str(tmpdir / 'test.nii.gz')
    modified_nifti.to_filename(nifti_in)
    nifti_out = str(tmpdir / 'test_zscored.nii.gz')

    result = concatenation.zscore_niimg(nifti_in, out_file=nifti_out)
    out_data = nb.load(result).get_fdata()

    assert not np.any(np.isnan(out_data)), 'NaNs found in z-scored NIfTI output'
    np.testing.assert_array_equal(out_data[0, 0, 0, :], 0.0)
    assert abs(np.mean(out_data[1, 1, 1, :])) < 1e-10

    # --- CIFTI branch ---
    cifti_img = nb.load(ds001419_data['cifti_file'])
    cdata = cifti_img.get_fdata()
    cdata[:, 0] = 0.0  # force one vertex to all-zero (axis 0 = time, axis 1 = vertices)
    modified_cifti = nb.Cifti2Image(cdata, cifti_img.header, cifti_img.nifti_header)
    cifti_in = str(tmpdir / 'test.dtseries.nii')
    modified_cifti.to_filename(cifti_in)
    cifti_out = str(tmpdir / 'test_zscored.dtseries.nii')

    result = concatenation.zscore_niimg(cifti_in, out_file=cifti_out)
    out_cdata = nb.load(result).get_fdata()

    assert not np.any(np.isnan(out_cdata)), 'NaNs found in z-scored CIFTI output'
    np.testing.assert_array_equal(out_cdata[:, 0], 0.0)
    assert abs(np.mean(out_cdata[:, 1])) < 1e-10
