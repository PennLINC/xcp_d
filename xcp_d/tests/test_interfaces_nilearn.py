"""Tests for the xcp_d.interfaces.nilearn module."""

import os

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from xcp_d.interfaces import nilearn


def test_nilearn_merge(ds001419_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.Merge."""
    tmpdir = tmp_path_factory.mktemp('test_nilearn_merge')

    in_file = ds001419_data['boldref']
    interface = nilearn.Merge(
        in_files=[in_file, in_file],
        out_file='merged.nii.gz',
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 4
    assert out_img.shape[3] == 2


def test_nilearn_smooth(ds001419_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.Smooth."""
    tmpdir = tmp_path_factory.mktemp('test_nilearn_smooth')

    in_file = ds001419_data['boldref']
    interface = nilearn.Smooth(
        in_file=in_file,
        fwhm=6,
        out_file='smoothed_1len.nii.gz',
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3

    interface = nilearn.Smooth(
        in_file=in_file,
        fwhm=[2, 3, 4],
        out_file='smoothed_3len.nii.gz',
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3


def test_nilearn_binarymath(ds001419_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.BinaryMath."""
    tmpdir = tmp_path_factory.mktemp('test_nilearn_binarymath')

    in_file = ds001419_data['brain_mask_file']
    interface = nilearn.BinaryMath(
        in_file=in_file,
        expression='img * 5',
        out_file='mathed.nii.gz',
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    out_data = out_img.get_fdata()
    assert np.max(out_data) == 5
    assert np.min(out_data) == 0


def test_nilearn_resampletoimage(datasets, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.ResampleToImage."""
    tmpdir = tmp_path_factory.mktemp('test_nilearn_resampletoimage')

    source_file = os.path.join(
        datasets['nibabies'],
        'sub-01',
        'ses-1mo',
        'func',
        'sub-01_ses-1mo_task-rest_acq-PA_run-001_space-MNIInfant_cohort-1_boldref.nii.gz',
    )
    target_file = os.path.join(
        datasets['nibabies'],
        'sub-01',
        'ses-1mo',
        'anat',
        'sub-01_ses-1mo_run-001_space-MNIInfant_cohort-1_desc-preproc_T1w.nii.gz',
    )
    target_img = nb.load(target_file)
    source_img = nb.load(source_file)
    assert not np.array_equal(target_img.header.get_zooms(), source_img.header.get_zooms())
    interface = nilearn.ResampleToImage(
        in_file=source_file,
        target_file=target_file,
        out_file='resampled.nii.gz',
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3
    assert np.array_equal(target_img.affine, out_img.affine)
    assert np.array_equal(target_img.header.get_zooms(), out_img.header.get_zooms())


def test_nilearn_denoisenifti(ds001419_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.DenoiseNifti."""
    tmpdir = tmp_path_factory.mktemp('test_nilearn_denoisenifti')

    preprocessed_bold = ds001419_data['nifti_file']
    mask = ds001419_data['brain_mask_file']
    confounds_file = ds001419_data['confounds_file']

    # Select some confounds to use for denoising
    confounds_df = pd.read_table(confounds_file)
    reduced_confounds_df = confounds_df[['csf', 'white_matter']]
    reduced_confounds_file = os.path.join(tmpdir, 'confounds.tsv')
    reduced_confounds_df.to_csv(reduced_confounds_file, sep='\t', index=False)

    # Create the censoring file
    censoring_df = confounds_df[['framewise_displacement']].copy()
    censoring_df['framewise_displacement'] = (censoring_df['framewise_displacement'] > 0.3).astype(
        int
    )
    censoring_df['censor_between'] = 0
    censoring_df['denoising'] = censoring_df['framewise_displacement']
    assert censoring_df['denoising'].sum() > 0
    temporal_mask = os.path.join(tmpdir, 'censoring.tsv')
    censoring_df.to_csv(temporal_mask, sep='\t', index=False)

    preprocessed_img = nb.load(preprocessed_bold)

    interface = nilearn.DenoiseNifti(
        preprocessed_bold=preprocessed_bold,
        confounds_tsv=reduced_confounds_file,
        temporal_mask=temporal_mask,
        mask=mask,
        TR=2,
        bandpass_filter=True,
        high_pass=0.01,
        low_pass=0.08,
        filter_order=2,
        num_threads=1,
    )
    results = interface.run(cwd=tmpdir)

    _check_denoising_outputs(preprocessed_img, results.outputs, cifti=False)


def test_nilearn_denoisecifti(ds001419_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.DenoiseCifti."""
    tmpdir = tmp_path_factory.mktemp('test_nilearn_denoisecifti')

    preprocessed_bold = ds001419_data['cifti_file']
    confounds_file = ds001419_data['confounds_file']

    # Select some confounds to use for denoising
    confounds_df = pd.read_table(confounds_file)
    reduced_confounds_df = confounds_df[['csf', 'white_matter']]
    reduced_confounds_file = os.path.join(tmpdir, 'confounds.tsv')
    reduced_confounds_df.to_csv(reduced_confounds_file, sep='\t', index=False)

    # Create the censoring file
    censoring_df = confounds_df[['framewise_displacement']].copy()
    censoring_df['framewise_displacement'] = (censoring_df['framewise_displacement'] > 0.3).astype(
        int
    )
    censoring_df['censor_between'] = 0
    censoring_df['denoising'] = censoring_df['framewise_displacement']
    assert censoring_df['denoising'].sum() > 0
    temporal_mask = os.path.join(tmpdir, 'censoring.tsv')
    censoring_df.to_csv(temporal_mask, sep='\t', index=False)

    preprocessed_img = nb.load(preprocessed_bold)

    interface = nilearn.DenoiseCifti(
        preprocessed_bold=preprocessed_bold,
        confounds_tsv=reduced_confounds_file,
        temporal_mask=temporal_mask,
        TR=2,
        bandpass_filter=True,
        high_pass=0.01,
        low_pass=0.08,
        filter_order=2,
    )
    results = interface.run(cwd=tmpdir)

    _check_denoising_outputs(preprocessed_img, results.outputs, cifti=True)


def test_nilearn_denoisecifti_voxelwise(tmp_path_factory):
    """Test that xcp_d.interfaces.nilearn.DenoiseCifti uses voxel-wise confounds correctly.

    CIFTI files are read as SxT, so the voxel-wise confounds must be transposed to TxS,
    like the BOLD data, before denoising.
    """
    from xcp_d.utils.utils import denoise_with_nilearn

    tmpdir = tmp_path_factory.mktemp('test_nilearn_denoisecifti_voxelwise')
    n_volumes, TR = 50, 2
    brain_models = nb.cifti2.BrainModelAxis.from_mask(
        np.ones((4, 4, 4), dtype=bool),
        affine=np.eye(4),
        name='thalamus_left',
    )
    series = nb.cifti2.SeriesAxis(start=0, step=TR, size=n_volumes)
    n_voxels = len(brain_models)
    assert n_voxels != n_volumes

    rng = np.random.default_rng(0)
    voxelwise_arr = rng.standard_normal((n_volumes, n_voxels))
    bold_arr = rng.standard_normal((n_volumes, n_voxels)) + 10 * voxelwise_arr + 100
    bold_file = os.path.join(tmpdir, 'bold.dtseries.nii')
    voxelwise_file = os.path.join(tmpdir, 'voxelwise.dtseries.nii')
    nb.Cifti2Image(bold_arr, header=(series, brain_models)).to_filename(bold_file)
    nb.Cifti2Image(voxelwise_arr, header=(series, brain_models)).to_filename(voxelwise_file)

    temporal_mask = os.path.join(tmpdir, 'censoring.tsv')
    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[20:25] = False
    pd.DataFrame({'denoising': (~sample_mask).astype(int)}).to_csv(
        temporal_mask,
        sep='\t',
        index=False,
    )

    interface = nilearn.DenoiseCifti(
        preprocessed_bold=bold_file,
        confounds_tsv=None,
        confounds_images=[voxelwise_file],
        temporal_mask=temporal_mask,
        TR=TR,
        bandpass_filter=False,
        high_pass=0,
        low_pass=0,
        filter_order=0,
    )
    results = interface.run(cwd=tmpdir)

    # CIFTI data arrays are TxS
    denoised_arr = nb.load(results.outputs.denoised_interpolated_bold).get_fdata()
    expected_arr = denoise_with_nilearn(
        preprocessed_bold=nb.load(bold_file).get_fdata(),
        confounds=None,
        voxelwise_confounds=[nb.load(voxelwise_file).get_fdata()],
        sample_mask=sample_mask,
        low_pass=None,
        high_pass=None,
        filter_order=0,
        TR=TR,
        num_threads=1,
    )
    assert denoised_arr.shape == (n_volumes, n_voxels)
    assert np.allclose(denoised_arr, expected_arr, atol=1e-4)


@pytest.mark.parametrize('cifti', [True, False])
@pytest.mark.parametrize('interpolation', ['spline', 'linear', 'spectral'])
def test_nilearn_denoise_interpolation(tmp_path_factory, interpolation, cifti):
    """Test that the denoising interfaces pass the interpolation method through."""
    from xcp_d.utils.utils import denoise_with_nilearn

    tmpdir = tmp_path_factory.mktemp('test_nilearn_denoise_interpolation')
    n_volumes, TR = 50, 2
    mask_arr = np.ones((3, 3, 3), dtype=bool)
    n_voxels = int(mask_arr.sum())

    rng = np.random.default_rng(0)
    bold_arr = (rng.standard_normal((n_volumes, n_voxels)) + 100).astype(np.float32)
    confounds_df = pd.DataFrame(rng.standard_normal((n_volumes, 2)), columns=['a', 'b'])
    confounds_tsv = os.path.join(tmpdir, 'confounds.tsv')
    confounds_df.to_csv(confounds_tsv, sep='\t', index=False)

    sample_mask = np.ones(n_volumes, dtype=bool)
    sample_mask[[0, 20, 21, 22, 23, 40]] = False
    temporal_mask = os.path.join(tmpdir, 'censoring.tsv')
    pd.DataFrame({'denoising': (~sample_mask).astype(int)}).to_csv(
        temporal_mask,
        sep='\t',
        index=False,
    )

    kwargs = {
        'confounds_tsv': confounds_tsv,
        'temporal_mask': temporal_mask,
        'TR': TR,
        'bandpass_filter': True,
        'high_pass': 0.01,
        'low_pass': 0.08,
        'filter_order': 2,
        'interpolation': interpolation,
    }
    if cifti:
        brain_models = nb.cifti2.BrainModelAxis.from_mask(
            mask_arr,
            affine=np.eye(4),
            name='thalamus_left',
        )
        series = nb.cifti2.SeriesAxis(start=0, step=TR, size=n_volumes)
        bold_file = os.path.join(tmpdir, 'bold.dtseries.nii')
        nb.Cifti2Image(bold_arr, header=(series, brain_models)).to_filename(bold_file)
        interface = nilearn.DenoiseCifti(preprocessed_bold=bold_file, **kwargs)
    else:
        bold_file = os.path.join(tmpdir, 'bold.nii.gz')
        mask_file = os.path.join(tmpdir, 'mask.nii.gz')
        nb.Nifti1Image(bold_arr.T.reshape((3, 3, 3, n_volumes)), np.eye(4)).to_filename(
            bold_file,
        )
        nb.Nifti1Image(mask_arr.astype(np.uint8), np.eye(4)).to_filename(mask_file)
        interface = nilearn.DenoiseNifti(preprocessed_bold=bold_file, mask=mask_file, **kwargs)

    results = interface.run(cwd=tmpdir)
    if cifti:
        denoised_arr = nb.load(results.outputs.denoised_interpolated_bold).get_fdata()
    else:
        denoised_img = nb.load(results.outputs.denoised_interpolated_bold)
        denoised_arr = denoised_img.get_fdata().reshape((n_voxels, n_volumes)).T

    expected_arr = denoise_with_nilearn(
        preprocessed_bold=bold_arr,
        confounds=confounds_df,
        voxelwise_confounds=None,
        sample_mask=sample_mask,
        low_pass=0.08,
        high_pass=0.01,
        filter_order=2,
        TR=TR,
        num_threads=1,
        interpolation=interpolation,
    )
    assert np.allclose(denoised_arr, expected_arr, atol=1e-4)


def _check_denoising_outputs(preprocessed_img, outputs, cifti):
    if cifti:
        ndim = 2
        hdr_attr = 'nifti_header'
    else:
        ndim = 4
        hdr_attr = 'header'

    preprocessed_img_header = getattr(preprocessed_img, hdr_attr)

    # denoised_interpolated_bold is the censored, denoised, interpolated, and filtered data
    assert os.path.isfile(outputs.denoised_interpolated_bold)
    filtered_denoised_img = nb.load(outputs.denoised_interpolated_bold)
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
