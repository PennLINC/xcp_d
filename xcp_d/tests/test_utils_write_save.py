"""Tests for the xcp_d.utils.write_save module."""

import os

import numpy as np
import pytest

from xcp_d.utils import write_save


def test_read_ndata(ds001419_data):
    """Test write_save.read_ndata."""
    # Try to load a gifti
    gifti_file = ds001419_data['gifti_file']
    with pytest.raises(ValueError, match='Unknown extension'):
        write_save.read_ndata(gifti_file)

    # Load cifti
    cifti_file = ds001419_data['cifti_file']
    cifti_data = write_save.read_ndata(cifti_file)
    assert cifti_data.shape == (91282, 60)

    # Load nifti
    nifti_file = ds001419_data['nifti_file']
    mask_file = ds001419_data['brain_mask_file']

    with pytest.raises(ValueError, match='must be provided'):
        write_save.read_ndata(nifti_file, maskfile=None)

    nifti_data = write_save.read_ndata(nifti_file, maskfile=mask_file)
    assert nifti_data.shape == (242716, 60)


def test_write_ndata(ds001419_data, tmp_path_factory):
    """Test write_save.write_ndata."""
    tmpdir = tmp_path_factory.mktemp('test_write_ndata')

    orig_cifti = ds001419_data['cifti_file']
    cifti_data = write_save.read_ndata(orig_cifti)
    cifti_data[1000, 50] = 1000

    # Write an unmodified CIFTI
    temp_cifti = os.path.join(tmpdir, 'file.dtseries.nii')
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 60)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000, 50] - 1000) < 1

    # Write a shortened CIFTI, so that the time axis will need to be created by write_ndata
    cifti_data = cifti_data[:, ::2]
    assert cifti_data.shape == (91282, 30)
    temp_cifti = os.path.join(tmpdir, 'file.dtseries.nii')
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 30)
    # It won't equal exactly 1000, but check that the modified value is in the right place
    assert (cifti_data_loaded[1000, 25] - 1000) < 1

    # Write a dscalar file (no time points)
    cifti_data = cifti_data[:, 24:25]
    assert cifti_data.shape == (91282, 1)
    temp_cifti = os.path.join(tmpdir, 'file.dscalar.nii')
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 1)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000, 0] - 1000) < 1

    # Write a 1D dscalar file (no time points)
    cifti_data = cifti_data[:, 0]
    assert cifti_data.shape == (91282,)
    temp_cifti = os.path.join(tmpdir, 'file.dscalar.nii')
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 1)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000, 0] - 1000) < 1

    # Try writing out a different CIFTI filetype (should fail)
    temp_cifti = os.path.join(tmpdir, 'file.dlabel.nii')
    with pytest.raises(ValueError, match='Unsupported CIFTI extension'):
        write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)

    # Try writing out a completely different filetype (should fail)
    out_file = os.path.join(tmpdir, 'file.txt')
    with pytest.raises(ValueError, match='Unsupported CIFTI extension'):
        write_save.write_ndata(cifti_data, template=orig_cifti, filename=out_file)

    # Try using a txt file as a template
    fake_template = os.path.join(tmpdir, 'file.txt')
    with open(fake_template, 'w') as fo:
        fo.write('TEST')

    with pytest.raises(ValueError, match='Unknown extension'):
        write_save.write_ndata(cifti_data, template=fake_template, filename=temp_cifti)


def test_write_ndata_raises_for_invalid_inputs(ds001419_data, tmp_path):
    """write_ndata raises ValueError/FileNotFoundError for invalid inputs."""
    orig_cifti = ds001419_data['cifti_file']
    cifti_data = write_save.read_ndata(orig_cifti)
    temp_out = os.path.join(tmp_path, 'out.dtseries.nii')

    with pytest.raises(ValueError, match='Input data must be a 1-2D array'):
        write_save.write_ndata(
            np.zeros((2, 3, 4)),  # 3D array
            template=orig_cifti,
            filename=temp_out,
        )

    with pytest.raises(FileNotFoundError, match='Template file does not exist'):
        write_save.write_ndata(
            cifti_data,
            template='/nonexistent/template.dtseries.nii',
            filename=temp_out,
        )

    nifti_template = ds001419_data['nifti_file']
    mask_file = ds001419_data['brain_mask_file']
    nifti_data = write_save.read_ndata(nifti_template, maskfile=mask_file)

    with pytest.raises(ValueError, match='A binary mask must be provided'):
        write_save.write_ndata(
            nifti_data,
            template=nifti_template,
            filename=os.path.join(tmp_path, 'out.nii.gz'),
            mask=None,
        )

    with pytest.raises(FileNotFoundError, match='The mask file does not exist'):
        write_save.write_ndata(
            nifti_data,
            template=nifti_template,
            filename=os.path.join(tmp_path, 'out.nii.gz'),
            mask='/nonexistent/mask.nii.gz',
        )


def test_read_and_write_ndata_float32(tmp_path):
    """Check that read_ndata reads float32 data and write_ndata writes float32 files.

    The inputs are float64 CIFTI and int16 NIfTI files.
    """
    import nibabel as nb

    n_volumes = 5
    rng = np.random.default_rng(0)

    brain_models = nb.cifti2.BrainModelAxis.from_mask(
        np.ones((3, 3, 3), dtype=bool),
        affine=np.eye(4),
        name='thalamus_left',
    )
    series = nb.cifti2.SeriesAxis(start=0, step=2, size=n_volumes)
    cifti_arr = rng.standard_normal((n_volumes, len(brain_models)))
    cifti_file = str(tmp_path / 'float64.dtseries.nii')
    nb.Cifti2Image(cifti_arr, header=(series, brain_models)).to_filename(cifti_file)
    assert nb.load(cifti_file).get_data_dtype() == np.float64

    cifti_data = write_save.read_ndata(cifti_file)
    assert cifti_data.dtype == np.float32
    assert np.array_equal(cifti_data, cifti_arr.T.astype(np.float32))

    for out_name in ('out.dtseries.nii', 'out.dscalar.nii'):
        out_file = str(tmp_path / out_name)
        write_save.write_ndata(cifti_arr.T, template=cifti_file, filename=out_file, TR=2)
        assert nb.load(out_file).get_data_dtype() == np.float32
        assert np.array_equal(write_save.read_ndata(out_file), cifti_data)

    # A shorter series, for which the time axis is created by write_ndata
    out_file = str(tmp_path / 'short.dtseries.nii')
    write_save.write_ndata(cifti_arr[:3].T, template=cifti_file, filename=out_file, TR=2)
    assert nb.load(out_file).get_data_dtype() == np.float32

    nifti_arr = rng.integers(0, 1000, size=(3, 3, 3, n_volumes)).astype(np.int16)
    nifti_file = str(tmp_path / 'int16.nii.gz')
    nb.Nifti1Image(nifti_arr, affine=np.eye(4)).to_filename(nifti_file)
    assert nb.load(nifti_file).get_data_dtype() == np.int16
    mask_file = str(tmp_path / 'mask.nii.gz')
    nb.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), affine=np.eye(4)).to_filename(mask_file)

    nifti_data = write_save.read_ndata(nifti_file, maskfile=mask_file)
    assert nifti_data.dtype == np.float32
    assert np.array_equal(nifti_data, nifti_arr.reshape(-1, n_volumes).astype(np.float32))

    out_file = str(tmp_path / 'out.nii.gz')
    write_save.write_ndata(
        nifti_data.astype(np.float64),
        template=nifti_file,
        filename=out_file,
        mask=mask_file,
        TR=2,
    )
    assert nb.load(out_file).get_data_dtype() == np.float32
    assert np.array_equal(write_save.read_ndata(out_file, maskfile=mask_file), nifti_data)
