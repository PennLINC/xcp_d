"""Tests for the xcp_d.utils.write_save module."""

import os

import pytest

from xcp_d.utils import write_save


def test_read_ndata(ds001419_data):
    """Test write_save.read_ndata."""
    # Try to load a gifti
    gifti_file = ds001419_data["gifti_file"]
    with pytest.raises(ValueError, match="Unknown extension"):
        write_save.read_ndata(gifti_file)

    # Load cifti
    cifti_file = ds001419_data["cifti_file"]
    cifti_data = write_save.read_ndata(cifti_file)
    assert cifti_data.shape == (91282, 60)

    # Load nifti
    nifti_file = ds001419_data["nifti_file"]
    mask_file = ds001419_data["brain_mask_file"]

    with pytest.raises(AssertionError, match="must be provided"):
        write_save.read_ndata(nifti_file, maskfile=None)

    nifti_data = write_save.read_ndata(nifti_file, maskfile=mask_file)
    assert nifti_data.shape == (249657, 60)


def test_write_ndata(ds001419_data, tmp_path_factory):
    """Test write_save.write_ndata."""
    tmpdir = tmp_path_factory.mktemp("test_write_ndata")

    orig_cifti = ds001419_data["cifti_file"]
    cifti_data = write_save.read_ndata(orig_cifti)
    cifti_data[1000, 50] = 1000

    # Write an unmodified CIFTI
    temp_cifti = os.path.join(tmpdir, "file.dtseries.nii")
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 60)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000, 50] - 1000) < 1

    # Write a shortened CIFTI, so that the time axis will need to be created by write_ndata
    cifti_data = cifti_data[:, ::2]
    assert cifti_data.shape == (91282, 30)
    temp_cifti = os.path.join(tmpdir, "file.dtseries.nii")
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 30)
    # It won't equal exactly 1000, but check that the modified value is in the right place
    assert (cifti_data_loaded[1000, 25] - 1000) < 1

    # Write a dscalar file (no time points)
    cifti_data = cifti_data[:, 24:25]
    assert cifti_data.shape == (91282, 1)
    temp_cifti = os.path.join(tmpdir, "file.dscalar.nii")
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 1)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000, 0] - 1000) < 1

    # Write a 1D dscalar file (no time points)
    cifti_data = cifti_data[:, 0]
    assert cifti_data.shape == (91282,)
    temp_cifti = os.path.join(tmpdir, "file.dscalar.nii")
    write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)
    assert os.path.isfile(temp_cifti)
    cifti_data_loaded = write_save.read_ndata(temp_cifti)
    assert cifti_data_loaded.shape == (91282, 1)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000, 0] - 1000) < 1

    # Try writing out a different CIFTI filetype (should fail)
    temp_cifti = os.path.join(tmpdir, "file.dlabel.nii")
    with pytest.raises(ValueError, match="Unsupported CIFTI extension"):
        write_save.write_ndata(cifti_data, template=orig_cifti, filename=temp_cifti)

    # Try writing out a completely different filetype (should fail)
    out_file = os.path.join(tmpdir, "file.txt")
    with pytest.raises(ValueError, match="Unsupported CIFTI extension"):
        write_save.write_ndata(cifti_data, template=orig_cifti, filename=out_file)

    # Try using a txt file as a template
    fake_template = os.path.join(tmpdir, "file.txt")
    with open(fake_template, "w") as fo:
        fo.write("TEST")

    with pytest.raises(ValueError, match="Unknown extension"):
        write_save.write_ndata(cifti_data, template=fake_template, filename=temp_cifti)
