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

    cifti_file = ds001419_data["cifti_file"]
    cifti_data = write_save.read_ndata(cifti_file)
    cifti_data[1000, 50] = 1000

    # Write an unmodified CIFTI
    temp_cifti_file = os.path.join(tmpdir, "cifti_file.dtseries.nii")
    write_save.write_ndata(cifti_data, template=cifti_file, filename=temp_cifti_file)
    assert os.path.isfile(temp_cifti_file)
    cifti_data_loaded = write_save.read_ndata(temp_cifti_file)
    assert cifti_data_loaded.shape == (91282, 60)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000, 50] - 1000) < 1

    # Write a shortened CIFTI
    cifti_data = cifti_data[:, ::2]
    assert cifti_data.shape == (91282, 30)

    temp_cifti_file = os.path.join(tmpdir, "shortened_cifti_file.dtseries.nii")
    write_save.write_ndata(cifti_data, template=cifti_file, filename=temp_cifti_file)
    assert os.path.isfile(temp_cifti_file)
    cifti_data_loaded = write_save.read_ndata(temp_cifti_file)
    assert cifti_data_loaded.shape == (91282, 30)
    # It won't equal exactly 1000, but check that the modified value is in the right place
    assert (cifti_data_loaded[1000, 25] - 1000) < 1

    # Write a CIFTI image (no time points)
    cifti_data = cifti_data[:, 25]
    assert cifti_data.shape == (91282,)

    temp_cifti_file = os.path.join(tmpdir, "shortened_cifti_file.dtseries.nii")
    write_save.write_ndata(cifti_data, template=cifti_file, filename=temp_cifti_file)
    assert os.path.isfile(temp_cifti_file)
    cifti_data_loaded = write_save.read_ndata(temp_cifti_file)
    assert cifti_data_loaded.shape == (91282,)
    # It won't equal exactly 1000
    assert (cifti_data_loaded[1000] - 1000) < 1
