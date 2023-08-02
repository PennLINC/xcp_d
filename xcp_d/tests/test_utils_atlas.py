"""Tests for the xcp_d.utils.atlas module."""
import os

import pytest

from xcp_d.utils import atlas


def test_get_atlas_names():
    """Test xcp_d.utils.atlas.get_atlas_names."""
    atlas_names = atlas.get_atlas_names("all")
    assert isinstance(atlas_names, list)
    assert all(isinstance(name, str) for name in atlas_names)


def test_get_atlas_nifti():
    """Test xcp_d.utils.atlas.get_atlas_nifti."""
    atlas_names = atlas.get_atlas_names("all")
    for atlas_name in atlas_names:
        atlas_file, atlas_labels_file, metadata_file = atlas.get_atlas_nifti(atlas_name)
        assert isinstance(atlas_file, str)
        assert isinstance(atlas_labels_file, str)
        assert isinstance(metadata_file, str)
        assert os.path.isfile(atlas_file)
        assert os.path.isfile(atlas_labels_file)
        assert os.path.isfile(metadata_file)

    with pytest.raises(FileNotFoundError, match="File(s) DNE"):
        atlas.get_atlas_nifti("tofail")


def test_get_atlas_cifti():
    """Test xcp_d.utils.atlas.get_atlas_cifti."""
    atlas_names = atlas.get_atlas_names("all")
    for atlas_name in atlas_names:
        atlas_file, atlas_labels_file, metadata_file = atlas.get_atlas_cifti(atlas_name)
        assert isinstance(atlas_file, str)
        assert isinstance(atlas_labels_file, str)
        assert isinstance(metadata_file, str)
        assert os.path.isfile(atlas_file)
        assert os.path.isfile(atlas_labels_file)
        assert os.path.isfile(metadata_file)

    with pytest.raises(FileNotFoundError, match="File(s) DNE"):
        atlas.get_atlas_cifti("tofail")
