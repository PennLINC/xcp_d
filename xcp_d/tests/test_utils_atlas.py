"""Tests for the xcp_d.utils.atlas module."""
import os

from xcp_d.utils import atlas


def test_get_atlas_names():
    """Test xcp_d.utils.atlas.get_atlas_names."""
    atlas_names = atlas.get_atlas_names()
    assert isinstance(atlas_names, list)
    assert all(isinstance(name, str) for name in atlas_names)


def test_get_atlas_nifti():
    """Test xcp_d.utils.atlas.get_atlas_nifti."""
    atlas_names = atlas.get_atlas_names()
    for atlas_name in atlas_names:
        atlas_file, atlas_labels_file = atlas.get_atlas_nifti(atlas_name)
        assert isinstance(atlas_file, str)
        assert isinstance(atlas_labels_file, str)
        assert os.path.isfile(atlas_file)
        assert os.path.isfile(atlas_labels_file)


def test_get_atlas_cifti():
    """Test xcp_d.utils.atlas.get_atlas_cifti."""
    atlas_names = atlas.get_atlas_names()
    for atlas_name in atlas_names:
        atlas_file, atlas_labels_file = atlas.get_atlas_cifti(atlas_name)
        assert isinstance(atlas_file, str)
        assert isinstance(atlas_labels_file, str)
        assert os.path.isfile(atlas_file)
        assert os.path.isfile(atlas_labels_file)
