"""Tests for the xcp_d.utils.atlas module."""

import os

import pytest

from xcp_d.utils import atlas


def test_get_atlas_names():
    """Test xcp_d.utils.atlas.select_atlases."""
    selected_atlases = atlas.select_atlases(atlases=["4S156Parcels", "4S256Parcels"], subset="all")
    assert isinstance(selected_atlases, list)
    assert all(isinstance(name, str) for name in selected_atlases)
    assert len(selected_atlases) == 2


def test_get_atlas_nifti():
    """Test xcp_d.utils.atlas.get_atlas_nifti."""
    selected_atlases = atlas.select_atlases(
        atlases=["4S156Parcels", "4S256Parcels", "Tian"],
        subset="all",
    )
    for selected_atlas in selected_atlases:
        atlas_file, atlas_labels_file, metadata_file = atlas.get_atlas_nifti(selected_atlas)
        assert isinstance(atlas_file, str)
        assert isinstance(atlas_labels_file, str)
        assert isinstance(metadata_file, str)
        assert os.path.isfile(atlas_file)
        assert os.path.isfile(atlas_labels_file)
        assert os.path.isfile(metadata_file)

    with pytest.raises(FileNotFoundError, match="DNE"):
        atlas.get_atlas_nifti("tofail")


def test_get_atlas_cifti():
    """Test xcp_d.utils.atlas.get_atlas_cifti."""
    selected_atlases = atlas.select_atlases(
        atlases=["4S156Parcels", "4S256Parcels", "MIDB", "MyersLabonte", "Tian"],
        subset="all",
    )
    for selected_atlas in selected_atlases:
        atlas_file, atlas_labels_file, metadata_file = atlas.get_atlas_cifti(selected_atlas)
        assert isinstance(atlas_file, str)
        assert isinstance(atlas_labels_file, str)
        assert isinstance(metadata_file, str)
        assert os.path.isfile(atlas_file)
        assert os.path.isfile(atlas_labels_file)
        assert os.path.isfile(metadata_file)

    with pytest.raises(FileNotFoundError, match="DNE"):
        atlas.get_atlas_cifti("tofail")
