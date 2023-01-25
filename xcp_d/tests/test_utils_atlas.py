"""Tests for the xcp_d.utils.atlas module."""
import os

import pytest

from xcp_d.utils import atlas


def test_get_atlas_names():
    """Check that get_atlas_names returns a list of strings."""
    atlas_names = atlas.get_atlas_names()
    assert isinstance(atlas_names, list)
    assert len(atlas_names) == 13
    assert all(isinstance(name, str) for name in atlas_names)


def test_get_atlas_file():
    """Check that get_atlas_file returns a filename."""
    for atlas_name in atlas.get_atlas_names():
        for is_cifti in [True, False]:
            atlas_file, node_labels_file = atlas.get_atlas_file(atlas_name, cifti=is_cifti)
            assert isinstance(atlas_file, str)
            assert os.path.isfile(atlas_file)
            assert isinstance(node_labels_file, str)
            assert os.path.isfile(node_labels_file)

    # Unknown atlases should raise RuntimeErrors.
    with pytest.raises(RuntimeError):
        atlas.get_atlas_file("fakeatlas", cifti=False)

    with pytest.raises(RuntimeError):
        atlas.get_atlas_file("fakeatlas", cifti=False)
