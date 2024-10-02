"""Tests for the xcp_d.utils.atlas module."""

from xcp_d.utils import atlas


def test_get_atlas_names():
    """Test xcp_d.utils.atlas.select_atlases."""
    selected_atlases = atlas.select_atlases(atlases=["4S156Parcels", "4S256Parcels"], subset="all")
    assert isinstance(selected_atlases, list)
    assert all(isinstance(name, str) for name in selected_atlases)
    assert len(selected_atlases) == 2
