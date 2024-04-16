"""Tests for the xcp_d.utils.doc module."""

import os

from xcp_d.utils import doc


def test_download_example_data(tmp_path_factory):
    """Test download_example_data."""
    tmpdir = tmp_path_factory.mktemp("test_download_example_data")
    example_data_dir = doc.download_example_data(out_dir=tmpdir)
    assert os.path.isdir(example_data_dir)
