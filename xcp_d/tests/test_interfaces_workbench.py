"""Tests for Workbench interfaces."""

import pytest
from PIL import Image

from xcp_d.interfaces.workbench import _check_image_is_nonempty


def test_check_image_is_nonempty(tmp_path):
    """A uniform Workbench capture is treated as a failed render."""
    blank_file = tmp_path / 'blank.png'
    Image.new('RGB', (5, 5), 'black').save(blank_file)

    with pytest.raises(RuntimeError, match='empty scene image'):
        _check_image_is_nonempty(blank_file)

    image_file = tmp_path / 'image.png'
    image = Image.new('RGB', (5, 5), 'black')
    image.putpixel((2, 2), (255, 255, 255))
    image.save(image_file)

    _check_image_is_nonempty(image_file)
