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

    with pytest.raises(FileNotFoundError, match="DNE"):
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

    with pytest.raises(FileNotFoundError, match="DNE"):
        atlas.get_atlas_cifti("tofail")


def test_copy_atlas(tmp_path_factory):
    """Test xcp_d.utils.atlas.copy_atlas."""
    tmpdir = tmp_path_factory.mktemp("test_copy_atlas")
    # NIfTI
    atlas_file, _, _ = atlas.get_atlas_nifti("Gordon")
    name_source = "sub-01_task-A_run-01_space-MNI152NLin2009cAsym_res-2_desc-z_bold.nii.gz"
    out_file = atlas.copy_atlas(
        name_source=name_source, in_file=atlas_file, output_dir=tmpdir, atlas="Y"
    )
    assert os.path.isfile(out_file)
    assert os.path.basename(out_file) == "space-MNI152NLin2009cAsym_atlas-Y_res-2_dseg.nii.gz"

    # CIFTI
    atlas_file, _, _ = atlas.get_atlas_cifti("Gordon")
    name_source = "sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    out_file = atlas.copy_atlas(
        name_source=name_source, in_file=atlas_file, output_dir=tmpdir, atlas="Y"
    )
    assert os.path.isfile(out_file)
    assert os.path.basename(out_file) == "space-fsLR_atlas-Y_den-91k_dseg.dlabel.nii"
