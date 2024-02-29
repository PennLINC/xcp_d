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
    selected_atlases = atlas.select_atlases(atlases=["4S156Parcels", "4S256Parcels"], subset="all")
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
    selected_atlases = atlas.select_atlases(atlases=["4S156Parcels", "4S256Parcels"], subset="all")
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


def test_copy_atlas(tmp_path_factory):
    """Test xcp_d.utils.atlas.copy_atlas."""
    tmpdir = tmp_path_factory.mktemp("test_copy_atlas")
    os.makedirs(os.path.join(tmpdir, "xcp_d"), exist_ok=True)

    # NIfTI
    atlas_file, _, _ = atlas.get_atlas_nifti("Gordon")
    name_source = "sub-01_task-A_run-01_space-MNI152NLin2009cAsym_res-2_desc-z_bold.nii.gz"
    out_file = atlas.copy_atlas(
        name_source=name_source, in_file=atlas_file, output_dir=tmpdir, atlas="Y"
    )
    assert os.path.isfile(out_file)
    assert os.path.basename(out_file) == "atlas-Y_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz"

    # CIFTI
    atlas_file, atlas_labels_file, atlas_metadata_file = atlas.get_atlas_cifti("Gordon")
    name_source = "sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    out_file = atlas.copy_atlas(
        name_source=name_source, in_file=atlas_file, output_dir=tmpdir, atlas="Y"
    )
    assert os.path.isfile(out_file)
    assert os.path.basename(out_file) == "atlas-Y_space-fsLR_den-91k_dseg.dlabel.nii"

    # TSV
    name_source = "sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    out_file = atlas.copy_atlas(
        name_source=name_source, in_file=atlas_labels_file, output_dir=tmpdir, atlas="Y"
    )
    assert os.path.isfile(out_file)
    assert os.path.basename(out_file) == "atlas-Y_dseg.tsv"

    # JSON
    name_source = "sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    out_file = atlas.copy_atlas(
        name_source=name_source, in_file=atlas_metadata_file, output_dir=tmpdir, atlas="Y"
    )
    assert os.path.isfile(out_file)
    assert os.path.basename(out_file) == "atlas-Y_dseg.json"

    # Ensure that out_file isn't overwritten if it already exists
    fake_in_file = os.path.join(tmpdir, "fake.json")
    with open(fake_in_file, "w") as fo:
        fo.write("fake")

    out_file = atlas.copy_atlas(
        name_source=name_source, in_file=fake_in_file, output_dir=tmpdir, atlas="Y"
    )
    assert os.path.isfile(out_file)
    assert os.path.basename(out_file) == "atlas-Y_dseg.json"
    # The file should not be overwritten, so the contents shouldn't be "fake"
    with open(out_file, "r") as fo:
        assert fo.read() != "fake"
