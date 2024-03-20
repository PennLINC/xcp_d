"""Tests for xcp_d.interfaces.bids."""

import os

import pytest

from xcp_d.interfaces import bids
from xcp_d.utils import atlas


def test_copy_atlas(tmp_path_factory):
    """Test xcp_d.interfaces.bids.CopyAtlas."""
    tmpdir = tmp_path_factory.mktemp("test_copy_atlas")
    os.makedirs(os.path.join(tmpdir, "xcp_d"), exist_ok=True)

    # NIfTI
    atlas_file, _, _ = atlas.get_atlas_nifti("Gordon")
    name_source = "sub-01_task-A_run-01_space-MNI152NLin2009cAsym_res-2_desc-z_bold.nii.gz"
    copyatlas = bids.CopyAtlas(
        name_source=name_source, in_file=atlas_file, output_dir=tmpdir, atlas="Y"
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert (
        os.path.basename(result.outputs.out_file)
        == "atlas-Y_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz"
    )

    # Check that the NIfTI file raises an error if the resolution varies
    # Gordon atlas is 1mm, HCP is 2mm
    atlas_file_diff_affine, _, _ = atlas.get_atlas_nifti("HCP")
    with pytest.raises(ValueError, match="is different from the input file affine"):
        copyatlas = bids.CopyAtlas(
            name_source=name_source,
            in_file=atlas_file_diff_affine,
            output_dir=tmpdir,
            atlas="Y",
        )
        copyatlas.run(cwd=tmpdir)

    # CIFTI
    atlas_file, atlas_labels_file, atlas_metadata_file = atlas.get_atlas_cifti("Gordon")
    name_source = "sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    copyatlas = bids.CopyAtlas(
        name_source=name_source, in_file=atlas_file, output_dir=tmpdir, atlas="Y"
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert (
        os.path.basename(result.outputs.out_file) == "atlas-Y_space-fsLR_den-91k_dseg.dlabel.nii"
    )

    # TSV
    name_source = "sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    copyatlas = bids.CopyAtlas(
        name_source=name_source, in_file=atlas_labels_file, output_dir=tmpdir, atlas="Y"
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert os.path.basename(result.outputs.out_file) == "atlas-Y_dseg.tsv"

    # JSON
    name_source = "sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii"
    copyatlas = bids.CopyAtlas(
        name_source=name_source, in_file=atlas_metadata_file, output_dir=tmpdir, atlas="Y"
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert os.path.basename(result.outputs.out_file) == "atlas-Y_dseg.json"

    # Ensure that out_file isn't overwritten if it already exists
    fake_in_file = os.path.join(tmpdir, "fake.json")
    with open(fake_in_file, "w") as fo:
        fo.write("fake")

    copyatlas = bids.CopyAtlas(
        name_source=name_source, in_file=fake_in_file, output_dir=tmpdir, atlas="Y"
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert os.path.basename(result.outputs.out_file) == "atlas-Y_dseg.json"
    # The file should not be overwritten, so the contents shouldn't be "fake"
    with open(result.outputs.out_file, "r") as fo:
        assert fo.read() != "fake"
