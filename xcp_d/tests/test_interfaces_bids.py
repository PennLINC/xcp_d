"""Tests for xcp_d.interfaces.bids."""

import os

import pytest

from xcp_d.data import load as load_data
from xcp_d.interfaces import bids


def test_copy_atlas(tmp_path_factory):
    """Test xcp_d.interfaces.bids.CopyAtlas."""
    tmpdir = tmp_path_factory.mktemp('test_copy_atlas')
    os.makedirs(os.path.join(tmpdir, 'xcp_d'), exist_ok=True)

    # NIfTI
    atlas_info = {
        'image': load_data(
            'atlases/atlas-Gordon/atlas-Gordon_space-MNI152NLin6Asym_res-01_dseg.nii.gz'
        ),
        'labels': load_data('atlases/atlas-Gordon/atlas-Gordon_dseg.tsv'),
        'metadata': {'thing': 'stuff'},
        'dataset': 'xcpdatlases',
    }
    name_source = 'sub-01_task-A_run-01_space-MNI152NLin2009cAsym_res-2_desc-z_bold.nii.gz'
    copyatlas = bids.CopyAtlas(
        name_source=name_source,
        in_file=atlas_info['image'],
        output_dir=tmpdir,
        atlas='Y',
        meta_dict=atlas_info['metadata'],
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert os.path.isfile(result.outputs.out_file.replace('.nii.gz', '.json'))
    assert (
        os.path.basename(result.outputs.out_file)
        == 'atlas-Y_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz'
    )

    # Check that the NIfTI file raises an error if the resolution varies
    # Gordon atlas is 1mm, HCP is 2mm
    atlas_info_diff_affine = {
        'image': load_data('atlases/atlas-HCP/atlas-HCP_space-MNI152NLin6Asym_res-02_dseg.nii.gz'),
        'labels': load_data('atlases/atlas-HCP/atlas-HCP_dseg.tsv'),
        'metadata': {'thing': 'stuff'},
        'dataset': 'xcpdatlases',
    }
    with pytest.raises(ValueError, match='is different from the input file affine'):
        copyatlas = bids.CopyAtlas(
            name_source=name_source,
            in_file=atlas_info_diff_affine['image'],
            output_dir=tmpdir,
            atlas='Y',
        )
        copyatlas.run(cwd=tmpdir)

    # CIFTI
    atlas_info = {
        'image': load_data('atlases/atlas-Gordon/atlas-Gordon_space-fsLR_den-32k_dseg.dlabel.nii'),
        'labels': load_data('atlases/atlas-Gordon/atlas-Gordon_dseg.tsv'),
        'metadata': {'thing': 'stuff'},
        'dataset': 'xcpdatlases',
    }
    name_source = 'sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii'
    copyatlas = bids.CopyAtlas(
        name_source=name_source,
        in_file=atlas_info['image'],
        output_dir=tmpdir,
        atlas='Y',
        meta_dict=atlas_info['metadata'],
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert os.path.isfile(result.outputs.out_file.replace('.dlabel.nii', '.json'))
    assert (
        os.path.basename(result.outputs.out_file) == 'atlas-Y_space-fsLR_den-91k_dseg.dlabel.nii'
    )

    # TSV
    name_source = 'sub-01_task-imagery_run-01_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii'
    copyatlas = bids.CopyAtlas(
        name_source=name_source, in_file=atlas_info['labels'], output_dir=tmpdir, atlas='Y'
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert os.path.basename(result.outputs.out_file) == 'atlas-Y_dseg.tsv'

    # Ensure that out_file isn't overwritten if it already exists
    fake_in_file = os.path.join(tmpdir, 'fake.tsv')
    with open(fake_in_file, 'w') as fo:
        fo.write('fake')

    copyatlas = bids.CopyAtlas(
        name_source=name_source, in_file=fake_in_file, output_dir=tmpdir, atlas='Y'
    )
    result = copyatlas.run(cwd=tmpdir)
    assert os.path.isfile(result.outputs.out_file)
    assert os.path.basename(result.outputs.out_file) == 'atlas-Y_dseg.tsv'
    # The file should not be overwritten, so the contents shouldn't be "fake"
    with open(result.outputs.out_file) as fo:
        assert fo.read() != 'fake'
