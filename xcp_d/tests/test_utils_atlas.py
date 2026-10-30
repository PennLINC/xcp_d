"""Tests for the xcp_d.utils.atlas module."""

import json
from pathlib import Path

import pytest

from xcp_d.utils import atlas

_XCPD_ATLASES = str(Path.home() / '.cache' / 'xcp_d' / 'XCPDAtlases')


def test_get_atlas_names():
    """Test xcp_d.utils.atlas.select_atlases."""
    selected_atlases = atlas.select_atlases(atlases=['4S156Parcels', '4S256Parcels'], subset='all')
    assert isinstance(selected_atlases, list)
    assert all(isinstance(name, str) for name in selected_atlases)
    assert len(selected_atlases) == 2


def test_collect_atlases(datasets, caplog, tmp_path_factory):
    """Test xcp_d.utils.atlas.collect_atlases."""
    schaefer_dset = datasets['schaefer100']

    atlas_datasets = {
        'xcpdatlases': _XCPD_ATLASES,
    }
    atlas_cache = atlas.collect_atlases(
        datasets=atlas_datasets,
        atlases=['Gordon', 'Schaefer100'],
        file_format='nifti',
        bids_filters={},
    )
    assert 'Gordon' in atlas_cache
    assert 'Schaefer100' not in atlas_cache
    assert 'No atlas images found for Schaefer100' in caplog.text

    # Add the schaefer dataset
    atlas_datasets['schaefer100'] = schaefer_dset
    atlas_cache = atlas.collect_atlases(
        datasets=atlas_datasets,
        atlases=['Gordon', 'Schaefer100'],
        file_format='nifti',
        bids_filters={},
    )
    assert 'Gordon' in atlas_cache
    assert 'Schaefer100' in atlas_cache

    # Skip over the schaefer dataset
    atlas_datasets['schaefer100'] = schaefer_dset
    atlas_cache = atlas.collect_atlases(
        datasets=atlas_datasets,
        atlases=['Gordon'],
        file_format='cifti',
        bids_filters={},
    )
    assert 'Gordon' in atlas_cache
    assert 'Schaefer100' not in atlas_cache

    # Add a duplicate atlas
    atlas_datasets['duplicate'] = _XCPD_ATLASES
    with pytest.raises(ValueError, match="Multiple datasets contain the same atlas 'Gordon'"):
        atlas.collect_atlases(
            datasets=atlas_datasets,
            atlases=['Gordon'],
            file_format='nifti',
            bids_filters={},
        )

    # Create a dataset that has atlases, but is missing information
    tmpdir = tmp_path_factory.mktemp('test_collect_atlases')
    # Make the dataset_description.json
    with open(tmpdir / 'dataset_description.json', 'w') as fo:
        json.dump({'DatasetType': 'derivative', 'BIDSVersion': '1.9.0', 'Name': 'Test'}, fo)

    # Create fake atlas file
    (tmpdir / 'tpl-MNI152NLin6Asym').mkdir()
    (
        tmpdir / 'tpl-MNI152NLin6Asym' / 'tpl-MNI152NLin6Asym_atlas-TEST_res-01_dseg.nii.gz'
    ).write_text('test')

    # First there's an image, but no TSV or metadata
    with pytest.raises(FileNotFoundError, match='No TSV file found for'):
        atlas.collect_atlases(
            datasets={'test': tmpdir},
            atlases=['TEST'],
            file_format='nifti',
            bids_filters={},
        )

    # Now there's an image and a TSV, but the TSV doesn't have a "name" column
    with open(
        tmpdir / 'tpl-MNI152NLin6Asym' / 'tpl-MNI152NLin6Asym_atlas-TEST_res-01_dseg.tsv', 'w'
    ) as fo:
        fo.write('index\n1\n')

    with pytest.raises(ValueError, match="'name' column not found"):
        atlas.collect_atlases(
            datasets={'test': tmpdir},
            atlases=['TEST'],
            file_format='nifti',
            bids_filters={},
        )

    # Now there's an image and a TSV, but the TSV doesn't have an "index" column
    with open(
        tmpdir / 'tpl-MNI152NLin6Asym' / 'tpl-MNI152NLin6Asym_atlas-TEST_res-01_dseg.tsv', 'w'
    ) as fo:
        fo.write('name\ntest\n')

    with pytest.raises(ValueError, match="'index' column not found"):
        atlas.collect_atlases(
            datasets={'test': tmpdir},
            atlases=['TEST'],
            file_format='nifti',
            bids_filters={},
        )

    # Now there's an image, a TSV, and metadata
    with open(
        tmpdir / 'tpl-MNI152NLin6Asym' / 'tpl-MNI152NLin6Asym_atlas-TEST_res-01_dseg.tsv', 'w'
    ) as fo:
        fo.write('index\tname\n1\ttest\n')

    atlas_cache = atlas.collect_atlases(
        datasets={'test': tmpdir},
        atlases=['TEST'],
        file_format='nifti',
        bids_filters={},
    )
    assert 'TEST' in atlas_cache
