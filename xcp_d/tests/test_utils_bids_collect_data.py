"""Tests for the xcp_d.utils.bids module."""

import os

import pytest
from bids.layout import BIDSLayout
from niworkflows.utils.testing import generate_bids_skeleton

import xcp_d.utils.bids as xbids
from xcp_d.data import load as load_data


def test_collect_data_ds001419(datasets):
    """Test the collect_data function."""
    bids_dir = datasets['ds001419']
    layout = BIDSLayout(bids_dir, validate=False)

    # NIFTI workflow, but also get a BIDSLayout
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='fmriprep',
        participant_label='01',
        bids_filters=None,
        file_format='nifti',
    )

    assert len(subj_data['bold']) == 4
    assert 'space-MNI152NLin6Asym' in subj_data['bold'][0]
    assert os.path.basename(subj_data['t1w']) == 'sub-01_desc-preproc_T1w.nii.gz'
    assert 'space-' not in subj_data['t1w']
    assert 'to-MNI152NLin6Asym' in subj_data['anat_to_template_xfm']
    assert 'from-MNI152NLin6Asym' in subj_data['template_to_anat_xfm']

    # CIFTI workflow
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='fmriprep',
        participant_label='01',
        bids_filters={'bold': {'task': 'rest'}},
        file_format='cifti',
    )

    assert len(subj_data['bold']) == 1
    assert 'space-fsLR' in subj_data['bold'][0]
    assert 'space-' not in subj_data['t1w']
    assert os.path.basename(subj_data['t1w']) == 'sub-01_desc-preproc_T1w.nii.gz'
    assert 'to-MNI152NLin6Asym' in subj_data['anat_to_template_xfm']
    assert 'from-MNI152NLin6Asym' in subj_data['template_to_anat_xfm']


def test_collect_data_nibabies(datasets):
    """Test the collect_data function."""
    bids_dir = datasets['nibabies']
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    cohort_files = layout.get(subject='01', cohort='1', space='MNIInfant', suffix='boldref')
    assert len(cohort_files) > 0

    # NIFTI workflow
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='fmriprep',
        participant_label='01',
        bids_filters=None,
        file_format='nifti',
    )

    assert len(subj_data['bold']) == 1
    assert 'space-MNIInfant' in subj_data['bold'][0]
    assert 'cohort-1' in subj_data['bold'][0]
    assert os.path.basename(subj_data['t1w']) == 'sub-01_ses-1mo_run-001_desc-preproc_T1w.nii.gz'
    assert 'space-' not in subj_data['t1w']
    assert 'to-MNIInfant' in subj_data['anat_to_template_xfm']
    assert 'from-MNIInfant' in subj_data['template_to_anat_xfm']

    # CIFTI workflow
    with pytest.raises(FileNotFoundError):
        subj_data = xbids.collect_data(
            layout=layout,
            input_type='fmriprep',
            participant_label='01',
            bids_filters=None,
            file_format='cifti',
        )


def test_collect_data_nibabies_t1w_only(tmp_path_factory, caplog):
    """Test that nibabies collects T1w when T2w is absent."""
    skeleton = load_data('tests/skeletons/nibabies_t1w_only.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_t1w_only') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is not None
    assert subj_data['t2w'] is None
    assert 'T1w found, but no T2w. Enabling T1w-only processing.' in caplog.text


def test_collect_data_nibabies_t2w_only(tmp_path_factory, caplog):
    """Test that nibabies collects T2w when T1w is absent and T2w is present."""
    skeleton = load_data('tests/skeletons/nibabies_t2w_only.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_t2w_only') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is None
    assert subj_data['t2w'] is not None
    assert 'T2w found, but no T1w. Enabling T2w-only processing.' in caplog.text


def test_collect_data_nibabies_no_t1w_t2w(tmp_path_factory, caplog):
    """Test that nibabies raises an error when T1w and T2w are absent."""
    skeleton = load_data('tests/skeletons/nibabies_no_t1w_t2w.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_no_t1w_t2w') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    with pytest.raises(FileNotFoundError, match='No T1w or T2w files found'):
        xbids.collect_data(
            layout=layout,
            input_type='nibabies',
            participant_label='01',
            bids_filters=None,
            file_format='cifti',
        )


def test_collect_data_nibabies_ignore_t2w(tmp_path_factory, caplog):
    """Test collect_data.

    Ensure that XCP-D does not collect T2w when T1w is present, no T1w-space T2w is available,
    and no T2w-to-T1w transform is available.

    This differs from "ignore_t1w" based on the transforms that are available.
    """
    skeleton = load_data('tests/skeletons/nibabies_ignore_t2w.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_ignore_t2w') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is not None
    assert subj_data['t2w'] is None
    assert 'Both T1w and T2w found. Checking for T1w-space T2w.' in caplog.text
    assert 'No T1w-space T2w found. Checking for T2w-space T1w.' in caplog.text
    assert 'No T2w-space T1w found. Attempting T2w-primary processing.' in caplog.text
    assert 'Neither T2w-to-template, nor T2w-to-T1w, transform found.' in caplog.text


def test_collect_data_nibabies_t2w_to_t1w(tmp_path_factory, caplog):
    """Test collect_data.

    Ensure that XCP-D collects T2w when T1w is present, no T1w-space T2w is available,
    but a T2w-to-T1w transform is available.

    This differs from "ignore_t1w" based on the transforms that are available.
    """
    skeleton = load_data('tests/skeletons/nibabies_t2w_to_t1w.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_t2w_to_t1w') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is not None
    assert subj_data['t2w'] is None
    assert 'Both T1w and T2w found. Checking for T1w-space T2w.' in caplog.text
    assert 'No T1w-space T2w found. Checking for T2w-space T1w.' in caplog.text
    assert 'No T2w-space T1w found. Attempting T2w-primary processing.' in caplog.text
    # assert 'Neither T2w-to-template, nor T2w-to-T1w, transform found.' in caplog.text


def test_collect_data_nibabies_ignore_t1w(tmp_path_factory, caplog):
    """Test collect_data.

    Ensure that XCP-D does not collect T1w when T2w is present, no T2w-space T1w is available,
    and no T1w-to-T2w transform is available.

    This differs from "ignore_t2w" based on the transforms that are available.
    """
    skeleton = load_data('tests/skeletons/nibabies_ignore_t1w.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_ignore_t1w') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is None
    assert subj_data['t2w'] is not None
    assert 'Both T1w and T2w found. Checking for T1w-space T2w.' in caplog.text
    assert 'No T1w-space T2w found. Checking for T2w-space T1w.' in caplog.text
    assert 'No T2w-space T1w found. Attempting T2w-primary processing.' in caplog.text
    assert 'T2w-to-template transform found, but no T1w-to-T2w transform found.' in caplog.text


def test_collect_data_nibabies_t1w_to_t2w(tmp_path_factory, caplog):
    """Test collect_data.

    Ensure that XCP-D collects T1w when T2w is present, no T2w-space T1w is available,
    but a T1w-to-T2w transform is available.

    This differs from "ignore_t2w" based on the transforms that are available.
    """
    skeleton = load_data('tests/skeletons/nibabies_t1w_to_t2w.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_t1w_to_t2w') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is None
    assert subj_data['t2w'] is not None
    assert 'Both T1w and T2w found. Checking for T1w-space T2w.' in caplog.text
    assert 'No T1w-space T2w found. Checking for T2w-space T1w.' in caplog.text
    assert 'No T2w-space T1w found. Attempting T2w-primary processing.' in caplog.text
    # assert 'T2w-to-template transform found, but no T1w-to-T2w transform found.' in caplog.text


def test_collect_data_nibabies_t1wspace_t2w(tmp_path_factory, caplog):
    """Test that nibabies collects T1w and T2w when T1w-space T2w is present.

    This differs from "ignore_t2w" in that there's a space-T1w_desc-preproc_T2w.
    """
    skeleton = load_data('tests/skeletons/nibabies_t1wspace_t2w.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_t1wspace_t2w') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is not None
    assert subj_data['t2w'] is not None
    assert 'space-T1w' in subj_data['t2w']
    assert 'Both T1w and T2w found. Checking for T1w-space T2w.' in caplog.text
    assert 'T1w-space T2w found. Processing anatomical images in T1w space.' in caplog.text


def test_collect_data_nibabies_t2wspace_t1w(tmp_path_factory, caplog):
    """Test that nibabies collects T1w and T2w when T2w-space T1w is present.

    This differs from "ignore_t1w" in that there's a space-T2w_desc-preproc_T1w.
    """
    skeleton = load_data('tests/skeletons/nibabies_t2wspace_t1w.yml')
    bids_dir = tmp_path_factory.mktemp('test_collect_data_nibabies_t2wspace_t1w') / 'bids'
    generate_bids_skeleton(str(bids_dir), str(skeleton))
    xcp_d_config = str(load_data('xcp_d_bids_config2.json'))
    layout = BIDSLayout(
        bids_dir,
        validate=False,
        config=['bids', 'derivatives', xcp_d_config],
    )
    subj_data = xbids.collect_data(
        layout=layout,
        input_type='nibabies',
        participant_label='01',
        bids_filters=None,
        file_format='cifti',
    )
    assert subj_data['t1w'] is not None
    assert subj_data['t2w'] is not None
    assert 'space-T2w' in subj_data['t1w']
    assert 'Both T1w and T2w found. Checking for T1w-space T2w.' in caplog.text
    assert 'No T1w-space T2w found. Checking for T2w-space T1w.' in caplog.text
    assert 'T2w-space T1w found. Processing anatomical images in T2w space.' in caplog.text
