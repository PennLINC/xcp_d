"""Tests for xcp_d.ingression.abcdbids."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from xcp_d.ingression import abcdbids


def test_convert_dcan2bids_no_subjects_raises(tmp_path):
    """convert_dcan2bids raises ValueError when no subjects found."""
    in_dir = tmp_path / 'dcan_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with pytest.raises(ValueError, match='No subject found'):
        abcdbids.convert_dcan2bids(str(in_dir), str(out_dir), participant_ids=None)


def test_convert_dcan2bids_discovers_subjects(tmp_path):
    """convert_dcan2bids discovers sub* directories when participant_ids is None."""
    in_dir = tmp_path / 'dcan_in'
    in_dir.mkdir()
    (in_dir / 'sub-01').mkdir()
    (in_dir / 'sub-02').mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(abcdbids, 'convert_dcan_to_bids_single_subject') as mock_single:
        result = abcdbids.convert_dcan2bids(str(in_dir), str(out_dir))
    assert result == ['sub-01', 'sub-02']
    assert mock_single.call_count == 2


def test_convert_dcan2bids_with_participant_ids(tmp_path):
    """convert_dcan2bids calls single-subject for each given participant."""
    in_dir = tmp_path / 'dcan_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(abcdbids, 'convert_dcan_to_bids_single_subject') as mock_single:
        result = abcdbids.convert_dcan2bids(
            str(in_dir), str(out_dir), participant_ids=['sub-01']
        )
    assert result == ['sub-01']
    mock_single.assert_called_once_with(
        in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='sub-01'
    )


def test_convert_dcan2bids_passes_participant_ids_through(tmp_path):
    """convert_dcan2bids returns the participant_ids list as given and passes to single-subject."""
    in_dir = tmp_path / 'dcan_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(abcdbids, 'convert_dcan_to_bids_single_subject') as mock_single:
        result = abcdbids.convert_dcan2bids(
            str(in_dir), str(out_dir), participant_ids=['01']
        )
    assert result == ['01']
    mock_single.assert_called_once_with(
        in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='01'
    )


def test_convert_dcan_to_bids_single_subject_asserts_inputs():
    """convert_dcan_to_bids_single_subject asserts in_dir exists and types."""
    with pytest.raises(AssertionError):
        abcdbids.convert_dcan_to_bids_single_subject(
            in_dir='/nonexistent',
            out_dir='/out',
            sub_ent='sub-01',
        )


def test_convert_dcan_to_bids_single_subject_raises_when_no_sessions(tmp_path):
    """convert_dcan_to_bids_single_subject raises when no session folders."""
    in_dir = tmp_path / 'in'
    in_dir.mkdir()
    (in_dir / 'sub-01').mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with pytest.raises(FileNotFoundError, match='No session volumes'):
        abcdbids.convert_dcan_to_bids_single_subject(
            in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='sub-01'
        )


def test_convert_dcan_to_bids_single_subject_skips_when_dataset_description_exists(
    tmp_path,
):
    """convert_dcan_to_bids_single_subject skips when dataset_description.json exists."""
    in_dir = tmp_path / 'in'
    in_dir.mkdir()
    sub_dir = in_dir / 'sub-01'
    sub_dir.mkdir()
    ses_dir = sub_dir / 'ses-01'
    ses_dir.mkdir()
    files_dir = ses_dir / 'files' / 'MNINonLinear'
    files_dir.mkdir(parents=True)
    (files_dir / 'T1w.nii.gz').write_bytes(b'')
    results = files_dir / 'Results'
    results.mkdir()
    task_dir = results / 'ses-01_task-rest_run-1'
    task_dir.mkdir()
    (task_dir / 'ses-01_task-rest_run-1_SBRef.nii.gz').write_bytes(b'')
    (task_dir / 'ses-01_task-rest_run-1.nii.gz').write_bytes(b'')
    (task_dir / 'brainmask_fs.2.0.nii.gz').write_bytes(b'')
    (task_dir / 'Movement_Regressors.txt').write_text('0 0 0 0 0 0\n')
    (task_dir / 'Movement_AbsoluteRMS.txt').write_text('0\n')
    (task_dir / 'ses-01_task-rest_run-1_Atlas.dtseries.nii').write_bytes(b'')
    fsaverage = files_dir / 'fsaverage_LR32k'
    fsaverage.mkdir()
    (fsaverage / '01.L.pial.32k_fs_LR.surf.gii').write_bytes(b'')
    (fsaverage / '01.R.pial.32k_fs_LR.surf.gii').write_bytes(b'')
    (files_dir / 'vent_2mm_01_mask_eroded.nii.gz').write_bytes(b'')
    (files_dir / 'wm_2mm_01_mask_eroded.nii.gz').write_bytes(b'')

    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    (out_dir / 'dataset_description.json').write_text('{}')

    with patch.object(abcdbids, 'collect_anatomical_files') as mock_anat:
        with patch.object(abcdbids, 'collect_meshes') as mock_mesh:
            with patch.object(abcdbids, 'collect_morphs') as mock_morph:
                with patch.object(abcdbids, 'collect_hcp_confounds'):
                    with patch.object(abcdbids, 'plot_bbreg'):
                        with patch.object(abcdbids, 'copy_files_in_dict'):
                            with patch.object(abcdbids, 'nb') as mock_nb:
                                mock_img = mock_nb.load.return_value
                                mock_img.header.get_zooms.return_value = (
                                    2.0,
                                    2.0,
                                    2.0,
                                    2.0,
                                )
                                abcdbids.convert_dcan_to_bids_single_subject(
                                    in_dir=str(in_dir),
                                    out_dir=str(out_dir),
                                    sub_ent='sub-01',
                                )
    mock_anat.assert_not_called()
    mock_mesh.assert_not_called()
    mock_morph.assert_not_called()
