"""Tests for xcp_d.ingression.hcpya."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from xcp_d.ingression import hcpya


def test_convert_hcp2bids_no_subjects_raises(tmp_path):
    """convert_hcp2bids raises ValueError when no subjects found."""
    in_dir = tmp_path / 'hcp_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with pytest.raises(ValueError, match='No subject found'):
        hcpya.convert_hcp2bids(str(in_dir), str(out_dir), participant_ids=None)


def test_convert_hcp2bids_with_participant_ids_calls_single_subject(tmp_path):
    """convert_hcp2bids calls convert_hcp_to_bids_single_subject for each ID."""
    in_dir = tmp_path / 'hcp_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(hcpya, 'convert_hcp_to_bids_single_subject') as mock_single:
        result = hcpya.convert_hcp2bids(
            str(in_dir), str(out_dir), participant_ids=['sub-01', 'sub-02']
        )
    assert result == ['sub-01', 'sub-02']
    assert mock_single.call_count == 2
    mock_single.assert_any_call(
        in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='sub-01'
    )
    mock_single.assert_any_call(
        in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='sub-02'
    )


def test_convert_hcp2bids_passes_participant_ids_through(tmp_path):
    """convert_hcp2bids returns the participant_ids list as given and passes to single-subject."""
    in_dir = tmp_path / 'hcp_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(hcpya, 'convert_hcp_to_bids_single_subject') as mock_single:
        result = hcpya.convert_hcp2bids(
            str(in_dir), str(out_dir), participant_ids=['01']
        )
    assert result == ['01']
    mock_single.assert_called_once_with(
        in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='01'
    )


def test_convert_hcp_to_bids_single_subject_asserts_inputs():
    """convert_hcp_to_bids_single_subject asserts in_dir, out_dir, sub_ent."""
    with pytest.raises(AssertionError):
        hcpya.convert_hcp_to_bids_single_subject(
            in_dir='/nonexistent',
            out_dir='/out',
            sub_ent='sub-01',
        )


def test_convert_hcp_to_bids_single_subject_skips_when_dataset_description_exists(
    tmp_path,
):
    """convert_hcp_to_bids_single_subject skips when dataset_description.json exists."""
    in_dir = tmp_path / 'in'
    in_dir.mkdir()
    sub_id = '01'
    (in_dir / sub_id).mkdir()
    mni = in_dir / sub_id / 'MNINonLinear'
    mni.mkdir()
    (mni / 'T1w.nii.gz').write_bytes(b'')
    fsaverage = mni / 'fsaverage_LR32k'
    fsaverage.mkdir()
    (fsaverage / '01.L.pial.32k_fs_LR.surf.gii').write_bytes(b'')
    (fsaverage / '01.R.pial.32k_fs_LR.surf.gii').write_bytes(b'')
    results = mni / 'Results'
    results.mkdir()
    task_dir = results / 'tfMRI_REST1_RL'
    task_dir.mkdir()
    (task_dir / 'SBRef_dc.nii.gz').write_bytes(b'')
    (task_dir / 'tfMRI_REST1_RL.nii.gz').write_bytes(b'')
    (task_dir / 'brainmask_fs.2.nii.gz').write_bytes(b'')
    (task_dir / 'brainmask_fs.2.0.nii.gz').write_bytes(b'')
    (task_dir / 'Movement_Regressors.txt').write_text('0 0 0 0 0 0\n')
    (task_dir / 'Movement_AbsoluteRMS.txt').write_text('0\n')
    (task_dir / 'tfMRI_REST1_RL_Atlas_MSMAll.dtseries.nii').write_bytes(b'')

    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    (out_dir / 'dataset_description.json').write_text('{}')

    with patch.object(hcpya, 'collect_anatomical_files') as mock_anat:
        with patch.object(hcpya, 'collect_meshes') as mock_mesh:
            with patch.object(hcpya, 'collect_morphs') as mock_morph:
                with patch.object(hcpya, 'collect_hcp_confounds'):
                    with patch.object(hcpya, 'plot_bbreg'):
                        with patch.object(hcpya, 'copy_files_in_dict'):
                            with patch.object(hcpya, 'nb') as mock_nb:
                                mock_img = mock_nb.load.return_value
                                mock_img.header.get_zooms.return_value = (
                                    2.0,
                                    2.0,
                                    2.0,
                                    2.0,
                                )
                                hcpya.convert_hcp_to_bids_single_subject(
                                    in_dir=str(in_dir),
                                    out_dir=str(out_dir),
                                    sub_ent='sub-01',
                                )
    mock_anat.assert_not_called()
    mock_mesh.assert_not_called()
    mock_morph.assert_not_called()
