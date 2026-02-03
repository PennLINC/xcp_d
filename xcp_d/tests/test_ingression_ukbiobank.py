"""Tests for xcp_d.ingression.ukbiobank."""

from unittest.mock import patch

import pytest

from xcp_d.ingression import ukbiobank


def test_convert_ukb2bids_no_subjects_raises(tmp_path):
    """convert_ukb2bids raises ValueError when no subjects found."""
    in_dir = tmp_path / 'ukb_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with pytest.raises(ValueError, match='No subject found'):
        ukbiobank.convert_ukb2bids(str(in_dir), str(out_dir), participant_ids=None)


def test_convert_ukb2bids_discovers_subjects(tmp_path):
    """convert_ukb2bids discovers subject dirs matching *_*_2_0."""
    in_dir = tmp_path / 'ukb_in'
    in_dir.mkdir()
    (in_dir / 'sub1_01_2_0').mkdir()
    (in_dir / 'sub2_02_2_0').mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(ukbiobank, 'convert_ukb_to_bids_single_subject') as mock_single:
        result = ukbiobank.convert_ukb2bids(str(in_dir), str(out_dir))
    assert 'sub1' in result
    assert 'sub2' in result
    assert mock_single.call_count == 2


def test_convert_ukb2bids_with_participant_ids(tmp_path):
    """convert_ukb2bids calls single-subject for given participants and sessions."""
    in_dir = tmp_path / 'ukb_in'
    in_dir.mkdir()
    sub_ses = in_dir / 'sub1_01_2_0'
    sub_ses.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(ukbiobank, 'convert_ukb_to_bids_single_subject') as mock_single:
        result = ukbiobank.convert_ukb2bids(
            str(in_dir),
            str(out_dir),
            participant_ids=['sub1'],
            bids_filters={'bold': {'session': ['01']}},
        )
    assert result == ['sub1']
    mock_single.assert_called()
    call_kw = mock_single.call_args[1]
    assert call_kw['sub_id'] == 'sub1'
    assert call_kw['ses_id'] == '01'


def test_convert_ukb_to_bids_single_subject_asserts_inputs():
    """convert_ukb_to_bids_single_subject asserts in_dir exists."""
    with pytest.raises(AssertionError):
        ukbiobank.convert_ukb_to_bids_single_subject(
            in_dir='/nonexistent',
            out_dir='/out',
            sub_id='01',
            ses_id='01',
        )


def test_convert_ukb_to_bids_single_subject_raises_when_bold_missing(tmp_path):
    """convert_ukb_to_bids_single_subject raises when BOLD file missing."""
    in_dir = tmp_path / 'in'
    in_dir.mkdir()
    (in_dir / 'fMRI').mkdir()
    (in_dir / 'fMRI' / 'rfMRI.ica').mkdir()
    (in_dir / 'T1').mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with pytest.raises(AssertionError, match='File DNE'):
        ukbiobank.convert_ukb_to_bids_single_subject(
            in_dir=str(in_dir),
            out_dir=str(out_dir),
            sub_id='01',
            ses_id='01',
        )
