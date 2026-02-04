"""Tests for xcp_d.ingression.ukbiobank."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xcp_d.ingression import ukbiobank


def _make_ukb_skeleton(tmp_path, sub_id, ses_id='01'):
    """Build minimal UK Biobank-like directory for one subject/session."""
    in_dir = tmp_path / f'{sub_id}_{ses_id}_2_0'
    in_dir.mkdir(parents=True)
    (in_dir / 'fMRI').mkdir()
    (in_dir / 'fMRI' / 'rfMRI.ica').mkdir()
    (in_dir / 'fMRI' / 'rfMRI.ica' / 'mc').mkdir()
    (in_dir / 'fMRI' / 'rfMRI.ica' / 'reg').mkdir()
    (in_dir / 'T1').mkdir()

    ica = in_dir / 'fMRI' / 'rfMRI.ica'
    (ica / 'filtered_func_data_clean.nii.gz').write_bytes(b'')
    (ica / 'mask.nii.gz').write_bytes(b'')
    (ica / 'example_func.nii.gz').write_bytes(b'')
    (ica / 'mc' / 'prefiltered_func_data_mcf.par').write_text('0 0 0 0 0 0\n')
    (ica / 'mc' / 'prefiltered_func_data_mcf_abs.rms').write_text('0\n')
    (ica / 'reg' / 'example_func2standard_warp.nii.gz').write_bytes(b'')

    (in_dir / 'fMRI' / 'rfMRI.json').write_text(json.dumps({'RepetitionTime': 1.0}))
    (in_dir / 'T1' / 'T1_brain_to_MNI.nii.gz').write_bytes(b'')

    return in_dir


def test_convert_ukb2bids_two_subjects_both_converted_reproduces_issue_1471(tmp_path):
    """convert_ukb2bids converts all subjects; second subject not skipped (Issue #1471).

    When dataset_description.json is written at the end of the first subject's
    conversion, the second subject's conversion must not be skipped. Otherwise
    only one subject is converted (same bug as https://github.com/PennLINC/xcp_d/issues/1471).
    """
    in_root = tmp_path / 'ukb_in'
    in_root.mkdir()
    sub1_dir = _make_ukb_skeleton(in_root.parent, 'sub1', '01')
    sub1_dir.rename(in_root / sub1_dir.name)
    sub2_dir = _make_ukb_skeleton(in_root.parent, 'sub2', '01')
    sub2_dir.rename(in_root / sub2_dir.name)

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    # Stash real load_data and ApplyWarp; we mock them so conversion succeeds without FSL.
    from xcp_d.data import load as real_load
    from nipype.interfaces.fsl.preprocess import ApplyWarp

    def fake_load_data(name):
        if 'MNI152_T1_2mm' in name:
            p = tmp_path / 'template.nii.gz'
            p.write_bytes(b'')
            return str(p)
        if 'itkIdentityTransform' in name:
            return str(real_load.readable(name))
        return str(real_load.readable(name))

    call_count = [0]

    def fake_apply_warp_run(self, cwd=None):
        call_count[0] += 1
        out_file = tmp_path / f'warped_{call_count[0]}.nii.gz'
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_bytes(b'')
        result = MagicMock()
        result.outputs.out_file = str(out_file)
        return result

    with patch.object(ukbiobank, 'load_data', side_effect=fake_load_data):
        with patch.object(ApplyWarp, 'run', side_effect=fake_apply_warp_run):
            result = ukbiobank.convert_ukb2bids(
                str(in_root),
                str(out_dir),
                participant_ids=['sub1', 'sub2'],
                bids_filters={'bold': {'session': ['01']}},
            )

    assert 'sub1' in result
    assert 'sub2' in result

    # Both subjects must have been fully converted (anat + func outputs).
    sub1_out = out_dir / 'sub-sub1' / 'ses-01'
    sub2_out = out_dir / 'sub-sub2' / 'ses-01'
    assert sub1_out.is_dir(), 'sub-sub1/ses-01 output dir should exist'
    assert sub2_out.is_dir(), 'sub-sub2/ses-01 output dir should exist'

    for sub_label, sub_out in [('sub-sub1', sub1_out), ('sub-sub2', sub2_out)]:
        anat_dir = sub_out / 'anat'
        func_dir = sub_out / 'func'
        assert (anat_dir / f'{sub_label}_ses-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.txt').exists()
        assert (func_dir / f'{sub_label}_ses-01_task-rest_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz').exists()
        assert (sub_out / f'{sub_label}_ses-01_scans.tsv').exists()


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
