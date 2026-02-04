"""Tests for xcp_d.ingression.abcdbids."""

from pathlib import Path
from unittest.mock import patch

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from xcp_d.ingression import abcdbids

# Small imaging grid: 5x5x5 = 125 voxels; 5x5x5x4 = 500 voxels (under limit)
_SHAPE_3D = (5, 5, 5)
_SHAPE_4D = (5, 5, 5, 4)


def _write_minimal_nifti(path, shape, zooms=None, dtype=np.float32, mask=False):
    """Write a minimal NIfTI with optional zooms (for TR in 4D).

    If mask is True, at least one voxel is set to 1 so nilearn accepts it as a valid mask.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(shape, dtype=dtype)
    if mask:
        arr.flat[0] = 1
    if zooms is None:
        zooms = (2.0,) * len(shape)
    aff = np.diag([float(z) for z in zooms] + [1.0])[:4, :4]
    img = nb.Nifti1Image(arr, aff)
    img.to_filename(str(path))


def _make_dcan_skeleton(tmp_path, sub_id='01', ses_id='01'):
    """Build minimal DCAN/ABCD-like directory tree with small NIFTIs."""
    in_dir = tmp_path / 'dcan_in'
    in_dir.mkdir()
    session_dir = in_dir / 'sub-01' / 'ses-01'
    mni = session_dir / 'files' / 'MNINonLinear'
    mni.mkdir(parents=True)
    fsaverage = mni / 'fsaverage_LR32k'
    fsaverage.mkdir()
    results = mni / 'Results'
    results.mkdir()
    task_name = 'ses-01_task-rest_run-1'
    task_dir = results / task_name
    task_dir.mkdir()

    _write_minimal_nifti(mni / 'T1w.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'ribbon.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)
    _write_minimal_nifti(mni / f'vent_2mm_{sub_id}_mask_eroded.nii.gz', _SHAPE_3D, mask=True)
    _write_minimal_nifti(mni / f'wm_2mm_{sub_id}_mask_eroded.nii.gz', _SHAPE_3D, mask=True)

    for hemi, surf in [('L', 'pial'), ('R', 'pial'), ('L', 'white'), ('R', 'white')]:
        (fsaverage / f'{sub_id}.{hemi}.{surf}.32k_fs_LR.surf.gii').write_bytes(b'')

    _write_minimal_nifti(task_dir / f'{task_name}_SBRef.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(
        task_dir / f'{task_name}.nii.gz',
        _SHAPE_4D,
        zooms=(2.0, 2.0, 2.0, 2.0),
    )
    _write_minimal_nifti(task_dir / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)

    n_vols = _SHAPE_4D[-1]
    mvreg = '\n'.join('0 0 0 0 0 0' for _ in range(n_vols)) + '\n'
    (task_dir / 'Movement_Regressors.txt').write_text(mvreg)
    (task_dir / 'Movement_AbsoluteRMS.txt').write_text(
        '\n'.join('0.5' for _ in range(n_vols)) + '\n'
    )
    (task_dir / f'{task_name}_Atlas.dtseries.nii').write_bytes(b'')

    return in_dir


def _add_dcan_subject(in_dir, sub_id, ses_id='01'):
    """Add a second DCAN subject to an existing in_dir (same layout as _make_dcan_skeleton)."""
    session_dir = in_dir / f'sub-{sub_id}' / f'ses-{ses_id}'
    mni = session_dir / 'files' / 'MNINonLinear'
    mni.mkdir(parents=True)
    fsaverage = mni / 'fsaverage_LR32k'
    fsaverage.mkdir()
    results = mni / 'Results'
    results.mkdir()
    task_name = f'ses-{ses_id}_task-rest_run-1'
    task_dir = results / task_name
    task_dir.mkdir()

    _write_minimal_nifti(mni / 'T1w.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'ribbon.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)
    _write_minimal_nifti(mni / f'vent_2mm_{sub_id}_mask_eroded.nii.gz', _SHAPE_3D, mask=True)
    _write_minimal_nifti(mni / f'wm_2mm_{sub_id}_mask_eroded.nii.gz', _SHAPE_3D, mask=True)

    for hemi, surf in [('L', 'pial'), ('R', 'pial'), ('L', 'white'), ('R', 'white')]:
        (fsaverage / f'{sub_id}.{hemi}.{surf}.32k_fs_LR.surf.gii').write_bytes(b'')

    _write_minimal_nifti(task_dir / f'{task_name}_SBRef.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(
        task_dir / f'{task_name}.nii.gz',
        _SHAPE_4D,
        zooms=(2.0, 2.0, 2.0, 2.0),
    )
    _write_minimal_nifti(task_dir / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)

    n_vols = _SHAPE_4D[-1]
    mvreg = '\n'.join('0 0 0 0 0 0' for _ in range(n_vols)) + '\n'
    (task_dir / 'Movement_Regressors.txt').write_text(mvreg)
    (task_dir / 'Movement_AbsoluteRMS.txt').write_text(
        '\n'.join('0.5' for _ in range(n_vols)) + '\n'
    )
    (task_dir / f'{task_name}_Atlas.dtseries.nii').write_bytes(b'')


def test_convert_dcan2bids_two_subjects_both_converted_reproduces_issue_1471(tmp_path):
    """convert_dcan2bids converts all subjects; second subject not skipped (Issue #1471).

    When dataset_description.json is written at the end of the first subject's
    conversion, the second subject's conversion must not be skipped. Otherwise
    only one subject is converted (same bug as https://github.com/PennLINC/xcp_d/issues/1471).
    """
    in_dir = _make_dcan_skeleton(tmp_path, sub_id='01', ses_id='01')
    _add_dcan_subject(in_dir, '02', ses_id='01')

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    result = abcdbids.convert_dcan2bids(
        str(in_dir), str(out_dir), participant_ids=['sub-01', 'sub-02']
    )

    assert result == ['sub-01', 'sub-02']

    sub01_dir = out_dir / 'sub-01'
    sub02_dir = out_dir / 'sub-02'
    assert sub01_dir.is_dir(), 'sub-01 output dir should exist'
    assert sub02_dir.is_dir(), 'sub-02 output dir should exist'

    for sub_ent, sub_dir in [('sub-01', sub01_dir), ('sub-02', sub02_dir)]:
        ses_dir = sub_dir / 'ses-01'
        anat_dir = ses_dir / 'anat'
        func_dir = ses_dir / 'func'
        xfm1 = anat_dir / f'{sub_ent}_ses-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.txt'
        xfm2 = anat_dir / f'{sub_ent}_ses-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.txt'
        assert xfm1.exists()
        assert xfm2.exists()
        prefix = f'{sub_ent}_ses-01_task-rest_run-1'
        bold_path = func_dir / f'{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
        assert bold_path.exists()
        assert (func_dir / f'{prefix}_desc-confounds_timeseries.tsv').exists()


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
        result = abcdbids.convert_dcan2bids(str(in_dir), str(out_dir), participant_ids=['sub-01'])
    assert result == ['sub-01']
    mock_single.assert_called_once_with(in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='sub-01')


def test_convert_dcan2bids_passes_participant_ids_through(tmp_path):
    """convert_dcan2bids returns the participant_ids list as given and passes to single-subject."""
    in_dir = tmp_path / 'dcan_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(abcdbids, 'convert_dcan_to_bids_single_subject') as mock_single:
        result = abcdbids.convert_dcan2bids(str(in_dir), str(out_dir), participant_ids=['01'])
    assert result == ['01']
    mock_single.assert_called_once_with(in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='01')


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


def test_convert_dcan_to_bids_single_subject_runs_even_when_dataset_description_exists(
    tmp_path,
):
    """convert_dcan_to_bids_single_subject runs conversion even if dataset_description.json exists.

    Dataset description is written by the batch converter after all subjects; the
    single-subject function must not skip when that file is already present.
    """
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
    mock_anat.assert_called_once()
    mock_mesh.assert_called_once()
    mock_morph.assert_called_once()


def test_convert_dcan_to_bids_single_subject_full_run(tmp_path):
    """convert_dcan_to_bids_single_subject runs with minimal DCAN tree and small NIFTIs."""
    in_dir = _make_dcan_skeleton(tmp_path, sub_id='01', ses_id='01')
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    abcdbids.convert_dcan_to_bids_single_subject(
        in_dir=str(in_dir),
        out_dir=str(out_dir),
        sub_ent='sub-01',
    )

    sub_dir = out_dir / 'sub-01'
    ses_dir = sub_dir / 'ses-01'
    anat_dir = ses_dir / 'anat'
    func_dir = ses_dir / 'func'

    # dataset_description.json is written by convert_dcan2bids after all subjects
    assert (sub_dir / 'sub-01_ses-01_scans.tsv').exists()
    scans_df = pd.read_csv(sub_dir / 'sub-01_ses-01_scans.tsv', sep='\t')
    assert 'filename' in scans_df.columns
    assert 'source_file' in scans_df.columns
    assert len(scans_df) >= 1

    assert (anat_dir / 'sub-01_ses-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.txt').exists()
    assert (anat_dir / 'sub-01_ses-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.txt').exists()

    prefix = 'sub-01_ses-01_task-rest_run-1'
    assert (func_dir / f'{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz').exists()
    assert (func_dir / f'{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json').exists()
    assert (func_dir / f'{prefix}_desc-confounds_timeseries.tsv').exists()
    assert (func_dir / f'{prefix}_desc-confounds_timeseries.json').exists()

    figures_dir = sub_dir / 'figures'
    assert (figures_dir / f'{prefix}_desc-bbregister_bold.svg').exists()
