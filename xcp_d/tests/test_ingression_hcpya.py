"""Tests for xcp_d.ingression.hcpya."""

import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from xcp_d.ingression import hcpya

# Small imaging grid: 5x5x5 = 125 voxels; 5x5x5x4 = 500 voxels (under limit)
_SHAPE_3D = (5, 5, 5)
_SHAPE_4D = (5, 5, 5, 4)
_AFFINE = np.eye(4)


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


def _make_hcp_skeleton(tmp_path, sub_id='01'):
    """Build minimal HCP-like directory tree with small NIFTIs."""
    in_dir = tmp_path / 'hcp_in'
    in_dir.mkdir()
    mni = in_dir / sub_id / 'MNINonLinear'
    mni.mkdir(parents=True)
    fsaverage = mni / 'fsaverage_LR32k'
    fsaverage.mkdir()
    results = mni / 'Results'
    results.mkdir()
    task_dir = results / 'tfMRI_REST1_RL'
    task_dir.mkdir()

    _write_minimal_nifti(mni / 'T1w.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'ribbon.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)

    for hemi, surf in [('L', 'pial'), ('R', 'pial'), ('L', 'white'), ('R', 'white')]:
        (fsaverage / f'{sub_id}.{hemi}.{surf}.32k_fs_LR.surf.gii').write_bytes(b'')

    _write_minimal_nifti(task_dir / 'SBRef_dc.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(
        task_dir / 'tfMRI_REST1_RL.nii.gz',
        _SHAPE_4D,
        zooms=(2.0, 2.0, 2.0, 2.0),
    )
    _write_minimal_nifti(task_dir / 'brainmask_fs.2.nii.gz', _SHAPE_3D, mask=True)
    _write_minimal_nifti(task_dir / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)

    n_vols = _SHAPE_4D[-1]
    mvreg = '\n'.join('0 0 0 0 0 0' for _ in range(n_vols)) + '\n'
    (task_dir / 'Movement_Regressors.txt').write_text(mvreg)
    (task_dir / 'Movement_AbsoluteRMS.txt').write_text(
        '\n'.join('0.5' for _ in range(n_vols)) + '\n'
    )

    (task_dir / 'tfMRI_REST1_RL_Atlas_MSMAll.dtseries.nii').write_bytes(b'')

    return in_dir


def _add_hcp_subject(in_dir, sub_id):
    """Add a second HCP subject to an existing in_dir (same layout as _make_hcp_skeleton)."""
    in_dir = Path(in_dir)
    mni = in_dir / sub_id / 'MNINonLinear'
    mni.mkdir(parents=True)
    fsaverage = mni / 'fsaverage_LR32k'
    fsaverage.mkdir()
    results = mni / 'Results'
    results.mkdir()
    task_dir = results / 'tfMRI_REST1_RL'
    task_dir.mkdir()

    _write_minimal_nifti(mni / 'T1w.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'ribbon.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(mni / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)

    for hemi, surf in [('L', 'pial'), ('R', 'pial'), ('L', 'white'), ('R', 'white')]:
        (fsaverage / f'{sub_id}.{hemi}.{surf}.32k_fs_LR.surf.gii').write_bytes(b'')

    _write_minimal_nifti(task_dir / 'SBRef_dc.nii.gz', _SHAPE_3D)
    _write_minimal_nifti(
        task_dir / 'tfMRI_REST1_RL.nii.gz',
        _SHAPE_4D,
        zooms=(2.0, 2.0, 2.0, 2.0),
    )
    _write_minimal_nifti(task_dir / 'brainmask_fs.2.nii.gz', _SHAPE_3D, mask=True)
    _write_minimal_nifti(task_dir / 'brainmask_fs.2.0.nii.gz', _SHAPE_3D, mask=True)

    n_vols = _SHAPE_4D[-1]
    mvreg = '\n'.join('0 0 0 0 0 0' for _ in range(n_vols)) + '\n'
    (task_dir / 'Movement_Regressors.txt').write_text(mvreg)
    (task_dir / 'Movement_AbsoluteRMS.txt').write_text(
        '\n'.join('0.5' for _ in range(n_vols)) + '\n'
    )
    (task_dir / 'tfMRI_REST1_RL_Atlas_MSMAll.dtseries.nii').write_bytes(b'')


def test_convert_hcp2bids_two_subjects_both_converted_reproduces_issue_1471(tmp_path):
    """convert_hcp2bids converts all subjects; second subject not skipped (Issue #1471).

    When dataset_description.json is written at the end of the first subject's
    conversion, the second subject's conversion must not be skipped. Otherwise
    only one subject is converted (see https://github.com/PennLINC/xcp_d/issues/1471).
    """
    from xcp_d.data import load as real_load

    in_dir = _make_hcp_skeleton(tmp_path, sub_id='01')
    _add_hcp_subject(in_dir, '02')

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    masks_dir = tmp_path / 'masks'
    masks_dir.mkdir()
    csf_path = masks_dir / 'space-MNI152NLin6Asym_res-2_label-CSF_mask.nii.gz'
    wm_path = masks_dir / 'space-MNI152NLin6Asym_res-2_label-WM_mask.nii.gz'
    _write_minimal_nifti(csf_path, _SHAPE_3D, mask=True)
    _write_minimal_nifti(wm_path, _SHAPE_3D, mask=True)

    def fake_load_data(name):
        if 'label-CSF_mask' in name:
            return str(csf_path)
        if 'label-WM_mask' in name:
            return str(wm_path)
        return str(real_load.readable(name))

    with patch.object(hcpya, 'load_data', side_effect=fake_load_data):
        result = hcpya.convert_hcp2bids(
            str(in_dir), str(out_dir), participant_ids=['sub-01', 'sub-02']
        )

    assert result == ['sub-01', 'sub-02']

    # Both subjects must have been converted (bug: only first was when dataset_description
    # was written inside single-subject and checked at start of next).
    sub01_dir = out_dir / 'sub-01'
    sub02_dir = out_dir / 'sub-02'
    assert sub01_dir.is_dir(), 'sub-01 output dir should exist'
    assert sub02_dir.is_dir(), 'sub-02 output dir should exist'

    for sub_ent, sub_dir in [('sub-01', sub01_dir), ('sub-02', sub02_dir)]:
        anat_dir = sub_dir / 'anat'
        func_dir = sub_dir / 'func'
        assert (anat_dir / f'{sub_ent}_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.txt').exists()
        assert (anat_dir / f'{sub_ent}_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.txt').exists()
        prefix = f'{sub_ent}_task-rest_dir-RL_run-1'
        bold_path = func_dir / f'{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz'
        assert bold_path.exists()
        assert (func_dir / f'{prefix}_desc-confounds_timeseries.tsv').exists()


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
    mock_single.assert_any_call(in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='sub-01')
    mock_single.assert_any_call(in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='sub-02')


def test_convert_hcp2bids_passes_participant_ids_through(tmp_path):
    """convert_hcp2bids returns the participant_ids list as given and passes to single-subject."""
    in_dir = tmp_path / 'hcp_in'
    in_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    with patch.object(hcpya, 'convert_hcp_to_bids_single_subject') as mock_single:
        result = hcpya.convert_hcp2bids(str(in_dir), str(out_dir), participant_ids=['01'])
    assert result == ['01']
    mock_single.assert_called_once_with(in_dir=str(in_dir), out_dir=str(out_dir), sub_ent='01')


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


def test_convert_hcp_to_bids_single_subject_full_run(tmp_path):
    """convert_hcp_to_bids_single_subject runs with minimal HCP tree and small NIFTIs."""
    from xcp_d.data import load as real_load

    in_dir = _make_hcp_skeleton(tmp_path, sub_id='01')
    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    # Small CSF/WM masks in same space as BOLD so extract_mean_signal works
    masks_dir = tmp_path / 'masks'
    masks_dir.mkdir()
    csf_path = masks_dir / 'space-MNI152NLin6Asym_res-2_label-CSF_mask.nii.gz'
    wm_path = masks_dir / 'space-MNI152NLin6Asym_res-2_label-WM_mask.nii.gz'
    _write_minimal_nifti(csf_path, _SHAPE_3D, mask=True)
    _write_minimal_nifti(wm_path, _SHAPE_3D, mask=True)

    def fake_load_data(name):
        if 'label-CSF_mask' in name:
            return str(csf_path)
        if 'label-WM_mask' in name:
            return str(wm_path)
        return str(real_load.readable(name))

    with patch.object(hcpya, 'load_data', side_effect=fake_load_data):
        hcpya.convert_hcp_to_bids_single_subject(
            in_dir=str(in_dir),
            out_dir=str(out_dir),
            sub_ent='sub-01',
        )

    sub_dir = out_dir / 'sub-01'
    anat_dir = sub_dir / 'anat'
    func_dir = sub_dir / 'func'

    assert (out_dir / 'dataset_description.json').exists()
    with open(out_dir / 'dataset_description.json') as f:
        desc = json.load(f)
    assert desc.get('Name') == 'HCP'
    assert desc.get('DatasetType') == 'derivative'

    assert (sub_dir / 'sub-01_scans.tsv').exists()
    scans_df = pd.read_csv(sub_dir / 'sub-01_scans.tsv', sep='\t')
    assert 'filename' in scans_df.columns
    assert 'source_file' in scans_df.columns
    assert len(scans_df) >= 1

    assert (anat_dir / 'sub-01_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.txt').exists()
    assert (anat_dir / 'sub-01_from-MNI152NLin6Asym_to-T1w_mode-image_xfm.txt').exists()

    prefix = 'sub-01_task-rest_dir-RL_run-1'
    assert (func_dir / f'{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz').exists()
    assert (func_dir / f'{prefix}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.json').exists()
    assert (func_dir / f'{prefix}_desc-confounds_timeseries.tsv').exists()
    assert (func_dir / f'{prefix}_desc-confounds_timeseries.json').exists()

    figures_dir = sub_dir / 'figures'
    assert (figures_dir / f'{prefix}_desc-bbregister_bold.svg').exists()
