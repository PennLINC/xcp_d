"""Tests for xcp_d.ingression.utils."""

import json
import os
from unittest.mock import patch

import nibabel as nb
import numpy as np
import pandas as pd
import pytest

from xcp_d.ingression import utils as ingress_utils


def _minimal_nifti_3d(shape=(4, 4, 4), path=None):
    """Write a minimal 3D NIfTI (e.g. mask) with zeros."""
    arr = np.zeros(shape, dtype=np.float32)
    img = nb.Nifti1Image(arr, np.eye(4))
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        img.to_filename(path)
    return img if path is None else path


def _minimal_nifti_4d(shape=(4, 4, 4, 5), path=None):
    """Write a minimal 4D NIfTI (e.g. BOLD) with zeros."""
    arr = np.zeros(shape, dtype=np.float32)
    img = nb.Nifti1Image(arr, np.eye(4))
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        img.to_filename(path)
    return img if path is None else path


def test_write_json(tmp_path):
    """write_json writes a dict to a JSON file."""
    outfile = tmp_path / 'out.json'
    data = {'a': 1, 'b': [2, 3]}
    result = ingress_utils.write_json(data, str(outfile))
    assert result == str(outfile)
    assert outfile.exists()
    with open(outfile) as f:
        assert json.load(f) == data


def test_copy_file(tmp_path):
    """copy_file copies source to destination."""
    src = tmp_path / 'src.txt'
    dst = tmp_path / 'dst.txt'
    src.write_text('hello')
    ingress_utils.copy_file(str(src), str(dst))
    assert dst.read_text() == 'hello'
    # Idempotent when content matches
    ingress_utils.copy_file(str(src), str(dst))
    assert dst.read_text() == 'hello'


def test_copy_file_overwrites_when_different(tmp_path):
    """copy_file overwrites destination when content differs."""
    src = tmp_path / 'src.txt'
    dst = tmp_path / 'dst.txt'
    src.write_text('hello')
    dst.write_text('old')
    ingress_utils.copy_file(str(src), str(dst))
    assert dst.read_text() == 'hello'


def test_copy_files_in_dict(tmp_path):
    """copy_files_in_dict copies each source to its list of destinations."""
    a = tmp_path / 'a.txt'
    b = tmp_path / 'b.txt'
    c = tmp_path / 'c.txt'
    a.write_text('a')
    copy_dictionary = {
        str(a): [str(b)],
        str(a): [str(c)],  # same source, different dest (last wins in dict)
    }
    copy_dictionary = {str(a): [str(b), str(c)]}
    ingress_utils.copy_files_in_dict(copy_dictionary)
    assert b.read_text() == 'a'
    assert c.read_text() == 'a'


def test_copy_files_in_dict_raises_when_value_not_list(tmp_path):
    """copy_files_in_dict raises ValueError when a value is not a list."""
    f = tmp_path / 'f.txt'
    f.write_text('x')
    with pytest.raises(ValueError, match='should be a list'):
        ingress_utils.copy_files_in_dict({str(f): str(tmp_path / 'out.txt')})


def test_collect_anatomical_files_none_present(tmp_path):
    """collect_anatomical_files returns empty dict when no expected files exist."""
    anat_orig = tmp_path / 'anat_orig'
    anat_orig.mkdir()
    anat_bids = tmp_path / 'anat_bids'
    anat_bids.mkdir()
    base = 'sub-01_space-MNI152NLin6Asym_res-2'
    out = ingress_utils.collect_anatomical_files(str(anat_orig), str(anat_bids), base)
    assert out == {}


def test_collect_anatomical_files_some_present(tmp_path):
    """collect_anatomical_files maps existing files to BIDS paths."""
    anat_orig = tmp_path / 'anat_orig'
    anat_bids = tmp_path / 'anat_bids'
    anat_orig.mkdir()
    anat_bids.mkdir()
    t1w = anat_orig / 'T1w.nii.gz'
    ribbon = anat_orig / 'ribbon.nii.gz'
    t1w.write_bytes(b'')
    ribbon.write_bytes(b'')
    base = 'sub-01_space-MNI152NLin6Asym_res-2'
    out = ingress_utils.collect_anatomical_files(str(anat_orig), str(anat_bids), base)
    assert len(out) == 2
    assert str(t1w) in out
    assert out[str(t1w)] == [os.path.join(anat_bids, f'{base}_desc-preproc_T1w.nii.gz')]
    assert str(ribbon) in out
    assert out[str(ribbon)] == [os.path.join(anat_bids, f'{base}_desc-ribbon_T1w.nii.gz')]


def test_collect_anatomical_files_brainmask_variants(tmp_path):
    """collect_anatomical_files accepts brainmask_fs or brainmask_fs.2.0."""
    anat_orig = tmp_path / 'anat_orig'
    anat_bids = tmp_path / 'anat_bids'
    anat_orig.mkdir()
    anat_bids.mkdir()
    (anat_orig / 'brainmask_fs.2.0.nii.gz').write_bytes(b'')
    base = 'sub-01_space-MNI152NLin6Asym_res-2'
    out = ingress_utils.collect_anatomical_files(str(anat_orig), str(anat_bids), base)
    assert len(out) == 1
    assert list(out.values())[0][0].endswith('_desc-brain_mask.nii.gz')


def test_collect_meshes_none_present(tmp_path):
    """collect_meshes returns empty dict when no surface files exist."""
    anat_orig = tmp_path / 'anat_orig'
    anat_bids = tmp_path / 'anat_bids'
    anat_orig.mkdir()
    (anat_orig / 'fsaverage_LR32k').mkdir()
    anat_bids.mkdir()
    out = ingress_utils.collect_meshes(str(anat_orig), str(anat_bids), 'subid', 'sub-01')
    assert out == {}


def test_collect_meshes_some_present(tmp_path):
    """collect_meshes maps L/R pial and white surfaces to BIDS paths."""
    anat_orig = tmp_path / 'anat_orig'
    anat_bids = tmp_path / 'anat_bids'
    fsaverage = anat_orig / 'fsaverage_LR32k'
    fsaverage.mkdir(parents=True)
    anat_bids.mkdir()
    (fsaverage / 'subid.L.pial.32k_fs_LR.surf.gii').write_bytes(b'')
    (fsaverage / 'subid.R.pial.32k_fs_LR.surf.gii').write_bytes(b'')
    (fsaverage / 'subid.L.white.32k_fs_LR.surf.gii').write_bytes(b'')
    (fsaverage / 'subid.R.white.32k_fs_LR.surf.gii').write_bytes(b'')
    out = ingress_utils.collect_meshes(str(anat_orig), str(anat_bids), 'subid', 'sub-01')
    assert len(out) == 4
    out_paths = [p for v in out.values() for p in v]
    assert any('hemi-L_pial' in p for p in out_paths)
    assert any('hemi-R_pial' in p for p in out_paths)
    assert any('hemi-L_smoothwm' in p for p in out_paths)
    assert any('hemi-R_smoothwm' in p for p in out_paths)


def test_collect_morphs_skips_when_files_missing(tmp_path):
    """collect_morphs skips morphometry when L or R file is missing."""
    anat_orig = tmp_path / 'anat_orig'
    anat_bids = tmp_path / 'anat_bids'
    fsaverage = anat_orig / 'fsaverage_LR32k'
    fsaverage.mkdir(parents=True)
    anat_bids.mkdir()
    (fsaverage / 'subid.L.thickness.32k_fs_LR.shape.gii').write_bytes(b'')
    # No R file
    with patch.object(ingress_utils, 'CiftiCreateDenseScalar') as mock_wb:
        out = ingress_utils.collect_morphs(str(anat_orig), str(anat_bids), 'subid', 'sub-01')
    assert out == {}
    mock_wb.assert_not_called()


def test_collect_morphs_calls_interface_when_both_hemis_present(tmp_path):
    """collect_morphs runs CiftiCreateDenseScalar when L and R files exist."""
    anat_orig = tmp_path / 'anat_orig'
    anat_bids = tmp_path / 'anat_bids'
    fsaverage = anat_orig / 'fsaverage_LR32k'
    fsaverage.mkdir(parents=True)
    anat_bids.mkdir()
    lh = fsaverage / 'subid.L.thickness.32k_fs_LR.shape.gii'
    rh = fsaverage / 'subid.R.thickness.32k_fs_LR.shape.gii'
    lh.write_bytes(b'')
    rh.write_bytes(b'')
    out_file = anat_bids / 'sub-01_space-fsLR_den-91k_thickness.dscalar.nii'

    with patch.object(ingress_utils, 'CiftiCreateDenseScalar') as MockWB:
        mock_run = MockWB.return_value.run
        mock_run.return_value.outputs = type('R', (), {'out_file': None})()
        out = ingress_utils.collect_morphs(str(anat_orig), str(anat_bids), 'subid', 'sub-01')
    assert MockWB.called
    assert str(lh) in out
    assert str(rh) in out
    assert out[str(lh)] == str(out_file)
    assert out[str(rh)] == str(out_file)


def test_extract_mean_signal(tmp_path):
    """extract_mean_signal returns mean time series in mask (minimal NIFTIs)."""
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    mask_path = tmp_path / 'mask.nii.gz'
    bold_path = tmp_path / 'bold.nii.gz'
    _minimal_nifti_3d(path=str(mask_path))
    _minimal_nifti_4d(shape=(4, 4, 4, 6), path=str(bold_path))
    # Ensure mask has at least one voxel so mean is defined
    mask_img = nb.load(str(mask_path))
    mask_data = np.asarray(mask_img.dataobj)
    mask_data[0, 0, 0] = 1
    nb.Nifti1Image(mask_data.astype(np.float32), mask_img.affine).to_filename(str(mask_path))
    result = ingress_utils.extract_mean_signal(str(mask_path), str(bold_path), str(work_dir))
    assert result.ndim == 1
    assert result.shape[0] == 6


def test_extract_mean_signal_raises_when_mask_missing(tmp_path):
    """extract_mean_signal asserts mask file exists."""
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    bold_path = tmp_path / 'bold.nii.gz'
    _minimal_nifti_4d(path=str(bold_path))
    with pytest.raises(AssertionError, match='File DNE'):
        ingress_utils.extract_mean_signal(
            str(tmp_path / 'nonexistent.nii.gz'), str(bold_path), str(work_dir)
        )


def test_collect_hcp_confounds(tmp_path):
    """collect_hcp_confounds writes TSV and JSON from Movement_Regressors and RMS."""
    task_dir = tmp_path / 'task'
    task_dir.mkdir()
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    # 4 time points, 6 motion columns (HCP order: trans x,y,z, rot x,y,z)
    mvreg = task_dir / 'Movement_Regressors.txt'
    mvreg.write_text('0.1 0.2 0.3 1 2 3\n0 0 0 0 0 0\n0 0 0 0 0 0\n0 0 0 0 0 0\n')
    rmsd_file = task_dir / 'Movement_AbsoluteRMS.txt'
    rmsd_file.write_text('0.5\n0.4\n0.3\n0.2\n')
    bold_path = tmp_path / 'bold.nii.gz'
    mask_path = tmp_path / 'brainmask.nii.gz'
    csf_path = tmp_path / 'csf.nii.gz'
    wm_path = tmp_path / 'wm.nii.gz'
    _minimal_nifti_4d(shape=(4, 4, 4, 4), path=str(bold_path))
    _minimal_nifti_3d(path=str(mask_path))
    _minimal_nifti_3d(path=str(csf_path))
    _minimal_nifti_3d(path=str(wm_path))
    prefix = 'sub-01_task-rest_run-1'

    with patch.object(ingress_utils, 'extract_mean_signal') as mock_extract:
        mock_extract.return_value = np.array([1.0, 2.0, 3.0, 4.0])
        ingress_utils.collect_hcp_confounds(
            task_dir_orig=str(task_dir),
            out_dir=str(out_dir),
            prefix=prefix,
            work_dir=str(work_dir),
            bold_file=str(bold_path),
            brainmask_file=str(mask_path),
            csf_mask_file=str(csf_path),
            wm_mask_file=str(wm_path),
        )

    tsv = out_dir / f'{prefix}_desc-confounds_timeseries.tsv'
    js = out_dir / f'{prefix}_desc-confounds_timeseries.json'
    assert tsv.exists()
    assert js.exists()
    df = pd.read_csv(tsv, sep='\t')
    assert 'trans_x' in df.columns
    assert 'rot_z' in df.columns
    assert 'global_signal' in df.columns
    assert 'framewise_displacement' in df.columns
    assert len(df) == 4
    with open(js) as f:
        meta = json.load(f)
    assert 'trans_x' in meta


def test_collect_hcp_confounds_raises_when_movement_missing(tmp_path):
    """collect_hcp_confounds asserts Movement_Regressors.txt exists."""
    task_dir = tmp_path / 'task'
    task_dir.mkdir()
    with pytest.raises(AssertionError):
        ingress_utils.collect_hcp_confounds(
            task_dir_orig=str(task_dir),
            out_dir=str(tmp_path),
            prefix='sub-01_task-rest',
            work_dir=str(tmp_path),
            bold_file='x.nii.gz',
            brainmask_file='y.nii.gz',
            csf_mask_file='z.nii.gz',
            wm_mask_file='w.nii.gz',
        )


def test_collect_ukbiobank_confounds(tmp_path):
    """collect_ukbiobank_confounds writes TSV and JSON from FSL par and rms."""
    task_dir = tmp_path / 'task'
    mc_dir = task_dir / 'mc'
    mc_dir.mkdir(parents=True)
    # FSL .par: 6 columns (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z), one row per volume
    par_file = mc_dir / 'prefiltered_func_data_mcf.par'
    par_file.write_text('0.01 0.02 0.03 0.1 0.2 0.3\n0 0 0 0 0 0\n')
    rms_file = mc_dir / 'prefiltered_func_data_mcf_abs.rms'
    rms_file.write_text('0.5\n0.4\n')
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    work_dir = tmp_path / 'work'
    work_dir.mkdir()
    bold_path = tmp_path / 'bold.nii.gz'
    mask_path = tmp_path / 'mask.nii.gz'
    _minimal_nifti_4d(shape=(4, 4, 4, 2), path=str(bold_path))
    _minimal_nifti_3d(path=str(mask_path))
    prefix = 'sub-01_ses-01_task-rest'

    with patch.object(ingress_utils, 'extract_mean_signal') as mock_extract:
        mock_extract.return_value = np.array([1.0, 2.0])
        ingress_utils.collect_ukbiobank_confounds(
            task_dir_orig=str(task_dir),
            out_dir=str(out_dir),
            prefix=prefix,
            work_dir=str(work_dir),
            bold_file=str(bold_path),
            brainmask_file=str(mask_path),
        )

    tsv = out_dir / f'{prefix}_desc-confounds_timeseries.tsv'
    js = out_dir / f'{prefix}_desc-confounds_timeseries.json'
    assert tsv.exists()
    assert js.exists()
    df = pd.read_csv(tsv, sep='\t')
    assert 'trans_x' in df.columns
    assert 'global_signal' in df.columns
    assert 'rmsd' in df.columns
    assert len(df) == 2


def test_collect_ukbiobank_confounds_raises_when_par_missing(tmp_path):
    """collect_ukbiobank_confounds asserts par file exists."""
    task_dir = tmp_path / 'task'
    task_dir.mkdir()
    (task_dir / 'mc').mkdir()
    with pytest.raises(AssertionError):
        ingress_utils.collect_ukbiobank_confounds(
            task_dir_orig=str(task_dir),
            out_dir=str(tmp_path),
            prefix='sub-01_task-rest',
            work_dir=str(tmp_path),
            bold_file='x.nii.gz',
            brainmask_file='y.nii.gz',
        )


def test_plot_bbreg(tmp_path):
    """plot_bbreg produces an SVG from minimal fixed/moving/contour NIFTIs."""
    # Use slightly larger grid so niworkflows cuts_from_bbox has enough room
    shape = (10, 10, 10)
    fixed = tmp_path / 'fixed.nii.gz'
    moving = tmp_path / 'moving.nii.gz'
    contour = tmp_path / 'contour.nii.gz'
    out_file = tmp_path / 'report.svg'
    _minimal_nifti_3d(shape=shape, path=str(fixed))
    _minimal_nifti_3d(shape=shape, path=str(moving))
    _minimal_nifti_3d(shape=shape, path=str(contour))
    result = ingress_utils.plot_bbreg(
        fixed_image=str(fixed),
        moving_image=str(moving),
        contour=str(contour),
        out_file=str(out_file),
    )
    assert result == str(out_file)
    assert out_file.exists()
    content = out_file.read_text()
    assert 'svg' in content.lower() or content.lstrip().startswith('<?xml')
