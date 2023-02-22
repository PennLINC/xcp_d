"""Tests for the xcp_d.interfaces.nilearn module."""
import os

import nibabel as nb
import numpy as np

from xcp_d.interfaces import nilearn


def test_nilearn_merge(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.Merge."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_merge")

    in_file = fmriprep_with_freesurfer_data["boldref"]
    interface = nilearn.Merge(
        in_files=[in_file, in_file],
        out_file="merged.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 4
    assert out_img.shape[3] == 2


def test_nilearn_smooth(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.Smooth."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_smooth")

    in_file = fmriprep_with_freesurfer_data["boldref"]
    interface = nilearn.Smooth(
        in_file=in_file,
        fwhm=6,
        out_file="smoothed_1len.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3

    interface = nilearn.Smooth(
        in_file=in_file,
        fwhm=[2, 3, 4],
        out_file="smoothed_3len.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3


def test_nilearn_binarymath(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.BinaryMath."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_binarymath")

    in_file = fmriprep_with_freesurfer_data["brain_mask_file"]
    interface = nilearn.BinaryMath(
        in_file=in_file,
        expression="img * 5",
        out_file="mathed.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    out_data = out_img.get_fdata()
    assert np.max(out_data) == 5
    assert np.min(out_data) == 0


def test_nilearn_resampletoimage(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.interfaces.nilearn.ResampleToImage."""
    tmpdir = tmp_path_factory.mktemp("test_nilearn_meanimage")

    source_file = fmriprep_with_freesurfer_data["boldref_t1w"]
    target_file = fmriprep_with_freesurfer_data["t1w"]
    target_img = nb.load(target_file)
    source_img = nb.load(source_file)
    assert not np.array_equal(target_img.header.get_zooms(), source_img.header.get_zooms())
    interface = nilearn.ResampleToImage(
        in_file=source_file,
        target_file=target_file,
        out_file="resampled.nii.gz",
    )
    results = interface.run(cwd=tmpdir)
    assert os.path.isfile(results.outputs.out_file)
    out_img = nb.load(results.outputs.out_file)
    assert out_img.ndim == 3
    assert np.array_equal(target_img.affine, out_img.affine)
    assert np.array_equal(target_img.header.get_zooms(), out_img.header.get_zooms())
