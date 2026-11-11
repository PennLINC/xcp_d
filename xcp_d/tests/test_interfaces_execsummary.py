"""Tests for the xcp_d.interfaces.execsummary module."""

import os

import nibabel as nb
import numpy as np
import trimesh
from PIL import Image

from xcp_d.interfaces import execsummary


def test_plotslicesforbrainsprite(datasets, tmp_path_factory):
    """Test the PlotSlicesForBrainSprite interface."""
    tmpdir = tmp_path_factory.mktemp('test_generate_confounds')

    anat_dir = os.path.join(datasets['pnc'], 'sub-1648798153', 'ses-PNC1', 'anat')

    anat = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_desc-preproc_T1w.nii.gz')
    lh_pial = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_pial.surf.gii')
    lh_wm = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_white.surf.gii')
    rh_pial = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_pial.surf.gii')
    rh_wm = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_white.surf.gii')

    img = nb.load(anat)
    n_slices = img.shape[0]

    interface = execsummary.PlotSlicesForBrainSprite(
        lh_wm=lh_wm,
        lh_pial=lh_pial,
        rh_wm=rh_wm,
        rh_pial=rh_pial,
        nifti=anat,
    )
    results = interface.run(cwd=tmpdir)
    assert len(results.outputs.out_files) == n_slices


def _write_sphere(filename, center, radius):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    vertices = (mesh.vertices + center).astype(np.float32)
    gii = nb.gifti.GiftiImage(
        darrays=[
            nb.gifti.GiftiDataArray(vertices, intent='NIFTI_INTENT_POINTSET'),
            nb.gifti.GiftiDataArray(
                mesh.faces.astype(np.int32),
                intent='NIFTI_INTENT_TRIANGLE',
            ),
        ]
    )
    nb.save(gii, filename)
    return filename


def _count_red_pixels(png_file):
    rgb = np.asarray(Image.open(png_file).convert('RGB')).astype(int)
    return np.sum((rgb[..., 0] > 80) & (rgb[..., 1] < 40) & (rgb[..., 2] < 40))


def _count_bright_gray_pixels(png_file):
    rgb = np.asarray(Image.open(png_file).convert('RGB')).astype(int)
    return np.sum((rgb.min(axis=-1) > 200) & (np.ptp(rgb, axis=-1) < 10))


def test_plotslicesforbrainsprite_synthetic(tmp_path_factory):
    """Test PlotSlicesForBrainSprite on a synthetic volume and spherical surfaces."""
    tmpdir = tmp_path_factory.mktemp('test_plotslicesforbrainsprite_synthetic')

    # A 2-mm LAS volume, so the image has to be reordered before plotting
    shape = (30, 36, 32)
    affine = np.diag([-2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [29, -35, -31]
    ijk = np.stack(np.indices(shape), axis=-1).reshape(-1, 3)
    xyz = nb.affines.apply_affine(affine, ijk).reshape((*shape, 3))
    data = np.where(np.linalg.norm(xyz, axis=-1) < 25, 100.0, 10.0)
    data += np.random.default_rng(0).normal(scale=1, size=shape)
    anat = os.path.join(tmpdir, 'anat.nii.gz')
    nb.Nifti1Image(data.astype(np.float32), affine).to_filename(anat)

    surfs = {
        'lh_pial': _write_sphere(os.path.join(tmpdir, 'lh_pial.surf.gii'), [-10, 0, 0], 12),
        'rh_pial': _write_sphere(os.path.join(tmpdir, 'rh_pial.surf.gii'), [10, 0, 0], 12),
        'lh_wm': _write_sphere(os.path.join(tmpdir, 'lh_wm.surf.gii'), [-10, 0, 0], 8),
        'rh_wm': _write_sphere(os.path.join(tmpdir, 'rh_wm.surf.gii'), [10, 0, 0], 8),
    }

    interface = execsummary.PlotSlicesForBrainSprite(nifti=anat, **surfs)
    results = interface.run(cwd=tmpdir)
    out_files = results.outputs.out_files
    assert [os.path.basename(f) for f in out_files] == [
        f'test_{i:03d}.png' for i in range(shape[0])
    ]

    # The PNGs must be at least as wide as the tiles in the mosaic
    width, _ = Image.open(out_files[0]).size
    assert width >= 218

    # Voxel i is at x = 29 - 2 * i, so the pial spheres (x from -22 to 22) only cross slices 4-25.
    # Slices drawn after ones with contours must not keep those contours.
    red_pixels = [_count_red_pixels(f) for f in out_files]
    assert all(red_pixels[i] > 0 for i in (5, 14, 24))
    assert all(red_pixels[i] == 0 for i in (0, 1, 28, 29))

    # The bright ball (x from -25 to 25) is drawn in the middle slices but not the outermost ones
    bright_pixels = [_count_bright_gray_pixels(f) for f in out_files]
    assert all(bright_pixels[i] > 0 for i in (5, 14, 24))
    assert all(bright_pixels[i] == 0 for i in (0, 29))


def test_crop_mesh_to_plane():
    """Cropping a mesh to the faces that span a plane does not change its cross-section."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=20)
    face_x = mesh.vertices[mesh.faces, 0]
    face_range = (face_x.min(axis=1), face_x.max(axis=1))

    assert execsummary._crop_mesh_to_plane(mesh, face_range, 25) is None

    for coord in (-19.5, -3.3, 0, 12.1):
        sub_mesh = execsummary._crop_mesh_to_plane(mesh, face_range, coord)
        assert sub_mesh.faces.shape[0] < mesh.faces.shape[0]

        full_section = mesh.section(plane_origin=[coord, 0, 0], plane_normal=[1, 0, 0])
        sub_section = sub_mesh.section(plane_origin=[coord, 0, 0], plane_normal=[1, 0, 0])
        assert len(full_section.discrete) == len(sub_section.discrete)
        for full_line, sub_line in zip(
            full_section.discrete,
            sub_section.discrete,
            strict=True,
        ):
            np.testing.assert_allclose(full_line, sub_line)
