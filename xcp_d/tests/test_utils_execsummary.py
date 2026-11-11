"""Test functions in xcp_d.utils.execsummary."""

import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from xcp_d.data import load as load_data
from xcp_d.tests.utils import chdir
from xcp_d.utils import execsummary


def test_plot_gii():
    """Test plot_gii with each view, using an ellipsoid with semi-axes of 10, 20, and 30 mm."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1)
    mesh.vertices = mesh.vertices * [10, 20, 30]

    # plot_gii draws on the first axes of a nilearn slicer
    expected_extents = {'x': (20, 30), 'y': (10, 30), 'z': (10, 20)}
    for view, (h_extent, v_extent) in expected_extents.items():
        fig, ax = plt.subplots()
        slicer = SimpleNamespace(axes={0: SimpleNamespace(ax=ax)})

        lines = execsummary.plot_gii(mesh, 0, 'red', slicer, view)
        assert len(lines) > 0
        assert ax.lines[-len(lines) :] == lines
        h_data = np.concatenate([line.get_xdata() for line in lines])
        v_data = np.concatenate([line.get_ydata() for line in lines])
        assert np.isclose(np.abs(h_data).max(), h_extent, rtol=0.05)
        assert np.isclose(np.abs(v_data).max(), v_extent, rtol=0.05)

        # A plane that misses the mesh draws nothing
        assert execsummary.plot_gii(mesh, 50, 'red', slicer, view) == []
        assert len(ax.lines) == len(lines)
        plt.close(fig)


def test_make_mosaic(tmp_path_factory):
    """Test make_mosaic."""
    tmpdir = tmp_path_factory.mktemp('test_make_mosaic')

    # Make a simple PNG file
    png_file = os.path.join(tmpdir, 'temp.png')
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_facecolor('yellow')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(png_file)
    plt.close(fig)

    png_files = [png_file] * 10
    with chdir(tmpdir):
        mosaic_file = execsummary.make_mosaic(png_files)

    assert os.path.isfile(mosaic_file)


def test_modify_pngs_scene_template(tmp_path_factory):
    """Test modify_pngs_scene_template."""
    tmpdir = tmp_path_factory.mktemp('test_modify_pngs_scene_template')

    pngs_scene_template = str(load_data('executive_summary_scenes/pngs_template.scene.gz'))

    with chdir(tmpdir):
        scene_file = execsummary.modify_pngs_scene_template(
            anat_file='anat.nii.gz',
            rh_pial_surf='rh_pial.surf.gii',
            lh_pial_surf='lh_pial.surf.gii',
            rh_wm_surf='rh_wm.surf.gii',
            lh_wm_surf='lh_wm.surf.gii',
            scene_template=pngs_scene_template,
        )

    assert os.path.isfile(scene_file)
    with open(scene_file) as fo:
        scene_text = fo.read()
    assert 'T2_IMG_PATH' not in scene_text
    assert 'T2_IMG_NAME' not in scene_text


def test_get_png_image_names():
    """Test get_png_image_names."""
    scene_index, image_descriptions = execsummary.get_png_image_names()
    assert len(scene_index) == len(image_descriptions) == 9
