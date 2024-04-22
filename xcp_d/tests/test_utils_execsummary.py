"""Test functions in xcp_d.utils.execsummary."""

import os

import matplotlib.pyplot as plt

from xcp_d.data import load as load_data
from xcp_d.tests.utils import chdir
from xcp_d.utils import execsummary


def test_make_mosaic(tmp_path_factory):
    """Test make_mosaic."""
    tmpdir = tmp_path_factory.mktemp("test_make_mosaic")

    # Make a simple PNG file
    png_file = os.path.join(tmpdir, "temp.png")
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_facecolor("yellow")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(png_file)
    plt.close(fig)

    png_files = [png_file] * 10
    with chdir(tmpdir):
        mosaic_file = execsummary.make_mosaic(png_files)

    assert os.path.isfile(mosaic_file)


def test_modify_brainsprite_scene_template(tmp_path_factory):
    """Test modify_brainsprite_scene_template."""
    tmpdir = tmp_path_factory.mktemp("test_modify_brainsprite_scene_template")

    brainsprite_scene_template = str(
        load_data(
            "executive_summary_scenes/brainsprite_template.scene.gz",
        )
    )

    with chdir(tmpdir):
        scene_file = execsummary.modify_brainsprite_scene_template(
            slice_number=10,
            anat_file="anat.nii.gz",
            rh_pial_surf="rh_pial.surf.gii",
            lh_pial_surf="lh_pial.surf.gii",
            rh_wm_surf="rh_wm.surf.gii",
            lh_wm_surf="lh_wm.surf.gii",
            scene_template=brainsprite_scene_template,
        )

    assert os.path.isfile(scene_file)


def test_modify_pngs_scene_template(tmp_path_factory):
    """Test modify_pngs_scene_template."""
    tmpdir = tmp_path_factory.mktemp("test_modify_pngs_scene_template")

    pngs_scene_template = str(load_data("executive_summary_scenes/pngs_template.scene.gz"))

    with chdir(tmpdir):
        scene_file = execsummary.modify_pngs_scene_template(
            anat_file="anat.nii.gz",
            rh_pial_surf="rh_pial.surf.gii",
            lh_pial_surf="lh_pial.surf.gii",
            rh_wm_surf="rh_wm.surf.gii",
            lh_wm_surf="lh_wm.surf.gii",
            scene_template=pngs_scene_template,
        )

    assert os.path.isfile(scene_file)


def test_get_n_frames(ds001419_data):
    """Test get_n_frames."""
    anat_file = ds001419_data["brain_mask_file"]
    frame_numbers = execsummary.get_n_frames(anat_file)
    assert len(frame_numbers) == 194


def test_get_png_image_names():
    """Test get_png_image_names."""
    scene_index, image_descriptions = execsummary.get_png_image_names()
    assert len(scene_index) == len(image_descriptions) == 9
