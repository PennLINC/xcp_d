"""Test functions in xcp_d.utils.execsummary."""
import os

from pkg_resources import resource_filename as pkgrf

from xcp_d.tests.utils import chdir
from xcp_d.utils import execsummary


def test_make_mosaic():
    """Test make_mosaic."""
    ...


def test_modify_brainsprite_scene_template(tmp_path_factory):
    """Test modify_brainsprite_scene_template."""
    tmpdir = tmp_path_factory.mktemp("test_modify_brainsprite_scene_template")

    brainsprite_scene_template = pkgrf(
        "xcp_d",
        "data/executive_summary_scenes/brainsprite_template.scene.gz",
    )

    with chdir(tmpdir):
        execsummary.modify_brainsprite_scene_template(
            slice_number=10,
            anat_file="anat.nii.gz",
            rh_pial_surf="rh_pial.surf.gii",
            lh_pial_surf="lh_pial.surf.gii",
            rh_wm_surf="rh_wm.surf.gii",
            lh_wm_surf="lh_wm.surf.gii",
            scene_template=brainsprite_scene_template,
        )

    assert os.path.isfile(os.path.join(tmpdir, "modified_scene.scene"))


def test_modify_pngs_scene_template(tmp_path_factory):
    """Test modify_pngs_scene_template."""
    tmpdir = tmp_path_factory.mktemp("test_modify_pngs_scene_template")

    pngs_scene_template = pkgrf("xcp_d", "data/executive_summary_scenes/pngs_template.scene.gz")

    with chdir(tmpdir):
        execsummary.modify_pngs_scene_template(
            anat_file="anat.nii.gz",
            rh_pial_surf="rh_pial.surf.gii",
            lh_pial_surf="lh_pial.surf.gii",
            rh_wm_surf="rh_wm.surf.gii",
            lh_wm_surf="lh_wm.surf.gii",
            scene_template=pngs_scene_template,
        )

    assert os.path.isfile(os.path.join(tmpdir, "modified_scene.scene"))


def test_get_n_frames(fmriprep_with_freesurfer_data):
    """Test get_n_frames."""
    anat_file = fmriprep_with_freesurfer_data["brain_mask_file"]
    frame_numbers = execsummary.get_n_frames(anat_file)
    assert len(frame_numbers) == 194


def test_get_png_image_names():
    """Test get_png_image_names."""
    scene_index, image_descriptions = execsummary.get_png_image_names()
    assert len(scene_index) == len(image_descriptions) == 9
