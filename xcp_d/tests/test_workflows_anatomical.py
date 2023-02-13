"""Tests for the xcp_d.workflow.anatomical module."""
import os
import shutil

import pytest

from xcp_d.tests.utils import get_test_data_path
from xcp_d.workflows import anatomical


@pytest.fixture
def surface_files(datasets, tmp_path_factory):
    """Collect real and fake surface files to test the anatomical workflow."""
    tmpdir = tmp_path_factory.mktemp("surface_files")
    anat_dir = os.path.join(datasets["ds001419"], "sub-01", "anat")

    files = {
        "native_lh_pial": os.path.join(anat_dir, "sub-01_hemi-L_pial.surf.gii"),
        "native_lh_wm": os.path.join(anat_dir, "sub-01_hemi-L_smoothwm.surf.gii"),
        "native_rh_pial": os.path.join(anat_dir, "sub-01_hemi-R_pial.surf.gii"),
        "native_rh_wm": os.path.join(anat_dir, "sub-01_hemi-R_smoothwm.surf.gii"),
        # Copied files
        "native_lh_inflated": os.path.join(tmpdir, "sub-01_hemi-L_desc-hcp_inflated.surf.gii"),
        "native_lh_midthickness": os.path.join(
            tmpdir, "sub-01_hemi-L_desc-hcp_midthickness.surf.gii"
        ),
        "native_lh_vinflated": os.path.join(tmpdir, "sub-01_hemi-L_desc-hcp_vinflated.surf.gii"),
        "native_lh_sulcal_depth": os.path.join(tmpdir, "sub-01_hemi-L_sulc.shape.gii"),
        "native_lh_sulcal_curv": os.path.join(tmpdir, "sub-01_hemi-L_curv.shape.gii"),
        "native_lh_cortical_thickness": os.path.join(tmpdir, "sub-01_hemi-L_thickness.shape.gii"),
        "native_rh_inflated": os.path.join(tmpdir, "sub-01_hemi-R_desc-hcp_inflated.surf.gii"),
        "native_rh_midthickness": os.path.join(
            tmpdir, "sub-01_hemi-R_desc-hcp_midthickness.surf.gii"
        ),
        "native_rh_vinflated": os.path.join(tmpdir, "sub-01_hemi-R_desc-hcp_vinflated.surf.gii"),
        "native_rh_sulcal_depth": os.path.join(tmpdir, "sub-01_hemi-R_sulc.shape.gii"),
        "native_rh_sulcal_curv": os.path.join(tmpdir, "sub-01_hemi-R_curv.shape.gii"),
        "native_rh_cortical_thickness": os.path.join(tmpdir, "sub-01_hemi-R_thickness.shape.gii"),
    }

    shutil.copyfile(files["native_lh_pial"], files["native_lh_midthickness"])
    shutil.copyfile(files["native_lh_pial"], files["native_lh_inflated"])
    shutil.copyfile(files["native_lh_pial"], files["native_lh_vinflated"])
    shutil.copyfile(files["native_lh_pial"], files["native_lh_sulcal_depth"])
    shutil.copyfile(files["native_lh_pial"], files["native_lh_sulcal_curv"])
    shutil.copyfile(files["native_lh_pial"], files["native_lh_cortical_thickness"])
    shutil.copyfile(files["native_rh_pial"], files["native_rh_midthickness"])
    shutil.copyfile(files["native_rh_pial"], files["native_rh_inflated"])
    shutil.copyfile(files["native_rh_pial"], files["native_rh_vinflated"])
    shutil.copyfile(files["native_rh_pial"], files["native_rh_sulcal_depth"])
    shutil.copyfile(files["native_rh_pial"], files["native_rh_sulcal_curv"])
    shutil.copyfile(files["native_rh_pial"], files["native_rh_cortical_thickness"])

    final_files = files.copy()
    for fref, fpath in files.items():
        std_fref = fref.replace("native_", "fsLR_")
        std_fname = os.path.basename(fpath)
        std_fname = std_fname.replace("sub-01_", "sub-01_space-fsLR_den-32k_")
        std_fpath = os.path.join(tmpdir, std_fname)
        shutil.copyfile(fpath, std_fpath)
        final_files[std_fref] = std_fpath

    return final_files


def test_init_warp_surfaces_to_template_wf_01(
    datasets,
    fmriprep_with_freesurfer_data,
    surface_files,
    tmp_path_factory,
):
    """Test surface-warping workflow with all surfaces available in standard space."""
    tmpdir = tmp_path_factory.mktemp("test_init_warp_surfaces_to_template_wf_01")

    subject_id = "01"
    surfaces_found = {
        "mesh": True,
        "morphometry": True,
        "shape": True,
    }
    standard_spaces_available = {
        "mesh": True,
        "morphometry": True,
        "shape": True,
    }

    wf = anatomical.init_warp_surfaces_to_template_wf(
        fmri_dir=datasets["ds001419"],
        subject_id=subject_id,
        output_dir=tmpdir,
        surfaces_found=surfaces_found,
        standard_spaces_available=standard_spaces_available,
        omp_nthreads=1,
        mem_gb=0.1,
    )

    wf.inputs.inputnode.lh_pial_surf = surface_files["fsLR_lh_pial"]
    wf.inputs.inputnode.rh_pial_surf = surface_files["fsLR_rh_pial"]
    wf.inputs.inputnode.lh_wm_surf = surface_files["fsLR_lh_wm"]
    wf.inputs.inputnode.rh_wm_surf = surface_files["fsLR_rh_wm"]
    # optional surface morphometry files
    wf.inputs.inputnode.lh_midthickness_surf = surface_files["fsLR_lh_midthickness"]
    wf.inputs.inputnode.rh_midthickness_surf = surface_files["fsLR_rh_midthickness"]
    wf.inputs.inputnode.lh_inflated_surf = surface_files["fsLR_lh_inflated"]
    wf.inputs.inputnode.rh_inflated_surf = surface_files["fsLR_rh_inflated"]
    wf.inputs.inputnode.lh_vinflated_surf = surface_files["fsLR_lh_vinflated"]
    wf.inputs.inputnode.rh_vinflated_surf = surface_files["fsLR_rh_vinflated"]
    # optional surface shape files
    wf.inputs.inputnode.lh_sulcal_depth = surface_files["fsLR_lh_sulcal_depth"]
    wf.inputs.inputnode.rh_sulcal_depth = surface_files["fsLR_rh_sulcal_depth"]
    wf.inputs.inputnode.lh_sulcal_curv = surface_files["fsLR_lh_sulcal_curv"]
    wf.inputs.inputnode.rh_sulcal_curv = surface_files["fsLR_rh_sulcal_curv"]
    wf.inputs.inputnode.lh_cortical_thickness = surface_files["fsLR_lh_cortical_thickness"]
    wf.inputs.inputnode.rh_cortical_thickness = surface_files["fsLR_rh_cortical_thickness"]
    # transforms (only used if warp_to_standard is True)
    wf.inputs.inputnode.t1w_to_template_xform = fmriprep_with_freesurfer_data[
        "t1w_to_template_xform"
    ]
    wf.inputs.inputnode.template_to_t1w_xform = fmriprep_with_freesurfer_data[
        "template_to_t1w_xform"
    ]

    wf.base_dir = tmpdir
    wf.run()

    # All of the possible fsLR surfaces should be available.
    out_anat_dir = os.path.join(tmpdir, "xcp_d", "sub-01", "anat")
    for key, filename in surface_files.items():
        if "fsLR" in key:
            out_fname = os.path.basename(filename)
            out_file = os.path.join(out_anat_dir, out_fname)
            assert os.path.isfile(out_file)


def test_init_warp_surfaces_to_template_wf_02(
    datasets,
    fmriprep_with_freesurfer_data,
    surface_files,
    tmp_path_factory,
):
    """Test surface-warping workflow with all surfaces available, but none in standard space."""
    tmpdir = tmp_path_factory.mktemp("test_init_warp_surfaces_to_template_wf_02")

    test_data_dir = get_test_data_path()
    os.environ["FS_LICENSE"] = os.path.join(test_data_dir, "license.txt")

    subject_id = "01"
    surfaces_found = {
        "mesh": True,
        "morphometry": True,
        "shape": True,
    }
    standard_spaces_available = {
        "mesh": False,
        "morphometry": False,
        "shape": False,
    }

    wf = anatomical.init_warp_surfaces_to_template_wf(
        fmri_dir=datasets["ds001419"],
        subject_id=subject_id,
        output_dir=tmpdir,
        surfaces_found=surfaces_found,
        standard_spaces_available=standard_spaces_available,
        omp_nthreads=1,
        mem_gb=0.1,
    )

    wf.inputs.inputnode.lh_pial_surf = surface_files["native_lh_pial"]
    wf.inputs.inputnode.rh_pial_surf = surface_files["native_rh_pial"]
    wf.inputs.inputnode.lh_wm_surf = surface_files["native_lh_wm"]
    wf.inputs.inputnode.rh_wm_surf = surface_files["native_rh_wm"]
    # optional surface morphometry files
    wf.inputs.inputnode.lh_midthickness_surf = surface_files["native_lh_midthickness"]
    wf.inputs.inputnode.rh_midthickness_surf = surface_files["native_rh_midthickness"]
    wf.inputs.inputnode.lh_inflated_surf = surface_files["native_lh_inflated"]
    wf.inputs.inputnode.rh_inflated_surf = surface_files["native_rh_inflated"]
    wf.inputs.inputnode.lh_vinflated_surf = surface_files["native_lh_vinflated"]
    wf.inputs.inputnode.rh_vinflated_surf = surface_files["native_rh_vinflated"]
    # optional surface shape files
    wf.inputs.inputnode.lh_sulcal_depth = surface_files["native_lh_sulcal_depth"]
    wf.inputs.inputnode.rh_sulcal_depth = surface_files["native_rh_sulcal_depth"]
    wf.inputs.inputnode.lh_sulcal_curv = surface_files["native_lh_sulcal_curv"]
    wf.inputs.inputnode.rh_sulcal_curv = surface_files["native_rh_sulcal_curv"]
    wf.inputs.inputnode.lh_cortical_thickness = surface_files["native_lh_cortical_thickness"]
    wf.inputs.inputnode.rh_cortical_thickness = surface_files["native_rh_cortical_thickness"]
    # transforms (only used if warp_to_standard is True)
    wf.inputs.inputnode.t1w_to_template_xform = fmriprep_with_freesurfer_data[
        "t1w_to_template_xform"
    ]
    wf.inputs.inputnode.template_to_t1w_xform = fmriprep_with_freesurfer_data[
        "template_to_t1w_xform"
    ]

    wf.base_dir = tmpdir
    wf.run()

    # All of the possible fsLR surfaces should be available.
    out_anat_dir = os.path.join(tmpdir, "xcp_d", "sub-01", "anat")
    for key, filename in surface_files.items():
        if "fsLR" in key:
            out_fname = os.path.basename(filename)
            out_file = os.path.join(out_anat_dir, out_fname)
            assert os.path.isfile(out_file), "\n".join(sorted(os.listdir(out_anat_dir)))


def test_init_warp_surfaces_to_template_wf_03(
    datasets,
    fmriprep_with_freesurfer_data,
    surface_files,
    tmp_path_factory,
):
    """Test surface-warping workflow with all surfaces available and some in standard space.

    With standard-space meshes, but native-space morphometries and shapes,
    the morphometries will be generated from scratch using the standard-space meshes,
    while the shapes will be warped from native to standard space.
    """
    tmpdir = tmp_path_factory.mktemp("test_init_warp_surfaces_to_template_wf_03")

    test_data_dir = get_test_data_path()
    os.environ["FS_LICENSE"] = os.path.join(test_data_dir, "license.txt")

    subject_id = "01"
    surfaces_found = {
        "mesh": True,
        "morphometry": True,
        "shape": True,
    }
    standard_spaces_available = {
        "mesh": True,
        "morphometry": False,
        "shape": False,
    }

    wf = anatomical.init_warp_surfaces_to_template_wf(
        fmri_dir=datasets["ds001419"],
        subject_id=subject_id,
        output_dir=tmpdir,
        surfaces_found=surfaces_found,
        standard_spaces_available=standard_spaces_available,
        omp_nthreads=1,
        mem_gb=0.1,
    )

    wf.inputs.inputnode.lh_pial_surf = surface_files["fsLR_lh_pial"]
    wf.inputs.inputnode.rh_pial_surf = surface_files["fsLR_rh_pial"]
    wf.inputs.inputnode.lh_wm_surf = surface_files["fsLR_lh_wm"]
    wf.inputs.inputnode.rh_wm_surf = surface_files["fsLR_rh_wm"]
    # optional surface morphometry files
    wf.inputs.inputnode.lh_midthickness_surf = surface_files["native_lh_midthickness"]
    wf.inputs.inputnode.rh_midthickness_surf = surface_files["native_rh_midthickness"]
    wf.inputs.inputnode.lh_inflated_surf = surface_files["native_lh_inflated"]
    wf.inputs.inputnode.rh_inflated_surf = surface_files["native_rh_inflated"]
    wf.inputs.inputnode.lh_vinflated_surf = surface_files["native_lh_vinflated"]
    wf.inputs.inputnode.rh_vinflated_surf = surface_files["native_rh_vinflated"]
    # optional surface shape files
    wf.inputs.inputnode.lh_sulcal_depth = surface_files["native_lh_sulcal_depth"]
    wf.inputs.inputnode.rh_sulcal_depth = surface_files["native_rh_sulcal_depth"]
    wf.inputs.inputnode.lh_sulcal_curv = surface_files["native_lh_sulcal_curv"]
    wf.inputs.inputnode.rh_sulcal_curv = surface_files["native_rh_sulcal_curv"]
    wf.inputs.inputnode.lh_cortical_thickness = surface_files["native_lh_cortical_thickness"]
    wf.inputs.inputnode.rh_cortical_thickness = surface_files["native_rh_cortical_thickness"]
    # transforms (only used if warp_to_standard is True)
    wf.inputs.inputnode.t1w_to_template_xform = fmriprep_with_freesurfer_data[
        "t1w_to_template_xform"
    ]
    wf.inputs.inputnode.template_to_t1w_xform = fmriprep_with_freesurfer_data[
        "template_to_t1w_xform"
    ]

    wf.base_dir = tmpdir
    wf.run()

    # All of the possible fsLR surfaces should be available.
    out_anat_dir = os.path.join(tmpdir, "xcp_d", "sub-01", "anat")
    for key, filename in surface_files.items():
        if "fsLR" in key:
            out_fname = os.path.basename(filename)
            out_file = os.path.join(out_anat_dir, out_fname)
            assert os.path.isfile(out_file)


def test_init_warp_surfaces_to_template_wf_04(
    datasets,
    fmriprep_with_freesurfer_data,
    surface_files,
    tmp_path_factory,
):
    """Test surface-warping workflow with all surfaces available and some in standard space.

    With standard-space meshes, but native-space morphometries and shapes,
    the morphometries will be generated from scratch using the standard-space meshes,
    while the shapes will be warped from native to standard space.
    """
    tmpdir = tmp_path_factory.mktemp("test_init_warp_surfaces_to_template_wf_04")

    test_data_dir = get_test_data_path()
    os.environ["FS_LICENSE"] = os.path.join(test_data_dir, "license.txt")

    subject_id = "01"
    surfaces_found = {
        "mesh": True,
        "morphometry": True,
        "shape": True,
    }
    standard_spaces_available = {
        "mesh": False,
        "morphometry": True,
        "shape": True,
    }

    wf = anatomical.init_warp_surfaces_to_template_wf(
        fmri_dir=datasets["ds001419"],
        subject_id=subject_id,
        output_dir=tmpdir,
        surfaces_found=surfaces_found,
        standard_spaces_available=standard_spaces_available,
        omp_nthreads=1,
        mem_gb=0.1,
    )

    wf.inputs.inputnode.lh_pial_surf = surface_files["fsLR_lh_pial"]
    wf.inputs.inputnode.rh_pial_surf = surface_files["fsLR_rh_pial"]
    wf.inputs.inputnode.lh_wm_surf = surface_files["fsLR_lh_wm"]
    wf.inputs.inputnode.rh_wm_surf = surface_files["fsLR_rh_wm"]
    # optional surface morphometry files
    wf.inputs.inputnode.lh_midthickness_surf = surface_files["native_lh_midthickness"]
    wf.inputs.inputnode.rh_midthickness_surf = surface_files["native_rh_midthickness"]
    wf.inputs.inputnode.lh_inflated_surf = surface_files["native_lh_inflated"]
    wf.inputs.inputnode.rh_inflated_surf = surface_files["native_rh_inflated"]
    wf.inputs.inputnode.lh_vinflated_surf = surface_files["native_lh_vinflated"]
    wf.inputs.inputnode.rh_vinflated_surf = surface_files["native_rh_vinflated"]
    # optional surface shape files
    wf.inputs.inputnode.lh_sulcal_depth = surface_files["native_lh_sulcal_depth"]
    wf.inputs.inputnode.rh_sulcal_depth = surface_files["native_rh_sulcal_depth"]
    wf.inputs.inputnode.lh_sulcal_curv = surface_files["native_lh_sulcal_curv"]
    wf.inputs.inputnode.rh_sulcal_curv = surface_files["native_rh_sulcal_curv"]
    wf.inputs.inputnode.lh_cortical_thickness = surface_files["native_lh_cortical_thickness"]
    wf.inputs.inputnode.rh_cortical_thickness = surface_files["native_rh_cortical_thickness"]
    # transforms (only used if warp_to_standard is True)
    wf.inputs.inputnode.t1w_to_template_xform = fmriprep_with_freesurfer_data[
        "t1w_to_template_xform"
    ]
    wf.inputs.inputnode.template_to_t1w_xform = fmriprep_with_freesurfer_data[
        "template_to_t1w_xform"
    ]

    wf.base_dir = tmpdir
    wf.run()

    # All of the possible fsLR surfaces should be available.
    out_anat_dir = os.path.join(tmpdir, "xcp_d", "sub-01", "anat")
    for key, filename in surface_files.items():
        if "fsLR" in key:
            out_fname = os.path.basename(filename)
            out_file = os.path.join(out_anat_dir, out_fname)
            assert os.path.isfile(out_file)
