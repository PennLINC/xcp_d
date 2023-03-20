"""Tests for the xcp_d.workflow.anatomical module."""
import os
import shutil

import pytest

from xcp_d.tests.utils import get_nodes, get_test_data_path
from xcp_d.workflows import anatomical


@pytest.fixture
def surface_files(datasets):
    """Collect real and fake surface files to test the anatomical workflow."""
    anat_dir = os.path.join(datasets["ds001419"], "sub-01", "anat")

    return {
        "native_lh_pial": os.path.join(anat_dir, "sub-01_hemi-L_pial.surf.gii"),
        "native_lh_wm": os.path.join(anat_dir, "sub-01_hemi-L_smoothwm.surf.gii"),
        "native_rh_pial": os.path.join(anat_dir, "sub-01_hemi-R_pial.surf.gii"),
        "native_rh_wm": os.path.join(anat_dir, "sub-01_hemi-R_smoothwm.surf.gii"),
    }


def test_init_warp_surfaces_to_template_wf_01(
    datasets,
    fmriprep_with_freesurfer_data,
    surface_files,
    tmp_path_factory,
):
    """Test surface-warping workflow with mesh surfaces are available in standard space."""
    tmpdir = tmp_path_factory.mktemp("test_init_warp_surfaces_to_template_wf_01")

    subject_id = "01"

    wf = anatomical.init_warp_surfaces_to_template_wf(
        fmri_dir=datasets["ds001419"],
        subject_id=subject_id,
        output_dir=tmpdir,
        warp_to_standard=False,
        omp_nthreads=1,
        mem_gb=0.1,
    )

    wf.inputs.inputnode.lh_pial_surf = surface_files["fsLR_lh_pial"]
    wf.inputs.inputnode.rh_pial_surf = surface_files["fsLR_rh_pial"]
    wf.inputs.inputnode.lh_wm_surf = surface_files["fsLR_lh_wm"]
    wf.inputs.inputnode.rh_wm_surf = surface_files["fsLR_rh_wm"]
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
    """Test surface-warping workflow with mesh surfaces available, but not in standard space.

    The transforms should be applied and all of the standard-space outputs should be generated.
    """
    tmpdir = tmp_path_factory.mktemp("test_init_warp_surfaces_to_template_wf_02")

    test_data_dir = get_test_data_path()
    os.environ["FS_LICENSE"] = os.path.join(test_data_dir, "license.txt")

    subject_id = "01"

    wf = anatomical.init_warp_surfaces_to_template_wf(
        fmri_dir=datasets["ds001419"],
        subject_id=subject_id,
        output_dir=tmpdir,
        warp_to_standard=True,
        omp_nthreads=1,
        mem_gb=0.1,
    )

    wf.inputs.inputnode.lh_pial_surf = surface_files["native_lh_pial"]
    wf.inputs.inputnode.rh_pial_surf = surface_files["native_rh_pial"]
    wf.inputs.inputnode.lh_wm_surf = surface_files["native_lh_wm"]
    wf.inputs.inputnode.rh_wm_surf = surface_files["native_rh_wm"]
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


def test_warp_anats_to_template_wf(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.workflows.anatomical.init_warp_anats_to_template_wf."""
    tmpdir = tmp_path_factory.mktemp("test_nifti_conn")

    t1w_to_template_xfm = fmriprep_with_freesurfer_data["t1w_to_template_xfm"]
    t1w = fmriprep_with_freesurfer_data["t1w"]
    t1seg = fmriprep_with_freesurfer_data["t1seg"]
    t2w = os.path.join(tmpdir, "sub-01_desc-preproc_T2w.nii.gz")  # pretend t1w is t2w
    shutil.copyfile(t1w, t2w)

    wf = anatomical.init_warp_anats_to_template_wf(
        output_dir=tmpdir,
        input_type="fmriprep",
        t2w_available=True,
        target_space="MNI152NLin2009cAsym",
        omp_nthreads=1,
        mem_gb=0.1,
        name="warp_anats_to_template_wf",
    )
    wf.inputs.inputnode.t1w_to_template_xfm = t1w_to_template_xfm
    wf.inputs.inputnode.t1w = t1w
    wf.inputs.inputnode.t1seg = t1seg
    wf.inputs.inputnode.t2w = t2w
    wf.base_dir = tmpdir
    wf_res = wf.run()
    wf_nodes = get_nodes(wf_res)

    out_anat_dir = os.path.join(tmpdir, "xcp_d", "sub-01", "anat")
    out_t1w = wf_nodes["warp_anats_to_template_wf.ds_t1w_std"].get_output("out_file")
    assert os.path.isfile(out_t1w), os.listdir(out_anat_dir)

    out_t2w = wf_nodes["warp_anats_to_template_wf.ds_t2w_std"].get_output("out_file")
    assert os.path.isfile(out_t2w), os.listdir(out_anat_dir)

    out_t1seg = wf_nodes["warp_anats_to_template_wf.ds_t1seg_std"].get_output("out_file")
    assert os.path.isfile(out_t1seg), os.listdir(out_anat_dir)
