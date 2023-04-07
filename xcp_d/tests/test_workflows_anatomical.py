"""Tests for the xcp_d.workflows.anatomical module."""
import os
import shutil

import pytest

from xcp_d.tests.utils import get_nodes
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
    }
    final_files = files.copy()
    for fref, fpath in files.items():
        std_fref = fref.replace("native_", "fsLR_")
        std_fname = os.path.basename(fpath)
        std_fname = std_fname.replace("sub-01_", "sub-01_space-fsLR_den-32k_")
        std_fpath = os.path.join(tmpdir, std_fname)
        shutil.copyfile(fpath, std_fpath)
        final_files[std_fref] = std_fpath

    return final_files


def test_warp_surfaces_to_template_wf(
    datasets,
    fmriprep_with_freesurfer_data,
    surface_files,
    tmp_path_factory,
):
    """Test surface-warping workflow with mesh surfaces available, but not in standard space.

    The transforms should be applied and all of the standard-space outputs should be generated.
    """
    tmpdir = tmp_path_factory.mktemp("test_warp_surfaces_to_template_wf")

    subject_id = "01"

    wf = anatomical.init_warp_surfaces_to_template_wf(
        fmri_dir=datasets["ds001419"],
        subject_id=subject_id,
        output_dir=tmpdir,
        omp_nthreads=1,
        mem_gb=0.1,
    )

    wf.inputs.inputnode.lh_pial_surf = surface_files["native_lh_pial"]
    wf.inputs.inputnode.rh_pial_surf = surface_files["native_rh_pial"]
    wf.inputs.inputnode.lh_wm_surf = surface_files["native_lh_wm"]
    wf.inputs.inputnode.rh_wm_surf = surface_files["native_rh_wm"]
    # transforms (only used if warp_to_standard is True)
    wf.inputs.inputnode.anat_to_template_xfm = fmriprep_with_freesurfer_data[
        "anat_to_template_xfm"
    ]
    wf.inputs.inputnode.template_to_anat_xfm = fmriprep_with_freesurfer_data[
        "template_to_anat_xfm"
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


def test_postprocess_anat_wf(fmriprep_with_freesurfer_data, tmp_path_factory):
    """Test xcp_d.workflows.anatomical.init_postprocess_anat_wf."""
    tmpdir = tmp_path_factory.mktemp("test_postprocess_anat_wf")

    anat_to_template_xfm = fmriprep_with_freesurfer_data["anat_to_template_xfm"]
    t1w = fmriprep_with_freesurfer_data["t1w"]
    anat_dseg = fmriprep_with_freesurfer_data["anat_dseg"]
    t2w = os.path.join(tmpdir, "sub-01_desc-preproc_T2w.nii.gz")  # pretend t1w is t2w
    shutil.copyfile(t1w, t2w)

    wf = anatomical.init_postprocess_anat_wf(
        output_dir=tmpdir,
        input_type="fmriprep",
        t1w_available=True,
        t2w_available=True,
        target_space="MNI152NLin2009cAsym",
        dcan_qc=False,
        omp_nthreads=1,
        mem_gb=0.1,
        name="postprocess_anat_wf",
    )
    wf.inputs.inputnode.anat_to_template_xfm = anat_to_template_xfm
    wf.inputs.inputnode.t1w = t1w
    wf.inputs.inputnode.anat_dseg = anat_dseg
    wf.inputs.inputnode.t2w = t2w
    wf.base_dir = tmpdir
    wf_res = wf.run()
    wf_nodes = get_nodes(wf_res)

    out_anat_dir = os.path.join(tmpdir, "xcp_d", "sub-01", "anat")
    out_t1w = wf_nodes["postprocess_anat_wf.ds_t1w_std"].get_output("out_file")
    assert os.path.isfile(out_t1w), os.listdir(out_anat_dir)

    out_t2w = wf_nodes["postprocess_anat_wf.ds_t2w_std"].get_output("out_file")
    assert os.path.isfile(out_t2w), os.listdir(out_anat_dir)

    out_anat_dseg = wf_nodes["postprocess_anat_wf.ds_anat_dseg_std"].get_output("out_file")
    assert os.path.isfile(out_anat_dseg), os.listdir(out_anat_dir)
