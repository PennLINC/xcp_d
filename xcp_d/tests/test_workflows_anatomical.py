"""Tests for the xcp_d.workflows.anatomical.surface module."""

import os
import shutil

import pytest

from xcp_d import config
from xcp_d.tests.tests import mock_config
from xcp_d.tests.utils import get_nodes
from xcp_d.workflows import anatomical
from xcp_d.workflows.base import clean_datasinks


@pytest.fixture
def surface_files(datasets, tmp_path_factory):
    """Collect real and fake surface files to test the anatomical workflow."""
    tmpdir = tmp_path_factory.mktemp('surface_files')
    anat_dir = os.path.join(datasets['pnc'], 'sub-1648798153', 'ses-PNC1', 'anat')

    files = {
        'native_lh_pial': os.path.join(
            anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_pial.surf.gii'
        ),
        'native_lh_wm': os.path.join(
            anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_white.surf.gii'
        ),
        'native_rh_pial': os.path.join(
            anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_pial.surf.gii'
        ),
        'native_rh_wm': os.path.join(
            anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_white.surf.gii'
        ),
    }
    final_files = files.copy()
    for fref, fpath in files.items():
        std_fref = fref.replace('native_', 'fsLR_')
        std_fname = os.path.basename(fpath)
        std_fname = std_fname.replace(
            'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_',
            'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_space-fsLR_den-32k_',
        ).replace(
            'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_',
            'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_space-fsLR_den-32k_',
        )
        std_fpath = os.path.join(tmpdir, std_fname)
        shutil.copyfile(fpath, std_fpath)
        final_files[std_fref] = std_fpath

    final_files['lh_subject_sphere'] = os.path.join(
        anat_dir,
        'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_desc-reg_sphere.surf.gii',
    )
    final_files['rh_subject_sphere'] = os.path.join(
        anat_dir,
        'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_desc-reg_sphere.surf.gii',
    )

    return final_files


def test_fsnative_to_fsLR_wf(
    pnc_data,
    surface_files,
    tmp_path_factory,
):
    """Test surface-warping workflow with mesh surfaces available, but not in standard space.

    The transforms should be applied and workflow should run without errors.
    The workflow no longer writes out files to the output directory.
    """
    tmpdir = tmp_path_factory.mktemp('test_fsnative_to_fsLR_wf')

    with mock_config():
        config.nipype.omp_nthreads = 1
        config.execution.output_dir = tmpdir

        wf = anatomical.surface.init_fsnative_to_fsLR_wf(
            software='FreeSurfer',
            omp_nthreads=1,
        )

        wf.inputs.inputnode.lh_pial_surf = surface_files['native_lh_pial']
        wf.inputs.inputnode.rh_pial_surf = surface_files['native_rh_pial']
        wf.inputs.inputnode.lh_wm_surf = surface_files['native_lh_wm']
        wf.inputs.inputnode.rh_wm_surf = surface_files['native_rh_wm']
        wf.inputs.inputnode.lh_subject_sphere = surface_files['lh_subject_sphere']
        wf.inputs.inputnode.rh_subject_sphere = surface_files['rh_subject_sphere']
        wf.base_dir = tmpdir
        wf.run()


def test_itk_warp_gifti_surface_wf(
    pnc_data,
    surface_files,
    tmp_path_factory,
):
    """Test workflow that warps surfaces to standard space using antsApplyTransformsToPoints."""
    tmpdir = tmp_path_factory.mktemp('test_itk_warp_gifti_surface_wf')

    wf = anatomical.plotting.init_itk_warp_gifti_surface_wf(name='wf')
    wf.inputs.inputnode.native_surf_gii = surface_files['native_lh_pial']
    wf.inputs.inputnode.itk_warp_file = pnc_data['template_to_anat_xfm']
    wf.base_dir = tmpdir
    wf = clean_datasinks(wf)
    wf_res = wf.run()
    wf_nodes = get_nodes(wf_res)
    assert os.path.isfile(wf_nodes['wf.csv_to_gifti'].get_output('out_file'))


def test_postprocess_anat_wf(ds001419_data, tmp_path_factory):
    """Test xcp_d.workflows.anatomical.volume.init_postprocess_anat_wf."""
    tmpdir = tmp_path_factory.mktemp('test_postprocess_anat_wf')

    anat_to_template_xfm = ds001419_data['anat_to_template_xfm']
    t1w = ds001419_data['t1w']
    t2w = os.path.join(tmpdir, 'sub-01_desc-preproc_T2w.nii.gz')  # pretend t1w is t2w
    shutil.copyfile(t1w, t2w)

    with mock_config():
        config.execution.output_dir = tmpdir
        config.workflow.input_type = 'fmriprep'
        config.nipype.omp_nthreads = 1
        config.nipype.mem_gb = 0.1

        wf = anatomical.volume.init_postprocess_anat_wf(
            t1w_available=True,
            t2w_available=True,
            target_space='MNI152NLin2009cAsym',
            name='postprocess_anat_wf',
        )

        wf.inputs.inputnode.anat_to_template_xfm = anat_to_template_xfm
        wf.inputs.inputnode.t1w = t1w
        wf.inputs.inputnode.t2w = t2w
        wf.base_dir = tmpdir
        wf = clean_datasinks(wf)
        wf_res = wf.run()

        wf_nodes = get_nodes(wf_res)

        out_anat_dir = os.path.join(tmpdir, 'xcp_d', 'sub-01', 'anat')
        out_t1w = wf_nodes['postprocess_anat_wf.ds_t1w_std'].get_output('out_file')
        assert os.path.isfile(out_t1w), os.listdir(out_anat_dir)

        out_t2w = wf_nodes['postprocess_anat_wf.ds_t2w_std'].get_output('out_file')
        assert os.path.isfile(out_t2w), os.listdir(out_anat_dir)
