"""Tests for the xcp_d.interfaces.execsummary module."""

import os

import nibabel as nb

from xcp_d.interfaces import execsummary


def test_plotslicesforbrainsprite(datasets, tmp_path_factory):
    """Test the PlotSlicesForBrainSprite interface."""
    tmpdir = tmp_path_factory.mktemp('test_generate_confounds')

    anat_dir = os.path.join(datasets['pnc'], 'sub-1648798153', 'ses-PNC1', 'anat')

    anat = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_T1w.nii.gz')
    lh_pial = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_pial.surf.gii')
    lh_wm = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-L_white.surf.gii')
    rh_pial = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_pial.surf.gii')
    rh_wm = os.path.join(anat_dir, 'sub-1648798153_ses-PNC1_acq-refaced_hemi-R_white.surf.gii')

    img = nb.load(anat)
    n_slices = img.shape[0]

    interface = execsummary.PlotSlicesForBrainSprite(
        n_procs=1,
        lh_wm=lh_wm,
        lh_pial=lh_pial,
        rh_wm=rh_wm,
        rh_pial=rh_pial,
        nifti=anat,
    )
    results = interface.run(cwd=tmpdir)
    assert len(results.outputs.out_files) == n_slices
