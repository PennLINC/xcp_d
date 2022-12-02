"""Tests for smoothing methods."""
import os
import tempfile

import numpy as np
from nipype.pipeline import engine as pe

from xcp_d.interfaces.nilearn import Smooth


def test_smoothing_nifti(fmriprep_without_freesurfer_data):
    """Test NIFTI smoothing."""
    #  Specify inputs
    in_file = fmriprep_without_freesurfer_data["nifti_file"]
    mask = fmriprep_without_freesurfer_data["brain_mask_file"]

    # Let's get into a temp dir
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    # Run AFNI'S FWHMx via CLI, the nipype interface doesn't have what we need
    os.system(
        (
            f'3dFWHMx -ShowMeClassicFWHM -acf -detrend -input {in_file} -mask {mask} '
            '-detprefix detrend.nii.gz -out test_file.out > test_fwhm.out'
        )
    )

    # Read the FWHM values from the .out file into an array
    with open("test_fwhm.out", "r") as file:
        first_line = file.readline()
    first_line = first_line.split()
    fwhm = []
    for item in first_line:
        item = float(item)
        fwhm.append(item)
    fwhm_unsmoothed = np.array(fwhm)

    # else this will need to be overwritten later
    os.system('rm -rf 3dFWHMx.1D test_fwhm.out test_file.out')

    # Smooth the data
    smooth_data = pe.Node(Smooth(fwhm=6),  # FWHM = kernel size
                          name="nifti_smoothing")  # Use fslmaths to smooth the image
    smooth_data.inputs.in_file = in_file
    results = smooth_data.run()
    out_file = results.outputs.out_file

    # Run AFNI'S FWHMx via CLI, the nipype interface doesn't have what we need
    # i.e : the "ShowMeClassicFWHM" option
    os.system(
        (
            f'3dFWHMx -ShowMeClassicFWHM -acf -detrend -input {out_file} -mask {mask} '
            '-detprefix detrend.nii.gz -out test_file.out > test_fwhm.out'
        )
    )

    # Read the FWHM values from the .out file into an array
    with open("test_fwhm.out", "r") as file:
        first_line = file.readline()
    first_line = first_line.split()
    fwhm = []
    for item in first_line:
        item = float(item)
        fwhm.append(item)
    fwhm_smoothed = np.array(fwhm)
    smoothed = np.sum((fwhm_smoothed)**2)
    unsmoothed = np.sum((fwhm_unsmoothed)**2)
    assert smoothed > unsmoothed
    return


# TODO: SMOOTHING ESTIMATIONS VIA CONNECTOME WORKBENCH ARE A ROADMAP ITEM
# establish necessary variables
# smoothing = 6
# # turn into standard deviation
# from xcp_d.utils.utils import fwhm2sigma
# sigma_lx = fwhm2sigma(smoothing)
# def test_smoothing_Cifti(fmriprep_with_freesurfer_data, sigma_lx):
#     # Specify inputs
#     in_file = fmriprep_with_freesurfer_data["cifti_file"]
#
#     # Let's get into a temp dir
#     tmpdir = tempfile.mkdtemp()
#     os.chdir(tmpdir)
#
#     # Run AFNI'S FWHMx
#     fwhm = afni.FWHMx()
#     fwhm.inputs.in_file = in_file
#     fwhm.inputs.detrend = True
#     results = fwhm.run()
#     in_file_smoothness = results.outputs.fwhm
#
#     # Smooth the data
#     smooth_data = pe.Node(CiftiSmooth(  # Call connectome workbench to smooth for each
#         #  hemisphere
#         sigma_surf=sigma_lx,  # the size of the surface kernel
#         sigma_vol=sigma_lx,  # the volume of the surface kernel
#         direction='COLUMN',  # which direction to smooth along@
#         right_surf=pkgrf(  # pull out atlases for each hemisphere
#             'xcp_d', 'data/ciftiatlas/'
#             'Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii'
#         ),
#         left_surf=pkgrf(
#             'xcp_d', 'data/ciftiatlas/'
#             'Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii'
#         )),
#         name="cifti_smoothing")
#     smooth_data.inputs.in_file = in_file
#     smooth_data.inputs.out_file = 'test.dtseries.nii'
#     results = smooth_data.run()
#     out_file = results.outputs.out_file
#
#     # Run AFNI's FWHMx on the smoothed data
#     fwhm = afni.FWHMx()
#     fwhm.inputs.in_file = out_file
#     fwhm.inputs.detrend = True
#     results = fwhm.run()
#     out_file_smoothness = results.outputs.fwhm
#
#     print(in_file_smoothness)
#     print(out_file_smoothness)
