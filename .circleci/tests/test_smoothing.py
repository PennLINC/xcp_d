"""Tests for smoothing methods."""
import os
import tempfile
import re

import numpy as np
from nipype.interfaces.workbench import CiftiSmooth
from nipype.pipeline import engine as pe
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.nilearn import Smooth
from xcp_d.utils.utils import fwhm2sigma


def test_smoothing_Nifti(data_dir):
    """Test NIFTI smoothing."""
    #  Specify inputs
    data_dir = os.path.join(data_dir,
                            "fmriprepwithfreesurfer")
    in_file = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    mask = data_dir + "/fmriprep/sub-colornest001/ses-1/func/" \
        "sub-colornest001_ses-1_task-rest_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
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


def test_smoothing_Cifti(data_dir, sigma_lx=fwhm2sigma(6)):

    # Specify inputs
    right_surf = pkgrf(  # pull out atlases for each hemisphere
        'xcp_d', 'data/ciftiatlas/'
        'Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii'
    )
    left_surf = pkgrf(
        'xcp_d', 'data/ciftiatlas/'
        'Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii'
    )

    data_dir = os.path.join(data_dir,
                            "fmriprepwithfreesurfer")
    in_file = os.path.join(
        data_dir,
        "fmriprep/sub-colornest001/ses-1/func",
        "sub-colornest001_ses-1_task-rest_run-1_space-fsLR_den-91k_bold.dtseries.nii")
    # What's the smoothness?
    in_file_smoothness = os.popen("wb_command -cifti-estimate-fwhm " + in_file
                                  + " -surface CORTEX_LEFT " + left_surf
                                  + " -surface CORTEX_RIGHT " + right_surf
                                  + " -whole-file -merged-volume").read()

    # Let's get into a temp dir
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    smooth_data = pe.Node(CiftiSmooth(  # Call connectome workbench to smooth for each
        #  hemisphere
        sigma_surf=sigma_lx,  # the size of the surface kernel
        sigma_vol=sigma_lx,  # the volume of the surface kernel
        direction='COLUMN',  # which direction to smooth along@
        right_surf=pkgrf(  # pull out atlases for each hemisphere
            'xcp_d', 'data/ciftiatlas/'
            'Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii'
        ),
        left_surf=pkgrf(
            'xcp_d', 'data/ciftiatlas/'
            'Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii'
        )),
        name="cifti_smoothing")
    smooth_data.inputs.in_file = in_file
    smooth_data.base_dir = os.getcwd()
    smooth_data.inputs.out_file = 'test.dtseries.nii'
    results = smooth_data.run()
    out_file = results.outputs.out_file

    # What's the smoothness?
    out_file_smoothness = os.popen("wb_command -cifti-estimate-fwhm " + out_file
                                   + " -surface CORTEX_LEFT " + left_surf
                                   + " -surface CORTEX_RIGHT " + right_surf
                                   + " -whole-file -merged-volume").read()
    smoothness = re.findall(r'\d.+', in_file_smoothness)
    in_file_smoothness = [x.split(',') for x in smoothness]
    in_file_smoothness = [item for sublist in in_file_smoothness for item in sublist]
    in_file_smoothness = list(map(float, in_file_smoothness))
    in_file_smoothness = np.sum((in_file_smoothness))

    smoothness = re.findall(r'\d.+', out_file_smoothness)
    out_file_smoothness = [x.split(',') for x in smoothness]
    out_file_smoothness = [item for sublist in out_file_smoothness for item in sublist]
    out_file_smoothness = list(map(float, out_file_smoothness))
    out_file_smoothness = np.sum((out_file_smoothness))
    assert in_file_smoothness < out_file_smoothness
