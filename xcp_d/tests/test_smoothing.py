"""Tests for smoothing methods."""

import os
import re
import tempfile

import numpy as np
from nipype.pipeline import engine as pe
from templateflow.api import get as get_template

from xcp_d.interfaces.nilearn import Smooth
from xcp_d.interfaces.workbench import CiftiSmooth
from xcp_d.utils.utils import fwhm2sigma


def test_smoothing_nifti(fmriprep_without_freesurfer_data):
    """Test NIFTI smoothing."""
    #  Specify inputs
    in_file = fmriprep_without_freesurfer_data['nifti_file']
    mask = fmriprep_without_freesurfer_data['brain_mask_file']

    # Let's get into a temp dir
    tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)

    # Run AFNI'S FWHMx via CLI, the nipype interface doesn't have what we need
    os.system(  # noqa: S605
        f'3dFWHMx -ShowMeClassicFWHM -acf -detrend -input {in_file} -mask {mask} '  # noqa: S605
        '-detprefix detrend.nii.gz -out test_file.out > test_fwhm.out'  # noqa: S605
    )  # noqa: S605

    # Read the FWHM values from the .out file into an array
    with open('test_fwhm.out') as file:
        first_line = file.readline()
    first_line = first_line.split()
    fwhm = []
    for item in first_line:
        item = float(item)
        fwhm.append(item)
    fwhm_unsmoothed = np.array(fwhm)

    # else this will need to be overwritten later
    os.remove('3dFWHMx.1D')
    os.remove('test_fwhm.out')
    os.remove('test_file.out')

    # Smooth the data
    smooth_data = pe.Node(
        Smooth(fwhm=6),
        name='nifti_smoothing',  # FWHM = kernel size
    )  # Use fslmaths to smooth the image
    smooth_data.inputs.in_file = in_file
    results = smooth_data.run()
    out_file = results.outputs.out_file

    # Run AFNI'S FWHMx via CLI, the nipype interface doesn't have what we need
    # i.e : the "ShowMeClassicFWHM" option
    os.system(  # noqa: S605
        f'3dFWHMx -ShowMeClassicFWHM -acf -detrend -input {out_file} -mask {mask} '  # noqa: S605
        '-detprefix detrend.nii.gz -out test_file.out > test_fwhm.out'  # noqa: S605
    )  # noqa: S605

    # Read the FWHM values from the .out file into an array
    with open('test_fwhm.out') as file:
        first_line = file.readline()
    first_line = first_line.split()
    fwhm = []
    for item in first_line:
        item = float(item)
        fwhm.append(item)
    fwhm_smoothed = np.array(fwhm)
    smoothed = np.sum((fwhm_smoothed) ** 2)
    unsmoothed = np.sum((fwhm_unsmoothed) ** 2)
    assert smoothed > unsmoothed
    return


def test_smoothing_cifti(ds001419_data, tmp_path_factory):
    """Test CIFTI smoothing."""
    sigma_lx = fwhm2sigma(6)

    tmpdir = tmp_path_factory.mktemp('test_smoothing_cifti')
    in_file = ds001419_data['cifti_file']
    # pull out atlases for each hemisphere
    right_surf = str(
        get_template(
            template='fsLR',
            space=None,
            hemi='R',
            density='32k',
            desc=None,
            suffix='sphere',
            raise_empty=True,
        )
    )
    left_surf = str(
        get_template(
            template='fsLR',
            space=None,
            hemi='L',
            density='32k',
            desc=None,
            suffix='sphere',
            raise_empty=True,
        )
    )

    # Estimate the smoothness of the unsmoothed file
    in_file_smoothness = os.popen(  # noqa: S605
        f'wb_command -cifti-estimate-fwhm {in_file} '  # noqa: S605
        f'-surface CORTEX_LEFT {left_surf} '  # noqa: S605
        f'-surface CORTEX_RIGHT {right_surf} '  # noqa: S605
        '-whole-file -merged-volume'  # noqa: S605
    ).read()  # noqa: S605
    in_file_smoothness = re.findall(r'\d.+', in_file_smoothness)
    in_file_smoothness = [x.split(',') for x in in_file_smoothness]
    in_file_smoothness = [item for sublist in in_file_smoothness for item in sublist]
    in_file_smoothness = list(map(float, in_file_smoothness))
    in_file_smoothness = np.sum(in_file_smoothness)

    # Smooth the file
    smooth_data = pe.Node(
        CiftiSmooth(
            sigma_surf=sigma_lx,  # the size of the surface kernel
            sigma_vol=sigma_lx,  # the volume of the surface kernel
            direction='COLUMN',  # which direction to smooth along@
            right_surf=right_surf,
            left_surf=left_surf,
            num_threads=1,
        ),
        name='cifti_smoothing',
        n_procs=1,
    )
    smooth_data.inputs.in_file = in_file
    smooth_data.base_dir = tmpdir
    smooth_data.inputs.out_file = os.path.join(tmpdir, 'test.dtseries.nii')
    results = smooth_data.run()
    out_file = results.outputs.out_file

    # Estimate the smoothness of the smoothed file
    out_file_smoothness = os.popen(  # noqa: S605
        f'wb_command -cifti-estimate-fwhm {out_file} '  # noqa: S605
        f'-surface CORTEX_LEFT {left_surf} '  # noqa: S605
        f'-surface CORTEX_RIGHT {right_surf} '  # noqa: S605
        '-whole-file -merged-volume'  # noqa: S605
    ).read()  # noqa: S605
    out_file_smoothness = re.findall(r'\d.+', out_file_smoothness)
    out_file_smoothness = [x.split(',') for x in out_file_smoothness]
    out_file_smoothness = [item for sublist in out_file_smoothness for item in sublist]
    out_file_smoothness = list(map(float, out_file_smoothness))
    out_file_smoothness = np.sum(out_file_smoothness)

    assert in_file_smoothness < out_file_smoothness
