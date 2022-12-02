"""Confirm affines are not changing."""
import nibabel as nb
import os
import numpy as np
from bids import BIDSLayout
import sys
data_dir = sys.argv[1]
out_dir = sys.argv[2]
input_type = sys.argv[3]


def test_affines(data_dir, out_dir, input_type):
    """Confirm affines don't change across XCP runs."""
    fmri_layout = BIDSLayout(str(data_dir), validate=False, derivatives=False)
    xcp_layout = BIDSLayout(str(out_dir), validate=False, derivatives=False)
    if input_type == 'cifti':  # Get the .dtseries.nii
        bold_file = fmri_layout.get(
            run=1,
            return_type="file",
            invalid_filters='allow',
            extension='.dtseries.nii',
            datatype='func'
        )
        denoised_file = xcp_layout.get(
            return_type="file",
            run=1,
            extension='.dtseries.nii',
            invalid_filters='allow',
            datatype='func'
        )

    elif input_type == 'nifti':  # Get the .nii.gz
        bold_file = fmri_layout.get(
            return_type="file",
            invalid_filters='allow',
            run=1,
            suffix='bold',
            extension='.nii.gz',
            datatype='func'
        )
        denoised_file = xcp_layout.get(
            return_type="file",
            run=1,
            suffix='bold',
            extension='.nii.gz',
            datatype='func'
        )

    else:  # Nibabies
        bold_file = fmri_layout.get(
            extension='.nii.gz',
            suffix='bold',
            return_type="file",
            space='MNIInfant',
            invalid_filters='allow',
            datatype='func'
        )
        denoised_file = xcp_layout.get(
            return_type="file",
            suffix='bold',
            space='MNIInfant',
            extension='.nii.gz',
            datatype='func')

    if isinstance(bold_file, list):
        bold_file = bold_file[0]
    if isinstance(denoised_file, list):
        denoised_file = denoised_file[0]

    if input_type == 'cifti':
        assert nb.load(bold_file)._nifti_header.get_intent() == nb.load(
            denoised_file)._nifti_header.get_intent()
    else:
        assert np.array_equal(nb.load(bold_file).affine, nb.load(denoised_file).affine)

    print("No affines changed.")


test_affines(data_dir, out_dir, input_type)
