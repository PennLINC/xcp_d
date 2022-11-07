"""Confirm affines are not changing."""
import nibabel as nb
import os
import numpy as np
from bids import BIDSLayout
import sys
data_dir = sys.argv[1]
out_dir = sys.argv[2]
input_type = sys.argv[3]

#TODO: Index via desc

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
        bold_file = bold_file[0]
        denoised_file = denoised_file[0]
    elif input_type == 'nifti':  # Get the .nii.gz
        bold_file = fmri_layout.get(
            return_type="file",
            invalid_filters='allow',
            run=1,
            extension='.nii.gz',
            datatype='func'
        )
        bold_file = bold_file[-1]
        denoised_file = xcp_layout.get(
            return_type="file",
            invalid_filters='allow',
            run=1,
            extension='.nii.gz',
            datatype='func'
        )
        denoised_file = denoised_file[1]

    else:  # Nibabies
        bold_file = fmri_layout.get(
            extension='.nii.gz',
            return_type="file",
            invalid_filters='allow',
            datatype='func'
        )
        bold_file = bold_file[-1]
        denoised_file = xcp_layout.get(
            return_type="file",
            invalid_filters='allow',
            extension='.nii.gz',
            datatype='func')
        denoised_file = denoised_file[1]

    if input_type == 'cifti':
        assert nb.load(bold_file)._nifti_header.get_intent() == nb.load(
            denoised_file)._nifti_header.get_intent()
    else:
        assert np.array_equal(nb.load(bold_file).affine, nb.load(denoised_file).affine)

    print("No affines changed.")


test_affines(data_dir, out_dir, input_type)
