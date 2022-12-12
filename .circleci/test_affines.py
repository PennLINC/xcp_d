"""Confirm affines are not changing."""
import sys

import nibabel as nb
import numpy as np
from bids import BIDSLayout

data_dir = sys.argv[1]
out_dir = sys.argv[2]
input_type = sys.argv[3]


def test_affines(data_dir, out_dir, input_type):
    """Confirm affines don't change across XCP runs."""
    fmri_layout = BIDSLayout(str(data_dir), validate=False, derivatives=False)
    xcp_layout = BIDSLayout(str(out_dir), validate=False, derivatives=False)
    if input_type == "cifti":  # Get the .dtseries.nii
        denoised_files = xcp_layout.get(
            invalid_filters="allow",
            datatype="func",
            run=1,
            extension=".dtseries.nii",
        )
        space = denoised_files[0].get_entities()["space"]
        bold_files = fmri_layout.get(
            invalid_filters="allow",
            datatype="func",
            run=1,
            space=space,
            extension=".dtseries.nii",
        )

    elif input_type == "nifti":  # Get the .nii.gz
        # Problem: it's collecting native-space data
        denoised_files = xcp_layout.get(
            datatype="func",
            run=1,
            suffix="bold",
            extension=".nii.gz",
        )
        space = denoised_files[0].get_entities()["space"]
        bold_files = fmri_layout.get(
            invalid_filters="allow",
            datatype="func",
            run=1,
            space=space,
            suffix="bold",
            extension=".nii.gz",
        )

    else:  # Nibabies
        denoised_files = xcp_layout.get(
            datatype="func",
            space="MNIInfant",
            suffix="bold",
            extension=".nii.gz",
        )
        bold_files = fmri_layout.get(
            invalid_filters="allow",
            datatype="func",
            space="MNIInfant",
            suffix="bold",
            extension=".nii.gz",
        )

    bold_file = bold_files[0].path
    denoised_file = denoised_files[0].path

    if input_type == "cifti":
        assert (
            nb.load(bold_file)._nifti_header.get_intent()
            == nb.load(denoised_file)._nifti_header.get_intent()
        )
    else:
        if not np.array_equal(nb.load(bold_file).affine, nb.load(denoised_file).affine):
            raise AssertionError(f"Affines do not match:\n\t{bold_file}\n\t{denoised_file}")

    print("No affines changed.")


test_affines(data_dir, out_dir, input_type)
