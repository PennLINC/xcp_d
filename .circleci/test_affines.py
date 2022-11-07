"""Confirm affines are not changing."""
import nibabel as nb
from bids import BIDSLayout


def test_affines(data_dir, out_dir, cifti):
    """Confirm affines don't change across XCP runs."""
    fmri_layout = BIDSLayout(str(data_dir), validate=False, derivatives=True)
    xcp_layout = BIDSLayout(str(out_dir), validate=False, derivatives=True)
    if cifti:  # Get the .dtseries.nii
        bold_file = fmri_layout.get(
            run='1',
            extenstion='.dtseries.nii',
            type='func'
        )
        denoised_file = xcp_layout.get(
            desc='denoised',
            run='1',
            extenstion='.dtseries.nii',
            type='func'
        )
    else:  # Get the .nii.gz
        bold_file = fmri_layout.get(
            desc='preproc_bold',
            run='1',
            extenstion='.nii.gz',
            type='func'
        )
        denoised_file = xcp_layout.get(
            desc='denoised',
            run='1',
            extenstion='.nii.gz',
            type='func'
        )

    assert nb.load(bold_file).affine == nb.load(denoised_file).affine
