# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions to support ingestion of non-BIDS preprocessing derivatives."""
import json
import os

import numpy as np
from nilearn.maskers import NiftiMasker


def copy_file(src, dst):
    """Copy a file from source to dest.

    source and dest must be file-like objects,
    i.e. any object with a read or write method, like for example StringIO.
    """
    import filecmp
    import shutil

    if not os.path.exists(dst) or not filecmp.cmp(src, dst):
        shutil.copyfile(src, dst)


def extract_mean_signal(mask, nifti):
    """Extract mean signal within mask from NIFTI."""
    masker = NiftiMasker(mask_img=mask)
    signals = masker.fit_transform(nifti)
    return np.mean(signals, axis=1)


def write_json(data, outfile):
    """Write dictionary to JSON file."""
    with open(outfile, "w") as f:
        json.dump(data, f)
    return outfile


def plot_bbreg(fixed_image, moving_image, contour, out_file="report.svg"):
    """Plot bbref_fig_fmriprep results."""
    import numpy as np
    from nilearn.image import load_img, resample_img, threshold_img
    from niworkflows.viz.utils import compose_view, cuts_from_bbox, plot_registration

    fixed_image_nii = load_img(fixed_image)
    moving_image_nii = load_img(moving_image)
    moving_image_nii = resample_img(
        moving_image_nii, target_affine=np.eye(3), interpolation="nearest"
    )
    contour_nii = load_img(contour) if contour is not None else None

    mask_nii = threshold_img(fixed_image_nii, 1e-3)

    n_cuts = 7
    if contour_nii:
        cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
    else:
        cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

    compose_view(
        plot_registration(
            fixed_image_nii,
            "fixed-image",
            estimate_brightness=True,
            cuts=cuts,
            label="fixed",
            contour=contour_nii,
            compress="auto",
        ),
        plot_registration(
            moving_image_nii,
            "moving-image",
            estimate_brightness=True,
            cuts=cuts,
            label="moving",
            contour=contour_nii,
            compress="auto",
        ),
        out_file=out_file,
    )
    return out_file
