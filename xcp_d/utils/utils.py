#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Miscellaneous utility functions for xcp_d."""
import glob
import os
import tempfile
from pathlib import Path

import nibabel as nb
import numpy as np
from nipype.interfaces.ants import ApplyTransforms
from scipy.signal import butter, filtfilt
from templateflow.api import get as get_template


def _t12native(fname):
    """Select T1w-to-scanner transform associated with a given BOLD file.

    TODO: Update names and refactor

    Parameters
    ----------
    fname : str
        The BOLD file from which to identify the transform.

    Returns
    -------
    t1w_to_native_xform : str
        Path to the T1w-to-scanner transform.

    Notes
    -----
    Only used in get_segfile, which should be removed ASAP.
    """
    import os

    pth, fname = os.path.split(fname)
    file_prefix = fname.split("space-")[0]
    t1w_to_native_xform = os.path.join(pth, f"{file_prefix}from-T1w_to-scanner_mode-image_xfm.txt")

    if not os.path.isfile(t1w_to_native_xform):
        raise FileNotFoundError(f"File not found: {t1w_to_native_xform}")

    return t1w_to_native_xform


def get_segfile(bold_file):
    """Select the segmentation file associated with a given BOLD file.

    This function identifies the appropriate MNI-space discrete segmentation file for carpet
    plots, then applies the necessary transforms to warp the file into BOLD reference space.
    The warped segmentation file will be written to a temporary file and its path returned.

    Parameters
    ----------
    bold_file : str
        Path to the BOLD file.

    Returns
    -------
    segfile : str
        The associated segmentation file.

    Notes
    -----
    Only used in concatenation code and should be dropped in favor of BIDSLayout methods ASAP.
    """
    # get transform files
    dd = Path(os.path.dirname(bold_file))
    anatdir = str(dd.parent) + "/anat"

    if Path(anatdir).is_dir():
        mni_to_t1 = glob.glob(anatdir + "/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5")[0]
    else:
        anatdir = str(dd.parent.parent) + "/anat"
        mni_to_t1 = glob.glob(anatdir + "/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5")[0]

    transformfilex = get_std2bold_xforms(
        bold_file=bold_file,
        template_to_t1w=mni_to_t1,
        t1w_to_native=_t12native(bold_file),
    )

    boldref = bold_file.split("desc-preproc_bold.nii.gz")[0] + "boldref.nii.gz"

    segfile = tempfile.mkdtemp() + "segfile.nii.gz"
    carpet = str(
        get_template(
            "MNI152NLin2009cAsym",
            resolution=1,
            desc="carpet",
            suffix="dseg",
            extension=[".nii", ".nii.gz"],
        ),
    )

    # seg_data file to bold space
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = carpet
    at.inputs.reference_image = boldref
    at.inputs.output_image = segfile
    at.inputs.interpolation = "MultiLabel"
    at.inputs.transforms = transformfilex
    os.system(at.cmdline)

    return segfile


def get_bold2std_and_t1w_xforms(bold_file, template_to_t1w, t1w_to_native):
    """Find transform files in reverse order to transform BOLD to MNI152NLin2009cAsym/T1w space.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    template_to_t1w : str
        The MNI-to-T1w transform file.
        The ``from`` field is assumed to be the same space as the BOLD file is in.
        The MNI space could be MNI152NLin2009cAsym, MNI152NLin6Asym, or MNIInfant.
    t1w_to_native : str
        The T1w-to-native space transform file.

    Returns
    -------
    xforms_to_MNI : list of str
        A list of paths to transform files for warping to MNI152NLin2009cAsym space.
    xforms_to_MNI_invert : list of bool
        A list of booleans indicating whether each transform in xforms_to_MNI indicating
        if each should be inverted (True) or not (False).
    xforms_to_T1w : list of str
        A list of paths to transform files for warping to T1w space.
    xforms_to_T1w_invert : list of bool
        A list of booleans indicating whether each transform in xforms_to_T1w indicating
        if each should be inverted (True) or not (False).

    Notes
    -----
    Only used for QCReport in init_boldpostprocess_wf.
    QCReport wants MNI-space data in MNI152NLin2009cAsym.
    """
    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    from xcp_d.utils.bids import get_entity

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, "space")

    if bold_space in ("native", "T1w"):
        base_std_space = get_entity(template_to_t1w, "from")
    elif f"from-{bold_space}" not in template_to_t1w:
        raise ValueError(f"Transform does not match BOLD space: {bold_space} != {template_to_t1w}")

    # Pull out the correct transforms based on bold_file name and string them together.
    xforms_to_T1w = [template_to_t1w]  # used for all spaces except T1w and native
    xforms_to_T1w_invert = [False]
    if bold_space == "MNI152NLin2009cAsym":
        # Data already in MNI152NLin2009cAsym space.
        xforms_to_MNI = ["identity"]
        xforms_to_MNI_invert = [False]

    elif bold_space == "MNI152NLin6Asym":
        # MNI152NLin6Asym --> MNI152NLin2009cAsym
        MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
            get_template(
                template="MNI152NLin2009cAsym",
                mode="image",
                suffix="xfm",
                extension=".h5",
                **{"from": "MNI152NLin6Asym"},
            ),
        )
        xforms_to_MNI = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]
        xforms_to_MNI_invert = [False]

    elif bold_space == "MNIInfant":
        # MNIInfant --> MNI152NLin2009cAsym
        MNIInfant_to_MNI152NLin2009cAsym = pkgrf(
            "xcp_d",
            "data/transform/tpl-MNIInfant_from-MNI152NLin2009cAsym_mode-image_xfm.h5",
        )
        xforms_to_MNI = [MNIInfant_to_MNI152NLin2009cAsym]
        xforms_to_MNI_invert = [False]

    elif bold_space == "T1w":
        # T1w --> ?? (extract from template_to_t1w) --> MNI152NLin2009cAsym
        # Should not be reachable, since xcpd doesn't support T1w-space BOLD inputs
        if base_std_space != "MNI152NLin2009cAsym":
            std_to_mni_xform = str(
                get_template(
                    template="MNI152NLin2009cAsym",
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": base_std_space},
                ),
            )
            xforms_to_MNI = [std_to_mni_xform, template_to_t1w]
            xforms_to_MNI_invert = [False, True]
        else:
            xforms_to_MNI = [template_to_t1w]
            xforms_to_MNI_invert = [True]

        xforms_to_T1w = ["identity"]
        xforms_to_T1w_invert = [False]

    elif bold_space == "native":
        # native (BOLD) --> T1w --> ?? (extract from template_to_t1w) --> MNI152NLin2009cAsym
        # Should not be reachable, since xcpd doesn't support native-space BOLD inputs
        if base_std_space != "MNI152NLin2009cAsym":
            std_to_mni_xform = str(
                get_template(
                    template="MNI152NLin2009cAsym",
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": base_std_space},
                ),
            )
            xforms_to_MNI = [std_to_mni_xform, template_to_t1w, t1w_to_native]
            xforms_to_MNI_invert = [False, True, True]
        else:
            xforms_to_MNI = [template_to_t1w, t1w_to_native]
            xforms_to_MNI_invert = [True, True]

        xforms_to_T1w = [t1w_to_native]
        xforms_to_T1w_invert = [True]

    else:
        raise ValueError(f"Space '{bold_space}' in {bold_file} not supported.")

    return xforms_to_MNI, xforms_to_MNI_invert, xforms_to_T1w, xforms_to_T1w_invert


def get_std2bold_xforms(bold_file, template_to_t1w, t1w_to_native):
    """Obtain transforms to warp atlases from MNI152NLin6Asym to the same space as the BOLD.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    template_to_t1w : str
        The MNI-to-T1w transform file.
        The ``from`` field is assumed to be the same space as the BOLD file is in.
    t1w_to_native : str
        The T1w-to-native space transform file.

    Returns
    -------
    transform_list : list of str
        A list of paths to transform files.

    Notes
    -----
    Used by:

    - get_segfile (to be removed)
    - to resample dseg in init_boldpostprocess_wf for QCReport
    - to warp atlases to the same space as the BOLD data in init_nifti_functional_connectivity_wf
    - to resample dseg to BOLD space for the executive summary plots

    Does not include inversion flag output because there is no need (yet).
    Can easily be added in the future.
    """
    import os

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    from xcp_d.utils.bids import get_entity

    # Extract the space of the BOLD file
    bold_space = get_entity(bold_file, "space")

    # Check that the MNI-to-T1w xform is from the right space
    if bold_space in ("native", "T1w"):
        base_std_space = get_entity(template_to_t1w, "from")
    elif f"from-{bold_space}" not in template_to_t1w:
        raise ValueError(f"Transform does not match BOLD space: {bold_space} != {template_to_t1w}")

    # Load useful inter-template transforms from templateflow
    MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
        get_template(
            template="MNI152NLin2009cAsym",
            mode="image",
            suffix="xfm",
            extension=".h5",
            **{"from": "MNI152NLin6Asym"},
        ),
    )

    # Find the appropriate transform(s)
    if bold_space == "MNI152NLin6Asym":
        # NLin6 --> NLin6 (identity)
        transform_list = ["identity"]

    elif bold_space == "MNI152NLin2009cAsym":
        # NLin6 --> NLin2009c
        transform_list = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]

    elif bold_space == "MNIInfant":
        # NLin6 --> NLin2009c --> MNIInfant
        MNI152NLin2009cAsym_to_MNI152Infant = pkgrf(
            "xcp_d",
            "data/transform/tpl-MNIInfant_from-MNI152NLin2009cAsym_mode-image_xfm.h5",
        )
        transform_list = [
            MNI152NLin2009cAsym_to_MNI152Infant,
            MNI152NLin6Asym_to_MNI152NLin2009cAsym,
        ]

    elif bold_space == "T1w":
        # NLin6 --> ?? (extract from template_to_t1w) --> T1w (BOLD)
        if base_std_space != "MNI152NLin6Asym":
            mni_to_std_xform = str(
                get_template(
                    template=base_std_space,
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": "MNI152NLin6Asym"},
                ),
            )
            transform_list = [template_to_t1w, mni_to_std_xform]
        else:
            transform_list = [template_to_t1w]

    elif bold_space == "native":
        # The BOLD data are in native space
        # NLin6 --> ?? (extract from template_to_t1w) --> T1w --> native (BOLD)
        if base_std_space != "MNI152NLin6Asym":
            mni_to_std_xform = str(
                get_template(
                    template=base_std_space,
                    mode="image",
                    suffix="xfm",
                    extension=".h5",
                    **{"from": "MNI152NLin6Asym"},
                ),
            )
            transform_list = [t1w_to_native, template_to_t1w, mni_to_std_xform]
        else:
            transform_list = [t1w_to_native, template_to_t1w]

    else:
        file_base = os.path.basename(bold_file)
        raise ValueError(f"Space '{bold_space}' in {file_base} not supported.")

    return transform_list


def fwhm2sigma(fwhm):
    """Convert full width at half maximum to sigma.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum.

    Returns
    -------
    float
        Sigma.
    """
    return fwhm / np.sqrt(8 * np.log(2))


def zscore_nifti(img, outputname, mask=None):
    """Normalize (z-score) a NIFTI image.

    Image and mask must be in the same space.
    TODO: Use Nilearn for masking.

    Parameters
    ----------
    img : str
        Path to the NIFTI image to z-score.
    outputname : str
        Output filename.
    mask : str or None, optional
        Path to binary mask file. Default is None.

    Returns
    -------
    outputname : str
        Output filename. Same as the ``outputname`` parameter.
    """
    img = nb.load(img)

    if mask:
        # z-score the data
        maskdata = nb.load(mask).get_fdata()
        imgdata = img.get_fdata()
        meandata = imgdata[maskdata > 0].mean()
        stddata = imgdata[maskdata > 0].std()
        zscore_fdata = (imgdata - meandata) / stddata
        # values where the mask is less than 1 are set to 0
        zscore_fdata[maskdata < 1] = 0
    else:
        # z-score the data
        imgdata = img.get_fdata()
        meandata = imgdata[np.abs(imgdata) > 0].mean()
        stddata = imgdata[np.abs(imgdata) > 0].std()
        zscore_fdata = (imgdata - meandata) / stddata

    # turn image to nifti and write it out
    dataout = nb.Nifti1Image(zscore_fdata, affine=img.affine, header=img.header)
    dataout.to_filename(outputname)
    return outputname


def butter_bandpass(data, fs, lowpass, highpass, order=2):
    """Apply a Butterworth bandpass filter to data.

    Parameters
    ----------
    data : numpy.ndarray
        Voxels/vertices by timepoints dimension.
    fs : float
        Sampling frequency. 1/TR(s).
    lowpass : float
        frequency
    highpass : float
        frequency
    order : int
        The order of the filter. This will be divided by 2 when calling scipy.signal.butter.

    Returns
    -------
    filtered_data : numpy.ndarray
        The filtered data.
    """
    nyq = 0.5 * fs  # nyquist frequency

    # normalize the cutoffs
    lowcut = np.float(highpass) / nyq
    highcut = np.float(lowpass) / nyq

    b, a = butter(order / 2, [lowcut, highcut], btype="band")  # get filter coeff

    filtered_data = np.zeros(data.shape)  # create something to populate filtered values with

    # apply the filter, loop through columns of regressors
    for ii in range(filtered_data.shape[0]):
        filtered_data[ii, :] = filtfilt(
            b, a, data[ii, :], padtype="odd", padlen=3 * (max(len(b), len(a)) - 1)
        )

    return filtered_data
