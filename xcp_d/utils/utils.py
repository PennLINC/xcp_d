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
from templateflow.api import get as get_template

from xcp_d.utils.doc import fill_doc


def _t12native(fname):
    """Select T1w-to-scanner transform associated with a given BOLD file.

    TODO: Update names and refactor

    Parameters
    ----------
    fname : str
        The BOLD file from which to identify the transform.

    Returns
    -------
    t12ref : str
        Path to the T1w-to-scanner transform.

    Notes
    -----
    Only used in get_segfile, which should be removed ASAP.
    """
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    fileup = filename.split("desc-preproc_bold.nii.gz")[0].split("space-")[0]
    t12ref = directx + "/" + fileup + "from-T1w_to-scanner_mode-image_xfm.txt"
    return t12ref


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
        mni_to_t1w=mni_to_t1,
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


def get_bold2std_and_t1w_xforms(bold_file, mni_to_t1w, t1w_to_native):
    """Find transform files in reverse order to transform BOLD to MNI152NLin2009cAsym/T1w space.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    mni_to_t1w : str
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
    import os
    import re

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    # Extract the space of the BOLD file
    file_base = os.path.basename(bold_file)
    bold_space = re.findall("space-([a-zA-Z0-9]+)", file_base)
    if not bold_space:
        bold_space = "native"
    else:
        bold_space = bold_space[0]

    if bold_space in ("native", "T1w"):
        base_std_space = re.findall("from-([a-zA-Z0-9]+)", mni_to_t1w)[0]
    elif f"from-{bold_space}" not in mni_to_t1w:
        raise ValueError(f"Transform does not match BOLD space: {bold_space} != {mni_to_t1w}")

    # Pull out the correct transforms based on bold_file name and string them together.
    xforms_to_T1w = [mni_to_t1w]  # used for all spaces except T1w and native
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
        # T1w --> ?? (extract from mni_to_t1w) --> MNI152NLin2009cAsym
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
            xforms_to_MNI = [std_to_mni_xform, mni_to_t1w]
            xforms_to_MNI_invert = [False, True]
        else:
            xforms_to_MNI = [mni_to_t1w]
            xforms_to_MNI_invert = [True]

        xforms_to_T1w = ["identity"]
        xforms_to_T1w_invert = [False]

    elif bold_space == "native":
        # native (BOLD) --> T1w --> ?? (extract from mni_to_t1w) --> MNI152NLin2009cAsym
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
            xforms_to_MNI = [std_to_mni_xform, mni_to_t1w, t1w_to_native]
            xforms_to_MNI_invert = [False, True, True]
        else:
            xforms_to_MNI = [mni_to_t1w, t1w_to_native]
            xforms_to_MNI_invert = [True, True]

        xforms_to_T1w = [t1w_to_native]
        xforms_to_T1w_invert = [True]

    else:
        raise ValueError(f"Space '{bold_space}' in {bold_file} not supported.")

    return xforms_to_MNI, xforms_to_MNI_invert, xforms_to_T1w, xforms_to_T1w_invert


def get_std2bold_xforms(bold_file, mni_to_t1w, t1w_to_native):
    """Obtain transforms to warp atlases from MNI152NLin6Asym to the same space as the BOLD.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    mni_to_t1w : str
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
    import re

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    # Extract the space of the BOLD file
    file_base = os.path.basename(bold_file)
    bold_space = re.findall("space-([a-zA-Z0-9]+)", file_base)
    if not bold_space:
        bold_space = "native"
    else:
        bold_space = bold_space[0]

    # Check that the MNI-to-T1w xform is from the right space
    if bold_space in ("native", "T1w"):
        base_std_space = re.findall("from-([a-zA-Z0-9]+)", mni_to_t1w)[0]
    elif f"from-{bold_space}" not in mni_to_t1w:
        raise ValueError(f"Transform does not match BOLD space: {bold_space} != {mni_to_t1w}")

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
        # NLin6 --> ?? (extract from mni_to_t1w) --> T1w (BOLD)
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
            transform_list = [mni_to_t1w, mni_to_std_xform]
        else:
            transform_list = [mni_to_t1w]

    elif bold_space == "native":
        # The BOLD data are in native space
        # NLin6 --> ?? (extract from mni_to_t1w) --> T1w --> native (BOLD)
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
            transform_list = [t1w_to_native, mni_to_t1w, mni_to_std_xform]
        else:
            transform_list = [t1w_to_native, mni_to_t1w]

    else:
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


@fill_doc
def stringforparams(params):
    """Infer nuisance regression description from parameter set.

    Parameters
    ----------
    %(params)s

    Returns
    -------
    bsignal : str
        String describing the parameters used for nuisance regression.
    """
    if params == "custom":
        bsignal = "A custom set of regressors was used, with no other regressors from XCP-D"

    elif params == "24P":
        bsignal = (
            "In total, 24 nuisance regressors were selected  from the nuisance "
            "confound matrices of fMRIPrep output. These nuisance regressors included "
            "six motion parameters with their temporal derivatives, "
            "and their quadratic expansion of those six motion parameters and their "
            "temporal derivatives"
        )

    elif params == "27P":
        bsignal = (
            "In total, 27 nuisance regressors were selected from the nuisance "
            "confound matrices of fMRIPrep output. These nuisance regressors included "
            "six motion parameters with their temporal derivatives, "
            "the quadratic expansion of those six motion parameters and "
            "their derivatives, the global signal, the mean white matter "
            "signal, and the mean CSF signal"
        )

    elif params == "36P":
        bsignal = (
            "In total, 36 nuisance regressors were selected from the nuisance "
            "confound matrices of fMRIPrep output. These nuisance regressors included "
            "six motion parameters, global signal, the mean white matter, "
            "the mean CSF signal with their temporal derivatives, "
            "and the quadratic expansion of six motion parameters, tissues signals and "
            "their temporal derivatives"
        )

    elif params == "aroma":
        bsignal = (
            "All the clean aroma components with the mean white matter "
            "signal, and the mean CSF signal were selected as nuisance regressors"
        )

    elif params == "acompcor":
        bsignal = (
            "The top 5 principal aCompCor components from WM and CSF compartments "
            "were selected as "
            "nuisance regressors. Additionally, the six motion parameters and their temporal "
            "derivatives were added as confounds."
        )

    elif params == "aroma_gsr":
        bsignal = (
            "All the clean aroma components with the mean white matter "
            "signal, and the mean CSF signal, and mean global signal were "
            "selected as nuisance regressors"
        )

    elif params == "acompcor_gsr":
        bsignal = (
            "The top 5 principal aCompCor components from WM and CSF "
            "compartments were selected as "
            "nuisance regressors. Additionally, the six motion parameters and their temporal "
            "derivatives were added as confounds. The average global signal was also added as a "
            "regressor."
        )

    else:
        raise ValueError(f"Parameter string not understood: {params}")

    return bsignal


def get_customfile(custom_confounds_folder, fmriprep_confounds_file):
    """Identify a custom confounds file.

    Parameters
    ----------
    custom_confounds_folder : str or None
        The path to the custom confounds file.
    fmriprep_confounds_file : str
        Path to the confounds file from the preprocessing pipeline.
        We expect the custom confounds file to have the same name.

    Returns
    -------
    custom_confounds_file : str or None
        The appropriate custom confounds file.
    """
    if custom_confounds_folder is None:
        return None

    if not os.path.isdir(custom_confounds_folder):
        raise ValueError(f"Custom confounds location does not exist: {custom_confounds_folder}")

    custom_confounds_filename = os.path.basename(fmriprep_confounds_file)
    custom_confounds_file = os.path.abspath(
        os.path.join(
            custom_confounds_folder,
            custom_confounds_filename,
        )
    )

    if not os.path.isfile(custom_confounds_file):
        raise FileNotFoundError(f"Custom confounds file not found: {custom_confounds_file}")

    return custom_confounds_file


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


def extract_timeseries(img, atlas_file, labels_file, TR):
    """Use Nilearn NiftiLabelsMasker to extract atlas time series from BOLD data.

    Parameters
    ----------
    img : str or niimg
    atlas_file : str
    labels_file : str
    TR : float

    Returns
    -------
    clean_time_series : :obj:`pandas.DataFrame` of shape (T, R)
        T = time. R = region.

    Notes
    -----
    Currently doesn't leverage masker's denoising capabilities.
    """
    import pandas as pd
    from nilearn import maskers

    # TODO: Standardize the atlas metadata format.
    labels = pd.read_table(labels_file)["labels"]

    masker = maskers.NiftiLabelsMasker(
        labels_img=atlas_file,
        labels=labels,
        runs=None,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=False,
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=TR,
        target_affine=None,
        target_shape=None,
        mask_strategy=None,
        mask_args=None,
        dtype=None,
        memory_level=1,
        verbose=0,
        reports=False,
    )
    clean_time_series = masker.fit_transform(X=img, confounds=None, sample_mask=None)
    clean_time_series = pd.DataFrame(data=clean_time_series, columns=labels)
    return clean_time_series


def denoise_nifti_with_nilearn(
    bold_file,
    mask_file,
    confounds_file,
    censoring_file,
    low_pass,
    high_pass,
    TR,
):
    """Denoise fMRI data with Nilearn.

    Parameters
    ----------
    bold_file : str or niimg
    mask_file : str
    confounds : pandas.DataFrame
    low_pass : float
    high_pass : float
    TR : float
    tmask : str
    """
    import os

    from nilearn import maskers

    from xcp_d.utils.utils import _denoise_with_nilearn

    out_file = os.path.abspath("desc-denoised_bold.nii.gz")

    # Use a NiftiMasker instead of apply_mask to retain TR in the image header.
    # Note that this doesn't use any of the masker's denoising capabilities.
    masker = maskers.NiftiMasker(
        mask_img=mask_file,
        runs=None,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=False,  # non-default
        detrend=False,
        high_variance_confounds=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
    )
    raw_data = masker.fit_transform(bold_file)

    clean_data = _denoise_with_nilearn(
        raw_data=raw_data,
        confounds_file=confounds_file,
        censoring_file=censoring_file,
        low_pass=low_pass,
        high_pass=high_pass,
        TR=TR,
    )

    clean_img = masker.inverse_transform(clean_data)

    clean_img.to_filename(out_file)
    return out_file


def denoise_cifti_with_nilearn(
    bold_file,
    confounds_file,
    censoring_file,
    low_pass,
    high_pass,
    TR,
):
    """Denoise a CIFTI file with Nilearn.

    The CIFTI file must be read into an array before Nilearn can be called.
    """
    import os

    from xcp_d.utils.utils import _denoise_with_nilearn
    from xcp_d.utils.write_save import read_ndata, write_ndata

    out_file = os.path.abspath("desc-denoised_bold.dtseries.nii")

    raw_data = read_ndata(bold_file)

    # Transpose from SxT (xcpd order) to TxS (nilearn order)
    raw_data = raw_data.T

    clean_data = _denoise_with_nilearn(
        raw_data=raw_data,
        confounds_file=confounds_file,
        censoring_file=censoring_file,
        low_pass=low_pass,
        high_pass=high_pass,
        TR=TR,
    )

    # Transpose from TxS (nilearn order) to SxT (xcpd order)
    clean_data = clean_data.T

    write_ndata(clean_data, template=bold_file, filename=out_file, TR=TR)

    return out_file


def _denoise_with_nilearn(
    raw_data,
    confounds_file,
    censoring_file,
    low_pass,
    high_pass,
    TR,
):
    """Denoise an array with Nilearn.

    This step does the following.

    Linearly detrend, but don't mean-center, the BOLD data.
    Regress out confounds from BOLD data.
    Use list of outliers to censor BOLD data during regression.
    Temporally filter BOLD data.
    """
    import pandas as pd
    from nilearn import signal

    confounds_df = pd.read_table(confounds_file)

    sample_mask_bool = pd.read_table(censoring_file)["framewise_displacement"].values.astype(bool)
    sample_mask = np.where(~sample_mask_bool)[0]

    clean_data = signal.clean(
        signals=raw_data,
        detrend=True,
        standardize=False,
        sample_mask=sample_mask,
        confounds=confounds_df,
        standardize_confounds=True,
        filter="butterworth",
        low_pass=low_pass,
        high_pass=high_pass,
        t_r=TR,
        ensure_finite=True,
    )

    return clean_data


def consolidate_confounds(
    fmriprep_confounds_file,
    namesource,
    params,
    custom_confounds_file=None,
):
    """Combine confounds files into a single tsv.

    Parameters
    ----------
    fmriprep_confounds_file : file
        file to fmriprep confounds tsv
    namesource : file
        file to extract entities from
    custom_confounds_file : file
        file to custom confounds tsv
    params : string
        confound parameters to load

    Returns
    -------
    out_file : file
        file to combined tsv
    """
    import os

    from xcp_d.utils.confounds import load_confound_matrix

    confounds_df = load_confound_matrix(
        original_file=namesource,
        custom_confounds=custom_confounds_file,
        confound_tsv=fmriprep_confounds_file,
        params=params,
    )

    out_file = os.path.abspath("confounds.tsv")
    confounds_df.to_csv(out_file, sep="\t", index=False)

    return out_file
