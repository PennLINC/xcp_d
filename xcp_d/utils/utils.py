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
    """
    directx = os.path.dirname(fname)
    filename = os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'
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
    """
    # get transform files
    dd = Path(os.path.dirname(bold_file))
    anatdir = str(dd.parent) + '/anat'

    if Path(anatdir).is_dir():
        mni_to_t1 = glob.glob(
            anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]
    else:
        anatdir = str(dd.parent.parent) + '/anat'
        mni_to_t1 = glob.glob(
            anatdir + '/*MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5')[0]

    transformfilex = get_transformfile(
        bold_file=bold_file,
        mni_to_t1w=mni_to_t1,
        t1w_to_native=_t12native(bold_file),
    )

    boldref = bold_file.split('desc-preproc_bold.nii.gz')[0] + 'boldref.nii.gz'

    segfile = tempfile.mkdtemp() + 'segfile.nii.gz'
    carpet = str(
        get_template(
            'MNI152NLin2009cAsym',
            resolution=1,
            desc='carpet',
            suffix='dseg',
            extension=['.nii', '.nii.gz'],
        ),
    )

    # seg_data file to bold space
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = carpet
    at.inputs.reference_image = boldref
    at.inputs.output_image = segfile
    at.inputs.interpolation = 'MultiLabel'
    at.inputs.transforms = transformfilex
    os.system(at.cmdline)

    return segfile


def get_transformfilex(bold_file, mni_to_t1w, t1w_to_native):
    """Obtain the correct transform files in reverse order to transform to MNI space/T1W space.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    mni_to_t1w : str
        The MNI-to-T1w transform file.
    t1w_to_native : str
        The T1w-to-native space transform file.

    Returns
    -------
    transformfileMNI : list of str
        A list of paths to transform files for warping to MNI space.
    transformfileT1W : list of str
        A list of paths to transform files for warping to T1w space.
    """
    import glob
    import os

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    # get file basename, anatdir and list all transforms in anatdir
    file_base = os.path.basename(str(bold_file))
    MNI6 = str(
        get_template(template='MNI152NLin2009cAsym',
                     mode='image',
                     suffix='xfm',
                     extension='.h5'))

    # get default template MNI152NLin2009cAsym for fmriprep
    if 'MNI152NLin2009cAsym' in os.path.basename(mni_to_t1w):
        template = 'MNI152NLin2009cAsym'

    # for infants
    elif 'MNIInfant' in os.path.basename(mni_to_t1w):
        template = 'MNIInfant'

    # in case fMRIPrep outputs are generated in MNI6, as
    # done in case of AROMA outputs
    elif 'MNI152NLin6Asym' in os.path.basename(mni_to_t1w):
        template = 'MNI152NLin6Asym'

    elif 'MNI152NLin6Asym' in os.path.basename(mni_to_t1w):
        template = 'MNI152NLin6Asym'

    # Pull out the correct transforms based on bold_file name
    # and string them together.
    if 'space-MNI152NLin2009cAsym' in file_base:
        transformfileMNI = str(MNI6)
        transformfileT1W = str(mni_to_t1w)

    elif 'space-MNI152NLin6Asym' in file_base:
        transformfileMNI = [MNI6]
        transformfileT1W = [str(MNI6), str(mni_to_t1w)]

    elif 'space-PNC' in file_base:
        mnisf = mni_to_t1w.split('from-')[0]
        pnc_to_t1w = mnisf + 'from-PNC*_to-T1w_mode-image_xfm.h5'
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template
                               + '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(pnc_to_t1w), str(t1w_to_mni)]
        transformfileT1W = str(pnc_to_t1w)

    elif 'space-NKI' in file_base:
        mnisf = mni_to_t1w.split('from-')[0]
        nki_to_t1w = mnisf + 'from-NKI_to-T1w_mode-image_xfm.h5'
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template
                               + '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(nki_to_t1w), str(t1w_to_mni)]
        transformfileT1W = str(nki_to_t1w)

    elif 'space-OASIS' in file_base:
        mnisf = mni_to_t1w.split('from')[0]
        oasis_to_t1w = mnisf + 'from-OASIS30ANTs_to-T1w_mode-image_xfm.h5'
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template
                               + '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(oasis_to_t1w), str(t1w_to_mni)]
        transformfileT1W = [str(oasis_to_t1w)]

    elif 'space-MNI152NLin6Sym' in file_base:
        mnisf = mni_to_t1w.split('from-')[0]
        mni6c_to_t1w = mnisf + 'from-MNI152NLin6Sym_to-T1w_mode-image_xfm.h5'
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template
                               + '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(mni6c_to_t1w), str(t1w_to_mni)]
        transformfileT1W = [str(mni6c_to_t1w)]

    elif 'space-MNIInfant' in file_base:
        transformfileMNI = str(
            pkgrf('xcp_d', 'data/transform/infant_to_2009_Composite.h5'))
        transformfileT1W = str(mni_to_t1w)

    elif 'space-MNIPediatricAsym' in file_base:
        mnisf = mni_to_t1w.split('from-')[0]
        mni6c_to_t1w = glob.glob(
            mnisf + 'from-MNIPediatricAsym*_to-T1w_mode-image_xfm.h5')[0]
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template
                               + '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(mni6c_to_t1w), str(t1w_to_mni)]
        transformfileT1W = [str(mni6c_to_t1w)]

    elif 'space-T1w' in file_base:
        mnisf = mni_to_t1w.split('from')[0]
        transformfileMNI = [str(t1w_to_mni)]
        transformfileT1W = [
            str(pkgrf('xcp_d', 'data/transform/oneratiotransform.txt'))
        ]

    elif 'space-' not in file_base:
        t1wf = t1w_to_native.split('from-T1w_to-scanner_mode-image_xfm.txt')[0]
        native_to_t1w = t1wf + 'from-T1w_to-scanner_mode-image_xfm.txt'
        mnisf = mni_to_t1w.split('from')[0]
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template
                               + '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(t1w_to_mni), str(native_to_t1w)]
        transformfileT1W = [str(native_to_t1w)]
    else:
        print('space not supported')

    return transformfileMNI, transformfileT1W


def get_maskfiles(bold_file, mni_to_t1w):
    """Identify BOLD- and T1-resolution brain masks from files.

    Parameters
    ----------
    bold_file : str
        Path to the preprocessed BOLD file.
    mni_to_t1w : str
        Path to the MNI-to-T1w transform file.

    Returns
    -------
    boldmask : str
        The path to the BOLD-resolution mask.
    t1mask : str
        The path to the T1-resolution mask.
    """
    boldmask = bold_file.split(
        'desc-preproc_bold.nii.gz')[0] + 'desc-brain_mask.nii.gz'
    t1mask = mni_to_t1w.split('from-')[0] + 'desc-brain_mask.nii.gz'
    return boldmask, t1mask


def get_transformfile(bold_file, mni_to_t1w, t1w_to_native):
    """Obtain transforms to warp atlases from MNI space to the same space as the bold file.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    mni_to_t1w : str
        The MNI-to-T1w transform file.
    t1w_to_native : str
        The T1w-to-native space transform file.

    Returns
    -------
    transform_list : list of str
        A list of paths to transform files.
    """
    import glob
    import os

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    file_base = os.path.basename(str(bold_file))  # file base is the bold_name

    # get the correct template via templateflow/ pkgrf
    fMNI6 = str(  # template
        get_template(template='MNI152NLin2009cAsym',
                     mode='image',
                     suffix='xfm',
                     extension='.h5'))
    FSL2MNI9 = pkgrf('xcp_d', 'data/transform/FSL2MNI9Composite.h5')

    # Transform to MNI9
    transform_list = []
    if 'space-MNI152NLin6Asym' in file_base:
        transform_list = [str(fMNI6)]
    elif 'space-MNI152NLin2009cAsym' in file_base:
        transform_list = [str(FSL2MNI9)]
    elif 'space-PNC' in file_base:
        #  get the PNC transforms
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_pnc = mnisf + 'from-T1w_to-PNC_mode-image_xfm.h5'
        #  get all the transform files together
        transform_list = [str(t1w_to_pnc), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-NKI' in file_base:
        #  get the NKI transforms
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_nki = mnisf + 'from-T1w_to-NKI_mode-image_xfm.h5'
        #  get all the transforms together
        transform_list = [str(t1w_to_nki), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-OASIS30ANTs' in file_base:
        #  get the relevant transform, put all transforms together
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_oasis = mnisf + 'from-T1w_to-OASIS30ANTs_mode-image_xfm.h5'
        transform_list = [str(t1w_to_oasis), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-MNI152NLin6Sym' in file_base:
        #  get the relevant transform, put all transforms together
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_mni6c = mnisf + 'from-T1w_to-MNI152NLin6Sym_mode-image_xfm.h5'
        transform_list = [str(t1w_to_mni6c), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-MNIInfant' in file_base:
        #  get the relevant transform, put all transforms together
        infant2mni9 = pkgrf('xcp_d',
                            'data/transform/infant_to_2009_Composite.h5')
        transform_list = [str(infant2mni9), str(FSL2MNI9)]
    elif 'space-MNIPediatricAsym' in file_base:
        #  get the relevant transform, put all transforms together
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_mni6cx = glob.glob(
            mnisf + 'from-T1w_to-MNIPediatricAsym*_mode-image_xfm.h5')[0]
        transform_list = [str(t1w_to_mni6cx), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-T1w' in file_base:
        #  put all transforms together
        transform_list = [str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-' not in file_base:
        #  put all transforms together
        transform_list = [str(t1w_to_native), str(mni_to_t1w), str(FSL2MNI9)]
    else:
        print('space not supported')

    if not transform_list:
        raise Exception(f"Transforms not found for {file_base}")

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
    if params == 'custom':
        bsignal = "A custom set of regressors was used, with no other regressors from XCP-D"
    if params == '24P':
        bsignal = "In total, 24 nuisance regressors were selected  from the nuisance \
        confound matrices of fMRIPrep output. These nuisance regressors included \
        six motion parameters with their temporal derivatives, \
        and their quadratic expansion of those six motion parameters and their \
        temporal derivatives"

    if params == '27P':
        bsignal = "In total, 27 nuisance regressors were selected from the nuisance \
        confound matrices of fMRIPrep output. These nuisance regressors included \
        six motion parameters with their temporal derivatives, \
        the quadratic expansion of those six motion parameters and  \
        their derivatives, the global signal, the mean white matter  \
        signal, and the mean CSF signal"

    if params == '36P':
        bsignal = "In total, 36 nuisance regressors were selected from the nuisance \
        confound matrices of fMRIPrep output. These nuisance regressors included \
        six motion parameters, global signal, the mean white matter,  \
        the mean CSF signal  with their temporal derivatives, \
        and the quadratic expansion of six motion parameters, tissues signals and  \
        their temporal derivatives"

    if params == 'aroma':
        bsignal = "All the clean aroma components with the mean white matter  \
        signal, and the mean CSF signal were selected as nuisance regressors"

    if params == 'acompcor':
        bsignal = "The top 5 principal aCompCor components from WM and CSF compartments \
        were selected as \
        nuisance regressors. Additionally, the six motion parameters and their temporal \
        derivatives were added as confounds."

    if params == 'aroma_gsr':
        bsignal = "All the clean aroma components with the mean white matter  \
        signal, and the mean CSF signal, and mean global signal were \
        selected as nuisance regressors"

    if params == 'acompcor_gsr':
        bsignal = "The top 5 principal aCompCor components from WM and CSF \
        compartments were selected as \
        nuisance regressors. Additionally, the six motion parameters and their temporal \
        derivatives were added as confounds. The average global signal was also added as a \
        regressor."

    return bsignal


def get_customfile(custom_confounds, bold_file):
    """Identify a custom confounds file.

    Parameters
    ----------
    custom_confounds : str
        The path to the custom confounds file.
        This shouldn't include the actual filename.
    bold_file : str
        Path to the associated preprocessed BOLD file.

    Returns
    -------
    custom_file : str
        The custom confounds file associated with the BOLD file.
    """
    if custom_confounds is not None:
        confounds_timeseries = bold_file.replace(
            "_space-" + bold_file.split("space-")[1],
            "_desc-confounds_timeseries.tsv")
        file_base = os.path.basename(
            confounds_timeseries.split('-confounds_timeseries.tsv')[0])
        custom_file = os.path.abspath(
            str(custom_confounds) + '/' + file_base + '-custom_timeseries.tsv')
    else:
        custom_file = None
    return custom_file


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
    dataout = nb.Nifti1Image(zscore_fdata,
                             affine=img.affine,
                             header=img.header)
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
    custom_confounds_file,
    namesource,
    params,
):
    """Combine confounds files into a single TSV."""
    import os

    from xcp_d.utils.confounds import load_confound_matrix

    out_file = os.path.abspath("confounds.tsv")

    # It looks like nipype is passing this along as "None".
    if custom_confounds_file == "None":
        custom_confounds_file = None

    if fmriprep_confounds_file and custom_confounds_file:
        confounds_df = load_confound_matrix(
            original_file=namesource,
            custom_confounds=custom_confounds_file,
            confound_tsv=fmriprep_confounds_file,
            params=params,
        )
    else:  # No custom confounds
        confounds_df = load_confound_matrix(
            original_file=namesource,
            confound_tsv=fmriprep_confounds_file,
            params=params,
        )

    confounds_df.to_csv(out_file, sep="\t", index=False)

    return out_file
