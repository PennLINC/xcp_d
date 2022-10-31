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
from scipy.signal import butter, detrend, filtfilt
from sklearn.linear_model import LinearRegression
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

    transformfilex = get_std2bold_xforms(
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


def get_xforms_to_std_and_t1w(bold_file, mni_to_t1w, t1w_to_native):
    """Obtain the correct transform files in reverse order to transform to MNI space/T1W space.

    Since ANTSApplyTransforms takes in the transform files as a stack,
    these are applied in the reverse order of which they are specified.

    Parameters
    ----------
    bold_file : str
        The preprocessed BOLD file.
    mni_to_t1w : str
        The MNI-to-T1w transform file.
        The ``from`` field should be the same space as the BOLD file is in.
    t1w_to_native : str
        The T1w-to-native space transform file.

    Returns
    -------
    xforms_to_MNI : list of str
        A list of paths to transform files for warping to MNI152NLin2009cAsym space.
    xforms_to_T1w : list of str
        A list of paths to transform files for warping to T1w space.

    Notes
    -----
    Only used for QCReport in init_boldpostprocess_wf.
    QCReport wants MNI-space data in MNI152NLin2009cAsym.
    """
    import os
    import re

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    # get file basename, anatdir and list all transforms in anatdir
    file_base = os.path.basename(bold_file)
    MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
        get_template(
            template='MNI152NLin2009cAsym',
            mode='image',
            suffix='xfm',
            extension='.h5',
            **{"from": "MNI152NLin6Asym"},
        ),
    )

    # Pull out the correct transforms based on bold_file name and string them together.
    if 'space-MNI152NLin2009cAsym' in file_base:
        # All good.
        xforms_to_MNI = ["identity"]
        xforms_to_T1w = [mni_to_t1w]

    elif 'space-MNI152NLin6Asym' in file_base:
        # NLin6 --> NLin2009c
        xforms_to_MNI = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]
        xforms_to_T1w = [MNI152NLin6Asym_to_MNI152NLin2009cAsym, mni_to_t1w]

    elif 'space-MNIInfant' in file_base:
        # MNIInfant --> NLin2009c
        xforms_to_MNI = [pkgrf('xcp_d', 'data/transform/infant_to_2009_Composite.h5')]
        xforms_to_T1w = [mni_to_t1w]
    else:
        found_space = re.findall("_space-([a-zA-Z0-9]+)_", file_base)
        if not found_space:
            space = "native"
        else:
            space = found_space[0]

        raise ValueError(f"Space {space} in {file_base} not supported.")

    return xforms_to_MNI, xforms_to_T1w


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
    t1w_to_native : str
        The T1w-to-native space transform file.

    Returns
    -------
    transform_list : list of str
        A list of paths to transform files.

    Notes
    -----
    Used by get_segfile (to be removed), to resample dseg in init_boldpostprocess_wf for QCReport,
    to warp atlases to the same space as the BOLD data in init_nifti_functional_connectivity_wf,
    and to resample dseg to BOLD space for the executive summary plots.
    """
    import os
    import re

    from pkg_resources import resource_filename as pkgrf
    from templateflow.api import get as get_template

    file_base = os.path.basename(bold_file)  # file base is the bold_name

    # Load useful inter-template transforms from templateflow
    MNI152NLin6Asym_to_MNI152NLin2009cAsym = str(
        get_template(
            template='MNI152NLin2009cAsym',
            mode='image',
            suffix='xfm',
            extension='.h5',
            **{"from": "MNI152NLin6Asym"},
        ),
    )

    # Find the appropriate transform(s)
    if 'space-MNI152NLin6Asym' in file_base:
        # NLin6 --> NLin6 (identity)
        transform_list = ["identity"]
    elif 'space-MNI152NLin2009cAsym' in file_base:
        # NLin6 --> NLin2009c
        transform_list = [MNI152NLin6Asym_to_MNI152NLin2009cAsym]
    elif 'space-MNIInfant' in file_base:
        # NLin6 --> NLin2009c --> MNIInfant
        MNI152Infant_to_MNI152NLin2009cAsym = pkgrf(
            'xcp_d',
            'data/transform/infant_to_2009_Composite.h5',
        )
        transform_list = [
            MNI152Infant_to_MNI152NLin2009cAsym,
            MNI152NLin6Asym_to_MNI152NLin2009cAsym,
        ]
    elif 'space-T1w' in file_base:
        # NLin6 --> ?? (extract from mni_to_t1w) --> T1w (BOLD)
        base_mni_space = re.findall("from-([a-zA-Z0-9]+)", mni_to_t1w)[0]
        mni_to_mni_xforms = str(
            get_template(
                template=base_mni_space,
                mode='image',
                suffix='xfm',
                extension='.h5',
                **{"from": "MNI152NLin6Asym"},
            ),
        )

        transform_list = [mni_to_t1w, mni_to_mni_xforms]
    elif 'space-' not in file_base:
        # The BOLD data are in native space
        # NLin6 --> ?? (extract from mni_to_t1w) --> T1w --> native (BOLD)
        base_mni_space = re.findall("from-([a-zA-Z0-9]+)", mni_to_t1w)[0]
        mni_to_mni_xforms = str(
            get_template(
                template=base_mni_space,
                mode='image',
                suffix='xfm',
                extension='.h5',
                **{"from": "MNI152NLin6Asym"},
            ),
        )

        transform_list = [t1w_to_native, mni_to_t1w, mni_to_mni_xforms]
    else:
        found_space = re.findall("_space-([a-zA-Z0-9]+)_", file_base)
        space = found_space[0]
        raise ValueError(f"Space {space} in {file_base} not supported.")

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

    b, a = butter(order / 2, [lowcut, highcut], btype='band')  # get filter coeff

    filtered_data = np.zeros(data.shape)  # create something to populate filtered values with

    # apply the filter, loop through columns of regressors
    for ii in range(filtered_data.shape[0]):
        filtered_data[ii, :] = filtfilt(b, a, data[ii, :], padtype='odd',
                                        padlen=3 * (max(len(b), len(a)) - 1))

    return filtered_data


def linear_regression(data, confound):
    """Perform linear regression with sklearn's LinearRegression.

    Parameters
    ----------
    data : numpy.ndarray
        vertices by timepoints for bold file
    confound : numpy.ndarray
       nuisance regressors - vertices by timepoints for confounds matrix

    Returns
    -------
    numpy.ndarray
        residual matrix after regression
    """
    regression = LinearRegression(n_jobs=1)
    regression.fit(confound.T, data.T)
    y_predicted = regression.predict(confound.T)

    return data - y_predicted.T


def demean_detrend_data(data):
    """Mean-center and remove linear trends over time from data.

    Parameters
    ----------
    data : numpy.ndarray
        vertices by timepoints for bold file

    Returns
    -------
    detrended : numpy.ndarray
        demeaned and detrended data
    """
    demeaned = detrend(data, axis=- 1, type='constant', bp=0,
                       overwrite_data=False)  # Demean data using "constant" detrend,
    # which subtracts mean
    detrended = detrend(demeaned, axis=- 1, type='linear', bp=0,
                        overwrite_data=False)  # Detrend data using linear method

    return detrended  # Subtract these predicted values from the demeaned data
