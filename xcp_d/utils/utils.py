#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import glob as glob
from templateflow.api import get as get_template
from pkg_resources import resource_filename as pkgrf


def get_transformfilex(bold_file, mni_to_t1w, t1w_to_native):
    """
    Obtain the correct transform files in reverse order to transform
    the atlases from MNI space to the same space as the bold file.
    First, we find the correct relevant transforms (i.e: t1w to native),
    then find the mni_to_t1w file.

    Lastly, we specify the FSL2MNI composite file.

    Since ANTSApplyTransforms takes in the transform files as a stack, these are
    applied in the reverse order of which they are specified.
    """

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
    elif 'MNI152NLin6ASym' in os.path.basename(mni_to_t1w):
        template = 'MNI152NLin6ASym'

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
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template +
                               '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(pnc_to_t1w), str(t1w_to_mni)]
        transformfileT1W = str(pnc_to_t1w)

    elif 'space-NKI' in file_base:
        mnisf = mni_to_t1w.split('from-')[0]
        nki_to_t1w = mnisf + 'from-NKI_to-T1w_mode-image_xfm.h5'
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template +
                               '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(nki_to_t1w), str(t1w_to_mni)]
        transformfileT1W = str(nki_to_t1w)

    elif 'space-OASIS' in file_base:
        mnisf = mni_to_t1w.split('from')[0]
        oasis_to_t1w = mnisf + 'from-OASIS30ANTs_to-T1w_mode-image_xfm.h5'
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template +
                               '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(oasis_to_t1w), str(t1w_to_mni)]
        transformfileT1W = [str(oasis_to_t1w)]

    elif 'space-MNI152NLin6Sym' in file_base:
        mnisf = mni_to_t1w.split('from-')[0]
        mni6c_to_t1w = mnisf + 'from-MNI152NLin6Sym_to-T1w_mode-image_xfm.h5'
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template +
                               '*_mode-image_xfm.h5')[0]
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
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template +
                               '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(mni6c_to_t1w), str(t1w_to_mni)]
        transformfileT1W = [str(mni6c_to_t1w)]

    elif 'space-T1w' in file_base:
        mnisf = mni_to_t1w.split('from')[0]
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template +
                               '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(t1w_to_mni)]
        transformfileT1W = [
            str(pkgrf('xcp_d', 'data/transform/oneratiotransform.txt'))
        ]

    elif 'space-' not in file_base:
        t1wf = t1w_to_native.split('from-T1w_to-scanner_mode-image_xfm.txt')[0]
        native_to_t1w = t1wf + 'from-T1w_to-scanner_mode-image_xfm.txt'
        mnisf = mni_to_t1w.split('from')[0]
        t1w_to_mni = glob.glob(mnisf + 'from-T1w_to-' + template +
                               '*_mode-image_xfm.h5')[0]
        transformfileMNI = [str(t1w_to_mni), str(native_to_t1w)]
        transformfileT1W = [str(native_to_t1w)]
    else:
        print('space not supported')

    return transformfileMNI, transformfileT1W


def get_maskfiles(bold_file, mni_to_t1w):

    boldmask = bold_file.split(
        'desc-preproc_bold.nii.gz')[0] + 'desc-brain_mask.nii.gz'
    t1mask = mni_to_t1w.split('from-')[0] + 'desc-brain_mask.nii.gz'
    return boldmask, t1mask


def get_transformfile(bold_file, mni_to_t1w, t1w_to_native):
    """"
    Obtain the correct transform files in reverse order to transform
    the atlases from MNI space to the same space as the bold file.
    First, we find the correct relevant transforms (i.e: t1w to native),
    then find the mni_to_t1w file.

    Lastly, we specify the FSL2MNI composite file.

    Since ANTSApplyTransforms takes in the transform files as a stack, these are
    applied in the reverse order of which they are specified.

    """

    file_base = os.path.basename(str(bold_file))  # file base is the bold_name

    # get the correct template via templateflow/ pkgrf
    fMNI6 = str(  # template
        get_template(template='MNI152NLin2009cAsym',
                     mode='image',
                     suffix='xfm',
                     extension='.h5'))
    FSL2MNI9 = pkgrf('xcp_d', 'data/transform/FSL2MNI9Composite.h5')

    # Transform to MNI9
    if 'space-MNI152NLin6Asym' in file_base:
        transformfile = [str(fMNI6)]
    elif 'space-MNI152NLin2009cAsym' in file_base:
        transformfile = str(FSL2MNI9)
    elif 'space-PNC' in file_base:
        #  get the PNC transforms
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_pnc = mnisf + 'from-T1w_to-PNC_mode-image_xfm.h5'
        #  get all the transform files together
        transformfile = [str(t1w_to_pnc), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-NKI' in file_base:
        #  get the NKI transforms
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_nki = mnisf + 'from-T1w_to-NKI_mode-image_xfm.h5'
        #  get all the transforms together
        transformfile = [str(t1w_to_nki), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-OASIS30ANTs' in file_base:
        #  get the relevant transform, put all transforms together
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_oasis = mnisf + 'from-T1w_to-OASIS30ANTs_mode-image_xfm.h5'
        transformfile = [str(t1w_to_oasis), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-MNI152NLin6Sym' in file_base:
        #  get the relevant transform, put all transforms together
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_mni6c = mnisf + 'from-T1w_to-MNI152NLin6Sym_mode-image_xfm.h5'
        transformfile = [str(t1w_to_mni6c), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-MNIInfant' in file_base:
        #  get the relevant transform, put all transforms together
        infant2mni9 = pkgrf('xcp_d',
                            'data/transform/infant_to_2009_Composite.h5')
        transformfile = [str(infant2mni9), str(FSL2MNI9)]
    elif 'space-MNIPediatricAsym' in file_base:
        #  get the relevant transform, put all transforms together
        mnisf = mni_to_t1w.split('from-')[0]
        t1w_to_mni6cx = glob.glob(
            mnisf + 'from-T1w_to-MNIPediatricAsym*_mode-image_xfm.h5')[0]
        transformfile = [str(t1w_to_mni6cx), str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-T1w' in file_base:
        #  put all transforms together
        transformfile = [str(mni_to_t1w), str(FSL2MNI9)]
    elif 'space-' not in file_base:
        #  put all transforms together
        transformfile = [str(t1w_to_native), str(mni_to_t1w), str(FSL2MNI9)]
    else:
        print('space not supported')
    return transformfile


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def stringforparams(params):
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
