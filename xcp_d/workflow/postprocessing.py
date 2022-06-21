# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing the bold/cifti
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_post_process_wf

"""
import numpy as np
import sklearn
from nipype.pipeline import engine as pe
from pkg_resources import resource_filename as pkgrf
from ..utils.utils import stringforparams
from templateflow.api import get as get_template
from ..interfaces import (FilteringData, regress)
from ..interfaces import (interpolate, RemoveTR, CensorScrub)
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench import CiftiSmooth
from nipype.interfaces.fsl import Smooth

from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_post_process_wf(
        mem_gb,
        TR,
        head_radius,
        lower_bpf,
        upper_bpf,
        bpf_order,
        smoothing,
        bold_file,
        params,
        motion_filter_type,
        band_stop_max,
        band_stop_min,
        initial_volumes_to_drop,
        cifti=False,
        dummytime=0,
        fd_thresh=0,
        name="post_process_wf"):
    """
    This workflow is organizing workflows including
    selectign confound matrix, regression and filtering
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_d.workflows import init_post_process_wf
            wf = init_init_post_process_wf_wf(
                mem_gb,
                TR,
                head_radius,
                lower_bpf,
                upper_bpf,
                bpf_order,
                smoothing,
                bold_file,
                params,
                motion_filter_type,
                band_stop_max,
                band_stop_min,
                cifti=False,
                dummytime,
                fd_thresh,
                name="post_process_wf",
                )
    Parameters
    ----------
    TR: float
         Repetition time in second
    bold_file: str
        bold file for post processing
    lower_bpf : float
        Lower band pass filter
    upper_bpf : float
        Upper band pass filter
    layout : BIDSLayout object
        BIDS dataset layout
    despike: bool
        afni depsike
    motion_filter_type: str
        respiratory motion filter type: lp or notch
    band_stop_min: float
        respiratory minimum frequency in breathe per minutes(bpm)
    band_stop_max, : float
        respiratory maximum frequency in breathe per minutes(bpm)
    layout : BIDSLayout object
        BIDS dataset layout
    omp_nthreads : int
        Maximum number of threads an individual process may use
    output_dir : str
        Directory in which to save xcp_d output
    fd_thresh
        Criterion for flagging framewise displacement outliers
    head_radius : float
        radius of the head for FD computation
    params: str
        nuissance regressors to be selected from fmriprep regressors
    smoothing: float
        smooth the derivatives output with kernel size (fwhm)
    custom_confounds: str
        path to custom nuissance regressors
    dummytime: float
        the first few seconds to be removed before postprocessing
    initial_volumes_to_drop: int
        the first volumes to be removed before postprocessing


    Inputs
    ------
    bold
       bold or cifti file
    bold_mask
       bold mask if bold is nifti
    custom_confounds
       custom regressors

    Outputs
    -------
    processed_bold
        processed or cleaned bold
    smoothed_bold
        smoothed processed bold
    tmask
        temporal mask
    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """ \

"""
    if dummytime > 0:
        nvolx = str(np.floor(dummytime / TR))
        workflow.__desc__ = workflow.__desc__ + """ \
Before nuissance regression and filtering of the data, the first {nvol} were
discarded. Furthermore, any volumes with framewise-displacement greater than
{fd_thresh} [@satterthwaite2;@power_fd_dvars;@satterthwaite_2013] were flagged
as outliers and excluded from nuissance regression.
""".format(nvol=nvolx, fd_thresh=fd_thresh)

    else:
        workflow.__desc__ = workflow.__desc__ + """ \
Before nuissance regression and filtering any volumes with
framewise-displacement greater than {fd_thresh}
[@satterthwaite2;@power_fd_dvars;@satterthwaite_2013] were  flagged as outlier
and excluded from further analyses.
""".format(fd_thresh=fd_thresh)

    workflow.__desc__ = workflow.__desc__ + """ \
The following nuissance regressors {regressors}
[@mitigating_2018;@benchmarkp;@satterthwaite_2013] were selected from nuissance
confound matrices of fMRIPrep output.  These nuissance regressors were regressed
out from the bold data with *LinearRegression* as implemented in Scikit-Learn
{sclver} [@scikit-learn].  The residual were then  band pass filtered within the
frequency band {highpass}-{lowpass} Hz.
 """.format(regressors=stringforparams(params=params),
            sclver=sklearn.__version__,
            lowpass=upper_bpf,
            highpass=lower_bpf)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold', 'bold_file', 'bold_mask', 'custom_confounds']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold', 'tmask', 'fd']),
        name='outputnode')

    inputnode.inputs.bold_file = bold_file

    filterdx = pe.Node(FilteringData(tr=TR,
                                     lowpass=upper_bpf,
                                     highpass=lower_bpf,
                                     filter_order=bpf_order),
                       name="filter_the_data",
                       mem_gb=0.25 * mem_gb)

    regressy = pe.Node(regress(tr=TR),
                       name="regress_the_data",
                       mem_gb=0.25 * mem_gb)

    censor_scrubwf = pe.Node(CensorScrub(fd_thresh=fd_thresh,
                                         motion_filter_type=motion_filter_type,
                                         low_freq=band_stop_max,
                                         high_freq=band_stop_min,
                                         head_radius=head_radius,
                                         ),
                             name="censor_scrub",
                             mem_gb=0.1 * mem_gb)

    # RF: rename to match
    interpolatewf = pe.Node(interpolate(TR=TR),
                            name="interpolation",
                            mem_gb=0.25 * mem_gb)

    if dummytime > 0:
        rm_dummytime = pe.Node(
            RemoveTR(initial_volumes_to_drop=initial_volumes_to_drop),
            name="remove_dummy_time",
            mem_gb=0.1*mem_gb)

    if dummytime > 0:
        workflow.connect([
            (inputnode, rm_dummytime, [('confound_file', 'fmriprep_confounds_file')]),
            (inputnode, rm_dummytime, [
                ('bold', 'bold_file'),
                ('bold_mask', 'mask_file')])])

        # if inputnode.inputs.custom_confounds:
        #    workflow.connect([ (inputnode, rm_dummytime, [('custom_confounds', 'custom_confounds')]),
        #                      (rm_dummytime, censor_scrubwf, [
        # ('custom_confoundsdropTR', 'custom_confounds')]),
        #                      (censor_scrubwf, regressy, [('custom_confounds_censored',
        # 'custom_confounds')]),])

        workflow.connect([
            (rm_dummytime, censor_scrubwf, [
                ('bold_file_dropped_TR', 'in_file'),
                ('fmriprep_confounds_file_dropped_TR', 'fmriprep_confounds_file')]),
            (inputnode, censor_scrubwf, [
                ('bold_file', 'bold_file'),
                ('bold_mask', 'mask_file')]),
            (censor_scrubwf, regressy, [
                ('bold_censored', 'in_file'),
                ('fmriprep_confounds_censored', 'confounds')]),
            (inputnode, regressy, [('bold_mask', 'mask')]),
            (inputnode, filterdx, [('bold_mask', 'mask')]),
            (inputnode, interpolatewf, [('bold_mask', 'mask_file')]),
            (regressy, interpolatewf, [('res_file', 'in_file')]),
            (censor_scrubwf, interpolatewf, [('tmask', 'tmask')]),
            (censor_scrubwf, outputnode, [('tmask', 'tmask')]),
            (inputnode, interpolatewf, [('bold_file', 'bold_file')]),
            (interpolatewf, filterdx, [('bold_interpolated', 'in_file')]),
            (filterdx, outputnode, [('filt_file', 'processed_bold')]),
            (censor_scrubwf, outputnode, [('fd_timeseries', 'fd')])
        ])
    else:
        # if inputnode.inputs.custom_confounds:
        #         workflow.connect([
        #             (inputnode, censor_scrubwf, [('custom_confounds', 'custom_confounds')]),
        #              (censor_scrubwf, regressy, [('custom_confounds_censored', 'custom_confounds')]) ])
        workflow.connect([
            (inputnode, censor_scrubwf, [
                ('bold', 'in_file'),
                ('bold_file', 'bold_file'),
                ('bold_mask', 'mask_file')]),
            (inputnode, censor_scrubwf, [('confound_file', 'fmriprep_confounds_file')]),
            (censor_scrubwf, regressy, [
                ('bold_censored', 'in_file'),
                ('fmriprep_confounds_censored', 'confounds')]),
            (inputnode, regressy, [('bold_mask', 'mask')]),
            (inputnode, interpolatewf, [('bold_mask', 'mask_file')]),
            (regressy, interpolatewf, [('res_file', 'in_file')]),
            (censor_scrubwf, interpolatewf, [('tmask', 'tmask')]),
            (censor_scrubwf, outputnode, [('tmask', 'tmask')]),
            (inputnode, interpolatewf, [('bold_file', 'bold_file')]),
            (interpolatewf, filterdx, [('bold_interpolated', 'in_file')]),
            (filterdx, outputnode, [('filt_file', 'processed_bold')]),
            (inputnode, filterdx, [('bold_mask', 'mask')]),
            (censor_scrubwf, outputnode, [('fd_timeseries', 'fd')])
        ])

    if smoothing:
        sigma_lx = fwhm2sigma(smoothing)
        if cifti:
            workflow.__desc__ = workflow.__desc__ + """
The processed bold  was smoothed with the workbench with kernel size (FWHM) of {kernelsize}  mm .
""".format(kernelsize=str(smoothing))
            smooth_data = pe.Node(CiftiSmooth(
                sigma_surf=sigma_lx,
                sigma_vol=sigma_lx,
                direction='COLUMN',
                right_surf=str(
                    get_template("fsLR",
                                 hemi='R',
                                 suffix='sphere',
                                 density='32k')[0]),
                left_surf=str(
                    get_template("fsLR",
                                 hemi='L',
                                 suffix='sphere',
                                 density='32k')[0])),
                name="cifti_smoothing",
                mem_gb=mem_gb)
            workflow.connect([
                (filterdx, smooth_data, [('filt_file', 'in_file')]),
                (smooth_data, outputnode, [('out_file', 'smoothed_bold')])
            ])

        else:
            workflow.__desc__ = workflow.__desc__ + """
The processed bold was smoothed with FSL and kernel size (FWHM) of {kernelsize} mm.
""".format(kernelsize=str(smoothing))
            smooth_data = pe.Node(Smooth(output_type='NIFTI_GZ',
                                         fwhm=smoothing),
                                  name="nifti_smoothing",
                                  mem_gb=mem_gb)

            workflow.connect([
                (filterdx, smooth_data, [('filt_file', 'in_file')]),
                (smooth_data, outputnode, [('smoothed_file', 'smoothed_bold')])
            ])

    return workflow


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def init_censoring_wf(
        mem_gb,
        head_radius,
        custom_confounds,
        low_freq,
        high_freq,
        TR,
        motion_filter_type,
        initial_volumes_to_drop,
        omp_nthreads,
        dummytime=0,
        fd_thresh=0,
        name='censoring'):
    """Creates a workflow that censors volumes in a BOLD dataset.

    This workflow does two steps: removing dummy volumes and censoring noisy
    timepoints.

    Parameters:
    -----------
      mem_gb
        Expected memory consumption in GB
      TR
        Repetition time (seconds)
      head_radius
        Radius of the head for FD calculation (mm)
      custom_confounds
        Path to a custom confounds file
      omp_nthreads: int
        Number of threads to use in parallel
      dummytime: float
        Time in seconds to remove from beginning of scan (default=0)
      fd_thresh: float
      initial_volumes_to_drop: int
        Number of volumes to drop from beginning of scan (default=0)


    Inputs:
    -------
      bold
        Path to a nii.gz file on disk
      bold_file
        Path to the original image in bids [delete me!!]
      bold_mask
        Path to a mask for ``bold``
      confound_file
        Path to the input confounds tsv file (expected fmriprep format)


    Outputs:
    --------
      bold_censored
        Nifti file after censoring has been applied
      fmriprep_confounds_censored

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold', 'bold_file', 'bold_mask', 'confound_file']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=[
        'bold_censored', 'fmriprep_confounds_censored', 'tmask', 'fd',
        'custom_confounds_censored'
    ]),
        name='outputnode')

    censor_scrub = pe.Node(
        CensorScrub(
            fd_thresh=fd_thresh,
            TR=TR,
            low_freq=low_freq,
            high_freq=high_freq,
            motion_filter_type=motion_filter_type,
            head_radius=head_radius,
            custom_confounds=custom_confounds),
        name="censor_scrub",
        mem_gb=mem_gb,
        n_procs=omp_nthreads)

    dummy_scan_wf = pe.Node(
        RemoveTR(initial_volumes_to_drop=initial_volumes_to_drop),
        name="remove_dummy_time",
        mem_gb=mem_gb,
        n_procs=omp_nthreads)

    if dummytime > 0:
        workflow.connect([
            (inputnode, dummy_scan_wf, [('confound_file', 'fmriprep_confounds_file')]),
            (inputnode, dummy_scan_wf, [
                ('bold', 'bold_file'),
                ('bold_mask', 'mask_file')]),
            (dummy_scan_wf, censor_scrub, [
                ('bold_file_dropped_TR', 'in_file'),
                ('fmriprep_confounds_file_dropped_TR', 'fmriprep_confounds_file')]),
            (inputnode, censor_scrub, [
                ('bold_file', 'bold_file'),
                ('bold_mask', 'mask_file')]),
            (censor_scrub, outputnode, [
                ('bold_censored', 'bold_censored'),
                ('fmriprep_confounds_censored', 'fmriprep_confounds_censored'),
                ('tmask', 'tmask'),
                ('fd_timeseries', 'fd')])
        ])
    else:
        if custom_confounds:
            workflow.connect([
                (censor_scrub, outputnode, [('custom_confounds_censored',
                                             'custom_confounds_censored')]),
            ])

        workflow.connect([
            (inputnode, censor_scrub, [
                ('bold', 'in_file'),
                ('bold_file', 'bold_file'),
                ('bold_mask', 'mask_file')]),
            (inputnode, censor_scrub, [('confound_file', 'fmriprep_confounds_file')]),
            (censor_scrub, outputnode, [
                ('bold_censored', 'bold_censored'),
                ('fmriprep_confounds_censored', 'fmriprep_confounds_censored'),
                ('tmask', 'tmask'),
                ('fd_timeseries', 'fd')])])

    return workflow


def init_resd_smoohthing(mem_gb,
                         smoothing,
                         omp_nthreads,
                         cifti=False,
                         name="smoothing"):

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['smoothed_bold']),
                         name='outputnode')

    sigma_lx = fwhm2sigma(smoothing)
    if cifti:
        workflow.__desc__ = """ \
The processed BOLD  was smoothed using Connectome Workbench with a gaussian kernel
size of {kernelsize} mm  (FWHM).
""".format(kernelsize=str(smoothing))

        smooth_data = pe.Node(CiftiSmooth(
            sigma_surf=sigma_lx,
            sigma_vol=sigma_lx,
            direction='COLUMN',
            right_surf=pkgrf(
                'xcp_d', 'data/ciftiatlas/'
                'Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii'
            ),
            left_surf=pkgrf(
                'xcp_d', 'data/ciftiatlas/'
                'Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii'
            )),
            name="cifti_smoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads)

        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('out_file',
                                                      'smoothed_bold')])])

    else:
        workflow.__desc__ = """ \
The processed BOLD was smoothed using  FSL with a gaussian kernel size of {kernelsize} mm  (FWHM).
""".format(kernelsize=str(smoothing))
        smooth_data = pe.Node(Smooth(output_type='NIFTI_GZ', fwhm=smoothing),
                              name="nifti_smoothing",
                              mem_gb=mem_gb,
                              n_procs=omp_nthreads)

        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('smoothed_file',
                                                      'smoothed_bold')])])
    return workflow
