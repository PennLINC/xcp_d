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
        motion_filter_order,
        band_stop_max,
        band_stop_min,
        initial_volumes_to_drop,
        cifti=False,
        dummytime=0,
        fd_thresh=0.2,
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
                motion_filter_order,
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
    motion_filter_order: int
        order for motion filter
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
    filtering_wf = pe.Node(FilteringData(TR=TR,
                                     lowpass=upper_bpf,
                                     highpass=lower_bpf,
                                     filter_order=bpf_order),
                       name="filter_the_data",
                       mem_gb=0.25 * mem_gb)

    regressy = pe.Node(regress(TR=TR),
                       name="regress_the_data",
                       mem_gb=0.25 * mem_gb)


    # RF: rename to match
    interpolatewf = pe.Node(interpolate(TR=TR),
                            name="interpolation",
                            mem_gb=0.25 * mem_gb)


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
                (filtering_wf, smooth_data, [('filtered_file', 'in_file')]),
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
                (filtering_wf, smooth_data, [('filtered_file', 'in_file')]),
                (smooth_data, outputnode, [('smoothed_file', 'smoothed_bold')])
            ])

    return workflow


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def init_pre_smoothing(mem_gb,
                       presmoothing,
                       omp_nthreads,
                       cifti=False,
                       name="presmoothing"):

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['presmoothed_bold']),
                         name='outputnode')

    sigma_lx = fwhm2sigma(presmoothing)  # Turn specified FWHM (Full-Width at Half Maximum) 
    # to standard deviation.
    if cifti:  # For ciftis
        workflow.__desc__ = """ \
The preprocessed BOLD input was pre-smoothed using Connectome Workbench with a
gaussian kernel size of {kernelsize} mm  (FWHM).
""".format(kernelsize=str(presmoothing))

        presmooth_data = pe.Node(CiftiSmooth(  # Call connectome workbench to smooth for each
            #  hemisphere
            sigma_surf=sigma_lx,  # the size of the surface kernel
            sigma_vol=sigma_lx,  # the volume of the surface kernel
            direction='COLUMN',  # which direction to smooth along@
            right_surf=pkgrf(  # pull out atlases for each hemisphere
                'xcp_d', 'data/ciftiatlas/'
                'Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii'
            ),
            left_surf=pkgrf(
                'xcp_d', 'data/ciftiatlas/'
                'Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii'
            )),
            name="cifti_presmoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads)

        #  Connect to workflow
        workflow.connect([(inputnode, presmooth_data, [('bold_file', 'in_file')]),
                          (presmooth_data, outputnode, [('out_file',
                                                         'presmoothed_bold')])])

    else:  #  for Nifti
        workflow.__desc__ = """ \
The preprocessed BOLD input was pre-smoothed using FSL with a
gaussian kernel size of {kernelsize} mm  (FWHM).
""".format(kernelsize=str(presmoothing))
        presmooth_data = pe.Node(Smooth(output_type='NIFTI_GZ', fwhm=presmoothing),
                                 name="nifti_presmoothing",
                                 mem_gb=mem_gb,
                                 n_procs=omp_nthreads)  #  Use fslmaths to smooth the image

        #  Connect to workflow
        workflow.connect([(inputnode, presmooth_data, [('bold_file', 'in_file')]),
                          (presmooth_data, outputnode, [('smoothed_file',
                                                         'presmoothed_bold')])])
    return workflow


def init_resd_smoothing(mem_gb,
                        smoothing,
                        omp_nthreads,
                        cifti=False,
                        name="smoothing"):

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['smoothed_bold']),
                         name='outputnode')

    sigma_lx = fwhm2sigma(smoothing) # Turn specified FWHM (Full-Width at Half Maximum) 
    # to standard deviation.
    if cifti:  # For ciftis
        workflow.__desc__ = """ \
The processed BOLD  was smoothed using Connectome Workbench with a gaussian kernel
size of {kernelsize} mm  (FWHM).
""".format(kernelsize=str(smoothing))

        smooth_data = pe.Node(CiftiSmooth(  # Call connectome workbench to smooth for each
            #  hemisphere
            sigma_surf=sigma_lx,  # the size of the surface kernel
            sigma_vol=sigma_lx,  # the volume of the surface kernel
            direction='COLUMN',  # which direction to smooth along@
            right_surf=pkgrf( # pull out atlases for each hemisphere
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

        #  Connect to workflow
        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('out_file',
                                                      'smoothed_bold')])])

    else:  #  for Nifti
        workflow.__desc__ = """ \
The processed BOLD was smoothed using  FSL with a gaussian kernel size of {kernelsize} mm  (FWHM).
""".format(kernelsize=str(smoothing))
        smooth_data = pe.Node(Smooth(output_type='NIFTI_GZ', fwhm=smoothing),  # FWHM = kernel size
                              name="nifti_smoothing",
                              mem_gb=mem_gb,
                              n_procs=omp_nthreads)  #  Use fslmaths to smooth the image

        #  Connect to workflow
        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('smoothed_file',
                                                      'smoothed_bold')])])
    return workflow
