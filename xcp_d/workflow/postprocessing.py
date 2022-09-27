# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing BOLD data."""
import numpy as np
import sklearn
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench import CiftiSmooth
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf
from templateflow.api import get as get_template

from xcp_d.interfaces.filtering import FilteringData
from xcp_d.interfaces.nilearn import Smooth
from xcp_d.utils.utils import fwhm2sigma, stringforparams


def init_post_process_wf(
    mem_gb,
    TR,
    lower_bpf,
    upper_bpf,
    bpf_order,
    smoothing,
    bold_file,
    params,
    cifti=False,
    dummytime=0,
    fd_thresh=0.2,
    name="post_process_wf",
):
    """Organize workflows including selecting confound matrix, regression, and filtering.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.postprocessing import init_post_process_wf
            wf = init_post_process_wf(
                mem_gb=0.1,
                TR=2.,
                lower_bpf=0.009,
                upper_bpf=0.08,
                bpf_order=2,
                smoothing=6,
                bold_file="/path/to/file.nii.gz",
                params="36P",
                cifti=False,
                dummytime=0,
                fd_thresh=0.2,
                name="post_process_wf",
            )

    Parameters
    ----------
    mem_gb : float
    TR: float
        Repetition time in second
    lower_bpf : float
        Lower band pass filter
    upper_bpf : float
        Upper band pass filter
    bpf_order : int
    smoothing: float
        smooth the derivatives output with kernel size (fwhm)
    bold_file: str
        bold file for post processing
    params: str
        nuisance regressors to be selected from fmriprep regressors
    cifti : bool
    dummytime: float
        the first few seconds to be removed before postprocessing
    fd_thresh
        Criterion for flagging framewise displacement outliers
    name : str
        Default is "post_process_wf".

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
        workflow.__desc__ = workflow.__desc__ + f""" \
Before nuisance regression and filtering of the data, the first {nvolx} were
discarded. Furthermore, any volumes with framewise-displacement greater than
{fd_thresh} [@satterthwaite2;@power_fd_dvars;@satterthwaite_2013] were flagged
as outliers and excluded from nuisance regression.
"""

    else:
        workflow.__desc__ = workflow.__desc__ + f""" \
Before nuisance regression and filtering any volumes with
framewise-displacement greater than {fd_thresh}
[@satterthwaite2;@power_fd_dvars;@satterthwaite_2013] were  flagged as outlier
and excluded from further analyses.
"""

    workflow.__desc__ = workflow.__desc__ + f""" \
The following nuisance regressors {stringforparams(params=params)}
[@mitigating_2018;@benchmarkp;@satterthwaite_2013] were selected from nuisance
confound matrices of fMRIPrep output.  These nuisance regressors were regressed
out from the bold data with *LinearRegression* as implemented in Scikit-Learn
{sklearn.__version__} [@scikit-learn].  The residual were then  band pass filtered within the
frequency band {lower_bpf}-{upper_bpf} Hz.
 """

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

    if smoothing:
        sigma_lx = fwhm2sigma(smoothing)
        if cifti:
            workflow.__desc__ = workflow.__desc__ + f"""
The processed bold  was smoothed with the workbench with kernel size (FWHM) of
{str(smoothing)}  mm .
"""
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
            workflow.__desc__ = workflow.__desc__ + f"""
The processed bold was smoothed with Nilearn and kernel size (FWHM) of {str(smoothing)} mm.
"""
            smooth_data = pe.Node(Smooth(fwhm=smoothing),
                                  name="nifti_smoothing",
                                  mem_gb=mem_gb)

            workflow.connect([
                (filtering_wf, smooth_data, [('filtered_file', 'in_file')]),
                (smooth_data, outputnode, [('out_file', 'smoothed_bold')])
            ])

    return workflow


def init_resd_smoothing(
    mem_gb,
    smoothing,
    omp_nthreads,
    cifti=False,
    name="smoothing",
):
    """Smooth BOLD residuals."""
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['smoothed_bold']),
                         name='outputnode')

    sigma_lx = fwhm2sigma(smoothing)  # Turn specified FWHM (Full-Width at Half Maximum)
    # to standard deviation.
    if cifti:  # For ciftis
        workflow.__desc__ = f""" \
The processed BOLD  was smoothed using Connectome Workbench with a gaussian kernel
size of {str(smoothing)} mm  (FWHM).
"""

        smooth_data = pe.Node(CiftiSmooth(  # Call connectome workbench to smooth for each
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
            name="cifti_smoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads)

        #  Connect to workflow
        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('out_file',
                                                      'smoothed_bold')])])

    else:  # for Nifti
        workflow.__desc__ = f""" \
The processed BOLD was smoothed using Nilearn with a gaussian kernel size of {str(smoothing)} mm
(FWHM).
"""
        smooth_data = pe.Node(Smooth(fwhm=smoothing),  # FWHM = kernel size
                              name="nifti_smoothing",
                              mem_gb=mem_gb,
                              n_procs=omp_nthreads)  # Use nilearn to smooth the image

        #  Connect to workflow
        workflow.connect([(inputnode, smooth_data, [('bold_file', 'in_file')]),
                          (smooth_data, outputnode, [('smoothed_file',
                                                      'smoothed_bold')])])
    return workflow
