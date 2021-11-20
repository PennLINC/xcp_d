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
from ..interfaces import (ConfoundMatrix,FilteringData,regress)
from ..interfaces import (interpolate,removeTR,censorscrub)
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
    motion_filter_order,
    contigvol,
    cifti=False,
    dummytime=0,
    fd_thresh=0,
    name="post_process_wf",
     ):
    """
    This workflow is organizing workflows including 
    selectign confound matrix, regression and filtering
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows import init_post_process_wf
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
                motion_filter_order,
                contigvol,
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
    contigvol: int 
        number of contigious volumes
    despike: bool
        afni depsike
    motion_filter_order: int 
        respiratory motion filter order
    motion_filter_type: str
        respiratory motion filter type: lp or notch 
    band_stop_min: float 
        respiratory minimum frequency in breathe per minutes(bpm)
    band_stop_max,: float
        respiratory maximum frequency in breathe per minutes(bpm)
    layout : BIDSLayout object
        BIDS dataset layout 
    omp_nthreads : int
        Maximum number of threads an individual process may use
    output_dir : str
        Directory in which to save xcp_abcd output
    fd_thresh
        Criterion for flagging framewise displacement outliers
    head_radius : float 
        radius of the head for FD computation
    params: str
        nuissance regressors to be selected from fmriprep regressors
    smoothing: float
        smooth the derivatives output with kernel size (fwhm)
    custom_conf: str
        path to cusrtom nuissance regressors 
    dummytime: float
        the first vols in seconds to be removed before postprocessing
    
    Inputs
    ------
    bold
       bold or cifti file 
    bold_mask
       bold mask if bold is nifti
    custom_conf
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
Before nuissance regression and filtering of the data, the first {nvol} were discarded,
.Furthermore, any volumes with framewise-displacement greater than 
{fd_thresh} [@satterthwaite2;@power_fd_dvars;@satterthwaite_2013] were  flagged as outliers
 and excluded from nuissance regression.
""".format(nvol=nvolx,fd_thresh=fd_thresh)

    else:
        workflow.__desc__ = workflow.__desc__ + """ \
Before nuissance regression and filtering any volumes with framewise-displacement greater than 
{fd_thresh} [@satterthwaite2;@power_fd_dvars;@satterthwaite_2013] were  flagged as outlier
 and excluded from further analyses.
""".format(fd_thresh=fd_thresh)

    workflow.__desc__ = workflow.__desc__ +  """ \
The following nuissance regressors {regressors} [@mitigating_2018;@benchmarkp;@satterthwaite_2013] were selected 
from nuissance confound matrices of fMRIPrep output.  These nuissance regressors were regressed out 
from the bold data with *LinearRegression* as implemented in Scikit-Learn {sclver} [@scikit-learn].
The residual were then  band pass filtered within the frequency band {highpass}-{lowpass} Hz. 
 """.format(regressors=stringforparams(params=params),sclver=sklearn.__version__,
             lowpass=upper_bpf,highpass=lower_bpf)



    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold','bold_file','bold_mask','custom_conf']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','tmask','fd']), name='outputnode')
    
    inputnode.inputs.bold_file = bold_file
    confoundmat = pe.Node(ConfoundMatrix(head_radius=head_radius, params=params,
                filtertype=motion_filter_type,cutoff=band_stop_max,
                low_freq=band_stop_max,high_freq=band_stop_min,TR=TR,
                filterorder=motion_filter_order),
                    name="ConfoundMatrix", mem_gb=0.1*mem_gb)
    
    filterdx  = pe.Node(FilteringData(tr=TR,lowpass=upper_bpf,highpass=lower_bpf,
                filter_order=bpf_order),
                    name="filter_the_data", mem_gb=0.25*mem_gb)

    regressy = pe.Node(regress(tr=TR),
               name="regress_the_data",mem_gb=0.25*mem_gb)

    censor_scrubwf = pe.Node(censorscrub(fd_thresh=fd_thresh,TR=TR,
                       head_radius=head_radius,contig=contigvol,
                       time_todrop=dummytime),
                      name="censor_scrub",mem_gb=0.1*mem_gb)
    interpolatewf = pe.Node(interpolate(TR=TR),
                  name="interpolation",mem_gb=0.25*mem_gb)
    if dummytime > 0:
        rm_dummytime = pe.Node(removeTR(time_todrop=dummytime,TR=TR),
                      name="remove_dummy_time",mem_gb=0.1*mem_gb)
    
    
    
    
    # get the confound matrix

    workflow.connect([
             # connect bold confound matrix to extract confound matrix 
            (inputnode, confoundmat, [('bold_file', 'in_file'),]),
         ])
    
    if dummytime > 0: 
        workflow.connect([
            (confoundmat,rm_dummytime,[('confound_file','fmriprep_conf'),]),
            (inputnode,rm_dummytime,[('bold','bold_file'),
                   ('bold_mask','mask_file'),]) 
             ])
        if inputnode.inputs.custom_conf:
           workflow.connect([ (inputnode,rm_dummytime,[('custom_conf','custom_conf')]),
                             (rm_dummytime,censor_scrubwf,[('custom_confdropTR','custom_conf')]),
                             (censor_scrubwf,regressy,[('customconf_censored','custom_conf')]),])

        workflow.connect([
              (rm_dummytime,censor_scrubwf,[('bold_file_TR','in_file'),
                         ('fmrip_confdropTR','fmriprep_conf'),]),
              (inputnode,censor_scrubwf,[('bold_file','bold_file'), 
                                    ('bold_mask','mask_file')]),
              (censor_scrubwf,regressy,[('bold_censored','in_file'),
                            ('fmriprepconf_censored','confounds')]),
              (inputnode,regressy,[('bold_mask','mask')]),
              (inputnode, filterdx,[('bold_mask','mask')]),
              (inputnode, interpolatewf,[('bold_mask','mask_file')]),
              (regressy,interpolatewf,[('res_file','in_file'),]),
               (censor_scrubwf,interpolatewf,[('tmask','tmask'),]),
               (censor_scrubwf,outputnode,[('tmask','tmask')]),
               (inputnode,interpolatewf,[('bold_file','bold_file')]),
               (interpolatewf,filterdx,[('bold_interpolated','in_file')]),
               (filterdx,outputnode,[('filt_file','processed_bold')]),
               (censor_scrubwf,outputnode,[('fd_timeseries','fd')])
                ])
    else:
        if inputnode.inputs.custom_conf:
                workflow.connect([
                    (inputnode,censor_scrubwf,[('custom_conf','custom_conf')]),
                     (censor_scrubwf,regressy,[('customconf_censored','custom_conf')]) ])
        
        
        workflow.connect([
              (inputnode,censor_scrubwf,[('bold','in_file'),
                                    ('bold_file','bold_file'), 
                                    ('bold_mask','mask_file'),]),
               (confoundmat,censor_scrubwf,[('confound_file','fmriprep_conf')]),    
               (censor_scrubwf,regressy,[('bold_censored','in_file'),
                            ('fmriprepconf_censored','confounds'),]),
               (inputnode,regressy,[('bold_mask','mask')]),
               (inputnode, interpolatewf,[('bold_mask','mask_file')]),
               (regressy,interpolatewf,[('res_file','in_file'),]),
               (censor_scrubwf,interpolatewf,[('tmask','tmask'),]),
               (censor_scrubwf,outputnode,[('tmask','tmask')]),
               (inputnode,interpolatewf,[('bold_file','bold_file')]),
               (interpolatewf,filterdx,[('bold_interpolated','in_file')]),
               (filterdx,outputnode,[('filt_file','processed_bold')]),
               (inputnode, filterdx,[('bold_mask','mask')]),
               (censor_scrubwf,outputnode,[('fd_timeseries','fd')])
                ])


    if smoothing:
        sigma_lx = fwhm2sigma(smoothing)
        if cifti:
            workflow.__desc__ = workflow.__desc__ + """ 
The processed bold  was smoothed with the workbench with kernel size (FWHM) of {kernelsize}  mm . 
"""         .format(kernelsize=str(smoothing))
            smooth_data = pe.Node(CiftiSmooth(sigma_surf = sigma_lx, sigma_vol=sigma_lx, direction ='COLUMN',
                  right_surf=str(get_template("fsLR", hemi='R',suffix='sphere',density='32k')[0]), 
                  left_surf=str(get_template("fsLR", hemi='L',suffix='sphere',density='32k')[0])),
                   name="cifti_smoothing", mem_gb=mem_gb)
            workflow.connect([
                   (filterdx, smooth_data,[('filt_file','in_file')]),
                   (smooth_data, outputnode,[('out_file','smoothed_bold')])       
                     ])

        else:
            workflow.__desc__ = workflow.__desc__ + """ 
The processed bold was smoothed with FSL and kernel size (FWHM) of {kernelsize} mm. 
"""         .format(kernelsize=str(smoothing))
            smooth_data  = pe.Node(Smooth(output_type = 'NIFTI_GZ',fwhm = smoothing),
                   name="nifti_smoothing", mem_gb=mem_gb )

            workflow.connect([
                   (filterdx, smooth_data,[('filt_file','in_file')]),
                   (smooth_data, outputnode,[('smoothed_file','smoothed_bold')])       
                     ])
    ## smoothing the datt if requested
        
    return workflow













def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

           
def init_censoring_wf( 
    mem_gb,
    TR,
    head_radius,
    contigvol,
    custom_conf,
    omp_nthreads,
    dummytime=0,
    fd_thresh=0,
    name='censoring'
    ):
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold','bold_file','bold_mask','confound_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_censored','fmriprepconf_censored','tmask','fd','customconf_censored']), name='outputnode')


    censorscrub_wf = pe.Node(censorscrub(fd_thresh=fd_thresh,TR=TR,
                       head_radius=head_radius,contig=contigvol,
                       time_todrop=dummytime,custom_conf=custom_conf),
                       name="censor_scrub",mem_gb=mem_gb,n_procs=omp_nthreads)
   
    dummy_scan_wf  = pe.Node(removeTR(time_todrop=dummytime,TR=TR),
                      name="remove_dummy_time",mem_gb=mem_gb,n_procs=omp_nthreads)

    if dummytime > 0: 
        workflow.connect([
            (inputnode,dummy_scan_wf,[('confound_file','fmriprep_conf'),]),
            (inputnode,dummy_scan_wf,[('bold','bold_file'),
                   ('bold_mask','mask_file'),]),

            (dummy_scan_wf,censorscrub_wf,[('bold_file_TR','in_file'),
                                ('fmrip_confdropTR','fmriprep_conf'),]),
            (inputnode,censorscrub_wf,[('bold_file','bold_file'), 
                                    ('bold_mask','mask_file'),]),
            (censorscrub_wf,outputnode,[('bold_censored','bold_censored'),
                                   ('fmriprepconf_censored','fmriprepconf_censored'),
                                   ('tmask','tmask'),('fd_timeseries','fd')]),
            ])
    
    else:
        if custom_conf:
                workflow.connect([
                    (censorscrub_wf,outputnode,[('customconf_censored','customconf_censored')]),
                ])
        
        workflow.connect([
              (inputnode,censorscrub_wf,[('bold','in_file'),
                                    ('bold_file','bold_file'), 
                                    ('bold_mask','mask_file'),]),
               (inputnode,censorscrub_wf,[('confound_file','fmriprep_conf')]), 
               (censorscrub_wf,outputnode,[('bold_censored','bold_censored'),
                                   ('fmriprepconf_censored','fmriprepconf_censored'),
                                   ('tmask','tmask'),('fd_timeseries','fd')]),  
                ])
    
    return workflow 



def init_resd_smoohthing(
    mem_gb,
    smoothing,
    omp_nthreads,
    cifti=False,
    name="smoothing"

   ):

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['smoothed_bold']), name='outputnode')


  
    sigma_lx = fwhm2sigma(smoothing)
    if cifti:
        workflow.__desc__ = """ \
The processed BOLD  was smoothed using Connectome Workbench with a gaussian kernel size of {kernelsize} mm  (FWHM). 
"""     .format(kernelsize=str(smoothing))
        smooth_data = pe.Node(CiftiSmooth(sigma_surf = sigma_lx, sigma_vol=sigma_lx, direction ='COLUMN',
                right_surf  = pkgrf('xcp_abcd','data/ciftiatlas/Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii'),
                left_surf  = pkgrf('xcp_abcd','data/ciftiatlas/Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii')),
                name="cifti_smoothing", mem_gb=mem_gb,n_procs=omp_nthreads)
        workflow.connect([
                   (inputnode, smooth_data,[('bold_file','in_file')]),
                   (smooth_data, outputnode,[('out_file','smoothed_bold')])       
                     ])

    else:
        workflow.__desc__ = """ \
The processed BOLD was smoothed using  FSL with a  gaussian kernel size of {kernelsize} mm  (FWHM). 
"""      .format(kernelsize=str(smoothing))
        smooth_data  = pe.Node(Smooth(output_type = 'NIFTI_GZ',fwhm = smoothing),
                   name="nifti_smoothing", mem_gb=mem_gb,n_procs=omp_nthreads )

        workflow.connect([
                   (inputnode, smooth_data,[('bold_file','in_file')]),
                   (smooth_data, outputnode,[('smoothed_file','smoothed_bold')])       
                     ])
    return workflow



#right_surf=str(get_template("fsLR", hemi='R',suffix='sphere',density='32k')[0]), 
#left_surf=str(get_template("fsLR", hemi='L',suffix='sphere',density='32k')[0])),