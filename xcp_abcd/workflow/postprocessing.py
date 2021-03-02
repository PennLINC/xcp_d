# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing the bold/cifti
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_post_process_wf

"""
import numpy as np
from nipype.pipeline import engine as pe
from templateflow.api import get as get_template
from ..interfaces import (ConfoundMatrix,FilteringData,regress)
from ..interfaces import (interpolate,removeTR,censorscrub)
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench import CiftiSmooth
from nipype.interfaces.fsl import Smooth
import sklearn
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def init_post_process_wf(
    mem_gb,
    TR,
    head_radius,
    lowpass,
    highpass,
    smoothing,
    params,
    motion_filter_type,
    band_stop_max,
    band_stop_min,
    motion_filter_order,
    contigvol,
    cifti=False,
    scrub=False,
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
                lowpass,
                highpass,
                smoothing,
                params,
                omp_nthreads,
                scrub,
                cifti,
                dummytime,
                output_dir,
                fd_thresh,
                TR,
                name="post_process_wf",
             )
    Parameters
    ----------
    
    mem_gb: float
        memory size in gigabytes
    lowpass: float
        low pass filter
    highpass: float
        high pass filter
    smoothing: float
        smooth kernel size in fwhm 
    params: str
        parameter regressed out from bold
    omp_nthreads: int
        number of threads
    scrub: bool
        scrubbing 
    cifti: bool
        if cifti or bold 
    dummytime: float
        volume(s) removed before postprocessing in seconds
    TR: float
        repetition time in seconds
    fd_thresh:
        threshold for FD for censoring/scrubbing
    smoothing:
        smoothing kernel size 
    
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
from nuissance confound matrices by fmriprep.  These nuissance regressors were regressed out 
from the bold data with *LinearRegression* as implemented in Scikit-Learn {sclver} [@scikit-learn].
The residual were then  band pass filtered within the frequency band {highpass}-{lowpass} Hz. 
 """.format(regressors=stringforparams(params=params),sclver=sklearn.__version__,
             lowpass=lowpass,highpass=highpass)



    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold', 'bold_mask','custom_conf']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','tmask']), name='outputnode')

    confoundmat = pe.Node(ConfoundMatrix(head_radius=head_radius, params=params,
                filtertype=motion_filter_type,cutoff=band_stop_max,
                low_freq=band_stop_max,high_freq=band_stop_min,TR=TR,
                filterorder=motion_filter_order),
                    name="ConfoundMatrix", mem_gb=mem_gb)
    
    filterdx  = pe.Node(FilteringData(tr=TR,lowpass=lowpass,highpass=highpass),
                    name="filter_the_data", mem_gb=mem_gb)

    regressy = pe.Node(regress(tr=TR),
               name="regress_the_data",mem_gb=mem_gb)
               
    censor_scrubwf = pe.Node(censorscrub(fd_thresh=fd_thresh,TR=TR,
                       head_radius=head_radius,
                       time_todrop=dummytime),
                      name="censor_scrub",mem_gb=mem_gb)
    interpolatewf = pe.Node(interpolate(TR=TR),
                  name="interpolation",mem_gb=mem_gb)
    if dummytime > 0:
        rm_dummytime = pe.Node(removeTR(time_todrop=dummytime,TR=TR),
                      name="remove_dummy_time",mem_gb=mem_gb)
    
    
    
    
    # get the confpund matrix
    workflow.connect([
             # connect bold confound matrix to extract confound matrix 
            (inputnode, confoundmat, [('bold', 'in_file'),]),
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
              (inputnode,censor_scrubwf,[('bold','bold_file'), 
                                    ('bold_mask','mask_file')]),
              (censor_scrubwf,regressy,[('bold_censored','in_file'),
                            ('fmriprepconf_censored','confounds')]),
              (inputnode,regressy,[('bold_mask','mask')]),
              (inputnode, filterdx,[('bold_mask','mask')]),
              (inputnode, interpolatewf,[('bold_mask','mask_file')]),
              (regressy,interpolatewf,[('res_file','in_file'),]),
               (censor_scrubwf,interpolatewf,[('tmask','tmask'),]),
               (censor_scrubwf,outputnode,[('tmask','tmask')]),
               (inputnode,interpolatewf,[('bold','bold_file')]),
               (interpolatewf,filterdx,[('bold_interpolated','in_file')]),
               (filterdx,outputnode,[('filt_file','processed_bold')])
                ])
    else:
        if inputnode.inputs.custom_conf:
                workflow.connect([
                    (inputnode,censor_scrubwf,[('custom_conf','custom_conf')]),
                     (censor_scrubwf,regressy,[('customconf_censored','custom_conf')]) ])
        
        
        workflow.connect([
              (inputnode,censor_scrubwf,[('bold','in_file'),
                                    ('bold','bold_file'), 
                                    ('bold_mask','mask_file'),]),
               (confoundmat,censor_scrubwf,[('confound_file','fmriprep_conf')]),    
               (censor_scrubwf,regressy,[('bold_censored','in_file'),
                            ('fmriprepconf_censored','confounds'),]),
               (inputnode,regressy,[('bold_mask','mask')]),
               (inputnode, interpolatewf,[('bold_mask','mask_file')]),
               (regressy,interpolatewf,[('res_file','in_file'),]),
               (censor_scrubwf,interpolatewf,[('tmask','tmask'),]),
               (censor_scrubwf,outputnode,[('tmask','tmask')]),
               (inputnode,interpolatewf,[('bold','bold_file')]),
               (interpolatewf,filterdx,[('bold_interpolated','in_file')]),
               (filterdx,outputnode,[('filt_file','processed_bold')]),
               (inputnode, filterdx,[('bold_mask','mask')]),
                ])


    if smoothing:
        sigma_lx = fwhm2sigma(smoothing)
        if cifti:
            workflow.__desc__ = workflow.__desc__ + """ \
The processed bold  was smoothed with the workbench with kernel size (FWHM) of {kernelsize}  mm . 
"""         .format(kernelsize=str(smoothing))
            lh_midthickness = str(get_template("fsLR", hemi='L',suffix='midthickness',density='32k',)[1])
            rh_midthickness = str(get_template("fsLR", hemi='R',suffix='midthickness',density='32k',)[1])

            smooth_data = pe.Node(CiftiSmooth(sigma_surf = sigma_lx, sigma_vol=sigma_lx, direction ='COLUMN',
                  right_surf=rh_midthickness, left_surf=lh_midthickness), name="cifti_smoothing", mem_gb=mem_gb)

        else:
            workflow.__desc__ = workflow.__desc__ + """ \
The processed bold was smoothed with FSL and kernel size (FWHM) of {kernelsize} mm. 
"""         .format(kernelsize=str(smoothing))
            smooth_data  = pe.Node(Smooth(output_type = 'NIFTI_GZ',fwhm = smoothing),
                   name="nifti_smoothing", mem_gb=mem_gb )

    workflow.connect([
                   (filterdx, smooth_data,[('filt_file','in_file')]),
                   (smooth_data, outputnode,[('out_file','smoothed_bold')])       
                     ])
    ## smoothing the datt if requested
        
    return workflow

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def stringforparams(params):
    if params == '24P':
        bsignal = "including six motion parameters with their temporal derivatives, \
            quadratic expansion of both six motion paramters and their derivatives  \
            to make a total of 24 nuissance regressors "
    if params == '27P':
        bsignal = "including six motion parameters with their temporal derivatives, \
            quadratic expansion of both six motion paramters and their derivatives, global signal,  \
            white and CSF signal to make a total 27 nuissance regressors"
    if params == '36P':
        bsignal= "including six motion parameters, white ,CSF and global signals,  with their temporal derivatives, \
            quadratic expansion of these nuissance regressors and their derivatives  \
            to make a total 36 nuissance regressors"
    return bsignal
           
    


    