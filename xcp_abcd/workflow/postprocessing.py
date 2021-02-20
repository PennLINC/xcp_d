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
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def init_post_process_wf(
    mem_gb,
    TR,
    head_radius,
    lowpass,
    highpass,
    smoothing,
    params,
    cifti=False,
    scrub=False,
    dummytime=0,
    fd_thresh=0,
    name="post_process_wf",
     ):

    

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold', 'bold_mask','custom_conf']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','tmask']), name='outputnode')

    confoundmat = pe.Node(ConfoundMatrix(head_radius=head_radius, params=params),
                    name="ConfoundMatrix", mem_gb=mem_gb)
    
    filterdx  = pe.Node(FilteringData(tr=TR,lowpass=lowpass,highpass=highpass),
                    name="filter_the_data", mem_gb=mem_gb)

    regressy = pe.Node(regress(tr=TR),
               name="regress_the_data",mem_gb=mem_gb)
    
    if dummytime > 0:
        rm_dummytime = pe.Node(removeTR(time_todrop=dummytime,TR=TR),
                      name="remove_dummy_time",mem_gb=mem_gb)
    
    if fd_thresh > 0:
        censor_scrubwf = pe.Node(censorscrub(fd_thresh=fd_thresh,TR=TR,
                       head_radius=head_radius,
                       time_todrop=dummytime),
                      name="censor_scrub",mem_gb=mem_gb)
    if not scrub:
        interpolatewf = pe.Node(interpolate(TR=TR),
                  name="interpolation",mem_gb=mem_gb)
    
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
           workflow.connect([ (inputnode,rm_dummytime,[('custom_conf','custom_conf')]),])

        if fd_thresh > 0:
            workflow.connect([
              (rm_dummytime,censor_scrubwf,[('bold_file_TR','in_file'),
                         ('fmrip_confdropTR','fmriprep_conf'),]),
              (inputnode,censor_scrubwf,[('bold','bold_file'), 
                                    ('bold_mask','mask_file')]),
              (censor_scrubwf,regressy,[('bold_censored','in_file'),
                            ('fmriprepconf_censored','confounds')]),
              (inputnode,regressy,[('bold_mask','maskfile')]),
              (regressy, filterdx,[('res_file','in_file')]),
               (inputnode, filterdx,[('bold_mask','mask')])
                ])
            if inputnode.inputs.custom_conf:
                workflow.connect([
                    (rm_dummytime,censor_scrubwf,[('custom_confdropTR','custom_conf')]),
                     (censor_scrubwf,regressy,[('customconf_censored','custom_conf')]) ])
        else:
            workflow.connect([
              (rm_dummytime,regressy,[('bold_file_TR','in_file'),
                         ('fmrip_confdropTR','confounds'),
                        ('custom_confdropTR','custom_conf')]),
              (inputnode,regressy,[('bold_mask','maskfile'),]),
              (regressy, filterdx,[('res_file','in_file')]),
               (inputnode, filterdx,[('bold_mask','mask')])])
            
            if inputnode.inputs.custom_conf:
                workflow.connect([
                    (rm_dummytime,regressy,[('custom_confdropTR','custom_conf')]),])
    else:
        if fd_thresh > 0:
            workflow.connect([
              (inputnode,censor_scrubwf,[('bold','in_file'),
                                    ('bold','bold_file'), 
                                    ('bold_mask','mask_file'),]),
               (confoundmat,censor_scrubwf,[('confound_file','fmriprep_conf')]),

              (censor_scrubwf,regressy,[('bold_censored','in_file'),
                            ('fmriprepconf_censored','confounds'),]),
              (inputnode,regressy,[('bold_mask','mask')]),
              (regressy, filterdx,[('res_file','in_file')]),
               (inputnode, filterdx,[('bold_mask','mask')])
                ])
            
            if inputnode.inputs.custom_conf:
                workflow.connect([
                    (inputnode,censor_scrubwf,[('custom_conf','custom_conf')]),
                     (censor_scrubwf,regressy,[('customconf_censored','custom_conf')]) ])


        else:
            workflow.connect([
             # connect bold confound matrix to extract confound matrix 
             (inputnode, regressy, [('bold', 'in_file'),
                                ('bold_mask', 'mask'),('custom_conf','custom_conf')]),
             (confoundmat,regressy,[('confound_file','confounds')]),
             (regressy, filterdx,[('res_file','in_file')]),
             (inputnode, filterdx,[('bold_mask','mask')]),
             ])
            
            if inputnode.inputs.custom_conf:
                workflow.connect([
                    (inputnode,regressy,[('custom_conf','custom_conf')]) ])
    
    if fd_thresh > 0 and not scrub:
        workflow.connect([
             (filterdx,interpolatewf,[('filt_file','in_file'),]),
             (inputnode,interpolatewf,[('bold_mask','mask_file'),]),
             (censor_scrubwf,interpolatewf,[('tmask','tmask'),]),
             (censor_scrubwf,outputnode,[('tmask','tmask')]),
             (inputnode,interpolatewf,[('bold','bold_file')]),
             (interpolatewf,outputnode,[('bold_interpolated','processed_bold')]),
        ])
    else:
        workflow.connect([
             # connect bold confound matrix to extract confound matrix 
            (filterdx,outputnode,[('filt_file','processed_bold')]),
        ])


    if smoothing:
        sigma_lx = fwhm2sigma(smoothing)
        if cifti:
            lh_midthickness = str(get_template("fsLR", hemi='L',suffix='midthickness',density='32k',)[1])
            rh_midthickness = str(get_template("fsLR", hemi='R',suffix='midthickness',density='32k',)[1])

            smooth_data = pe.Node(CiftiSmooth(sigma_surf = sigma_lx, sigma_vol=sigma_lx, direction ='COLUMN',
                  right_surf=rh_midthickness, left_surf=lh_midthickness), name="cifti_smoothing", mem_gb=mem_gb)

        else:
            smooth_data  = pe.Node(Smooth(output_type = 'NIFTI_GZ',fwhm = smoothing),
                   name="nifti_smoothing", mem_gb=mem_gb )

        if fd_thresh > 0 and not scrub:
            workflow.connect([
                    (interpolatewf, smooth_data,[('bold_interpolated','in_file')]),
                   (smooth_data, outputnode,[('out_file','smoothed_bold')])    
                   ])
        else:
            workflow.connect([
                   (filterdx, smooth_data,[('filt_file','in_file')]),
                   (smooth_data, outputnode,[('out_file','smoothed_bold')])       
                     ])
    ## smoothing the datt if requested
        
    return workflow

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))



    