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
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench import CiftiSmooth
from nipype.interfaces.fsl import Smooth

def init_post_process_wf(
    mem_gb,
    TR,
    head_radius,
    lowpass,
    highpass,
    smoothing,
    params,
    surface=False,
    name="post_process_wf",
     ):

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold', 'bold_mask','customs_conf']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold']), name='outputnode')


    confoundmat = pe.Node(ConfoundMatrix(head_radius=head_radius, params=params),
                    name="ConfoundMatrix", mem_gb=mem_gb)
    
    filterdx  = pe.Node(FilteringData(tr=TR,lowpass=lowpass,highpass=highpass),
                    name="filter_the_data", mem_gb=mem_gb)

    regressy = pe.Node (regress(tr=TR),
               name="regress_the_data", mem_gb=mem_gb)
    
    workflow.connect([
             # connect bold confound matrix to extract confound matrix 
            (inputnode, confoundmat, [('bold', 'in_file'),]),
                            
            (inputnode, regressy, [('bold', 'in_file'),
                                ('bold_mask', 'mask'), 
                                ('customs_conf','customs_conf')]),
            (confoundmat,regressy,[('confound_file','confounds')]),
            (regressy, filterdx,[('res_file','in_file')]),
            (inputnode, filterdx,[('bold_mask','mask')]),
            (filterdx,outputnode,[('filt_file','processed_bold')]),
        ])
    
    if smoothing:
        sigma_lx = fwhm2sigma(smoothing)
        if inputnode.inputs.bold.endswith('dtseries.nii'):
            
            lh_midthickness = str(get_template("fsLR", hemi='L',suffix='midthickness',density='32k',)[1])
            rh_midthickness = str(get_template("fsLR", hemi='R',suffix='midthickness',density='32k',)[1])
            smooth_data = pe.Node(CiftiSmooth(sigma_surf = sigma_lx, sigma_vol=sigma_lx, direction ='COLUMN',
                  right_surf=rh_midthickness, left_surf=lh_midthickness), name="cifti_smoothing", mem_gb=mem_gb)

        elif inputnode.inputs.bold.endswith('nii.gz'):
           smooth_data  = pe.Node(Smooth(output_type = 'NIFTI_GZ',fwhm = smoothing),
                   name="nifti_smoothing", mem_gb=mem_gb )
    
    ## smoothing the datt if requested
        workflow.connect([
                (filterdx, smooth_data,[('filt_file','in_file')]),
                (smooth_data, outputnode,[('out_file','smoothed_bold')])       
            ])
    return workflow

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))



    