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
from ..interfaces import (computealff, surfaceReho)
from nipype.interfaces import utility as niu
from ..utils import CiftiSeparateMetric
from nipype.interfaces.workbench import CiftiSmooth
from nipype.interfaces.fsl import Smooth
from templateflow.api import get as get_template
from nipype.interfaces.afni import ReHo as ReHo

 


def init_compute_alff_wf(
    mem_gb,
    TR,
    lowpass,
    highpass,
    smoothing,
    name="compue_alff_wf",
    ):

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_bold', 'bold_mask']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['alff_out','smoothed_alff']), name='outputnode')

    alff_compt = pe.Node(computealff(tr=TR,lowpass=lowpass,highpass=highpass),
                      mem_gb=mem_gb,name='alff_compt')
    
    workflow.connect([ 
            (inputnode,alff_compt,[('clean_bold','in_file'),
                           ('bold_mask','mask')]),
            (alff_compt,outputnode,[('alff_out','alff_out')]),
            ])
    
    if smoothing:
        if inputnode.inputs.clean_bold.endswith('nii.gz'):
            smooth_data  = pe.Node(Smooth(output_type = 'NIFTI_GZ',fwhm = smoothing),
                   name="nifti smoothing", mem_gb=mem_gb )

        elif inputnode.inputs.bold.endswith('dtseries.nii'): 
            sigma_lx = fwhm2sigma(smoothing)
            lh_midthickness = str(get_template("fsLR", hemi='L',suffix='midthickness',density='32k',)[1])
            rh_midthickness = str(get_template("fsLR", hemi='R',suffix='midthickness',density='32k',)[1])
            smooth_data = pe.Node(CiftiSmooth(sigma_surf = sigma_lx, sigma_vol=sigma_lx, direction ='COLUMN',
                  right_surf=rh_midthickness, left_surf=lh_midthickness), name="cifti smoothing", mem_gb=mem_gb)
        
        workflow.connect([
           (alff_compt, smooth_data,[('alff_out','in_file')]),
           (smooth_data, outputnode,[('out_file','smoothed_alff')]),
           ])
    return workflow


def init_surface_reho_wf(
    mem_gb,
    smoothing,
    name="surface_reho_wf",
    ):

    workflow = pe.Workflow(name=name)
    
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_bold']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['lh_reho','rh_reho']), name='outputnode')

    lh_surf = pe.Node(CiftiSeparateMetric(metric='CORTEX_LEFT',direction="COLUMN"), 
                  name="separate_lh", mem_gb=mem_gb)
    rh_surf = pe.Node(CiftiSeparateMetric(metric='CORTEX_RIGHT',direction="COLUMN"), 
                  name="separate_rh", mem_gb=mem_gb )

    lh_reho = pe.Node(surfaceReho(),name="reho_lh", mem_gb=mem_gb)
    rh_reho = pe.Node(surfaceReho(),name="reho_rh", mem_gb=mem_gb)

    workflow.connect([
         (inputnode,lh_surf,[('clean_bold','in_file')]),
         (inputnode,rh_surf,[('clean_bold','in_file')]),
         (lh_surf,lh_reho,[('out_file','surf_bold')]),
         (rh_surf,rh_reho,[('out_file','surf_bold')]),
         (lh_reho,outputnode,[('surf_gii','lh_reho')]),
         (rh_reho,outputnode,[('surf_gii','rh_reho')]),
        ])

    return workflow

def init_3d_reho_wf(
    mem_gb,
    smoothing,
    name="3d_reho_wf",
    ):

    workflow = pe.Workflow(name=name)
    
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_bold']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['reho_out']), name='outputnode')

    compute_reho = pe.Node(ReHo(neighborhood=vertices), name="reho_3d", mem_gb=mem_gb)

    workflow.connect([
         (inputnode, compute_reho,[('clean_bold','in_file')]),
         ( compute_reho,outputnode,[('out_file','reho_out')]),
        ])

    return workflow


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))