# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing the bold/cifti
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_post_process_wf

"""
import numpy as np
from nipype.pipeline import engine as pe
from ..interfaces import (computealff, surfaceReho,brainplot)
from nipype.interfaces import utility as niu
from ..utils import CiftiSeparateMetric
from nipype.interfaces.workbench import CiftiSmooth
from nipype.interfaces.fsl import Smooth
from templateflow.api import get as get_template
from nipype.interfaces.afni import ReHo as ReHo
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def init_compute_alff_wf(
    mem_gb,
    TR,
    lowpass,
    highpass,
    smoothing,
    cifti,
    name="compute_alff_wf",
    ):

    """
    This workflow compute alff for both nifit and cifti 
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows import init_compute_alff_wf
            wf = init_compute_alff_wf(
                mem_gb,
                TR,
                lowpass,
                highpass,
                smoothing,
                cifti,
                name="compute_alff_wf",
             )
    Parameters
    ----------
    
    mem_gb: float
        memory size in gigabytes
    TR: float 
        repetition time 
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
    cifti: bool
        if cifti or bold 
    
    Inputs
    ------
    clean_bold
       residual and filtered 
    bold_mask
       bold mask if bold is nifti

    Outputs
    -------
    alff_out
        alff output
    smoothed_alff
        smoothed alff  output 
    tmask
        temporal mask
    """

    workflow = Workflow(name=name)

    workflow.__desc__ = """ \
The amplitude of low-frequency fluctuation (ALFF) [@alff] was computed 
by transforming  the processed BOLD timeseries  to the frequency domain. 
The power spectral of was computed within freqeuncy band {highpass} - {lowpass} 
Hz and the averaged square root of power spectral were obtained  at each voxel as ALFF. 
""" .format(highpass=highpass,lowpass=lowpass)

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
        if not cifti:
            workflow.__desc__ = workflow.__desc__ + """ \
The processed bold was smoothed with FSL and kernel size of {kernelsize} mm. 
"""         .format(kernelsize=str(smoothing))
            smooth_data  = pe.Node(Smooth(output_type = 'NIFTI_GZ',fwhm = smoothing),
                   name="ciftismoothing", mem_gb=mem_gb )
            workflow.connect([
               (alff_compt, smooth_data,[('alff_out','in_file')]),
               (smooth_data, outputnode,[('smoothed_file','smoothed_alff')]),
             ])

        else:
            workflow.__desc__ = workflow.__desc__ + """ \
The ALL  was smoothed with workbench and
kernel size of {kernelsize} mm. 
"""         .format(kernelsize=str(smoothing))
            sigma_lx = fwhm2sigma(smoothing)
            lh_midthickness = str(get_template("fsLR", hemi='L',suffix='vaavg_midthickness',density='32k'))
            rh_midthickness = str(get_template("fsLR", hemi='R',suffix='vaavg_midthickness',density='32k'))
            smooth_data = pe.Node(CiftiSmooth(sigma_surf = sigma_lx, sigma_vol=sigma_lx, direction ='COLUMN',
                  right_surf=rh_midthickness, left_surf=lh_midthickness), name="ciftismoothing", mem_gb=mem_gb)
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

    """
    This workflow compute surface reho
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows import init_surface_reho_wf
            wf = init_surface_reho_wf(
                mem_gb,
                smoothing,
                name="surface_reho_wf",
             )
    Parameters
    ----------
    
    mem_gb: float
        memory size in gigabytes
    smoothing: float
        smooth kernel size in fwhm 
    
    Inputs
    ------
    clean_bold
       residual and filtered, cifti 

    Outputs
    -------
    lh_reho
        left hemisphere surface reho
    rh_reho
        right hemisphere surface reho
    """


    workflow = Workflow(name=name)
    workflow.__desc__ = """ \
The left and right hemispheres were extacted from the processed bold with the workbench. 
For each hemisphere, regional homogeneity (ReHo) was computed with surface-based 
*2dReHo* [@surface_reho] . For a given vertex on the surface, the nearest-neighbor 
vertices were identified and computed Kendall's coefficient of concordance (KCC) as reho. 
""" 

    
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_bold']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['lh_reho','rh_reho']), name='outputnode')

    lh_surf = pe.Node(CiftiSeparateMetric(metric='CORTEX_LEFT',direction="COLUMN"), 
                  name="separate_lh", mem_gb=mem_gb)
    rh_surf = pe.Node(CiftiSeparateMetric(metric='CORTEX_RIGHT',direction="COLUMN"), 
                  name="separate_rh", mem_gb=mem_gb )

    lh_reho = pe.Node(surfaceReho(surf_hemi='L'),name="reho_lh", mem_gb=mem_gb)
    rh_reho = pe.Node(surfaceReho(surf_hemi='R'),name="reho_rh", mem_gb=mem_gb)

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
    name="afni_reho_wf",
    ):

    """
    This workflow compute surface reho
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows import init_3d_reho_wf
            wf = init_3d_reho_wf(
                mem_gb,
                smoothing,
                name="afni_reho_wf",
             )
    Parameters
    ----------
    
    mem_gb: float
        memory size in gigabytes
    smoothing: float
        smooth kernel size in fwhm 
    
    Inputs
    ------
    clean_bold
       residual and filtered, nifti
    bold_mask
       bold mask 

    Outputs
    -------
    reho_out
        reho output
    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """ \
The regional homogeneity (ReHo) for the processed bold was computed with 
AFNI *3dReHo* [@afni] with vertices neighborhood. 
""" 
    
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_bold','bold_mask']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['reho_out']), name='outputnode')

    compute_reho = pe.Node(ReHo(neighborhood='vertices'), name="reho_3d", mem_gb=mem_gb)

    workflow.connect([
         (inputnode, compute_reho,[('clean_bold','in_file'),
                         ('bold_mask','mask_file')]),
         (compute_reho,outputnode,[('out_file','reho_out')]),
        ])

    return workflow


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


