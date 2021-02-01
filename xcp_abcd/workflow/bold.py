# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing the bold
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_boldpostprocess_wf

"""
import sys
import os
from copy import deepcopy
import nibabel as nb
from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging
from ..utils import collect_data

from  ..workflow import (init_fcon_ts_wf,
    init_post_process_wf,
    init_compute_alff_wf,
    init_3d_reho_wf)

LOGGER = logging.getLogger('nipype.workflow')

def init_boldpostprocess_wf(
     bold_file,
     mni_to_t1w,
     lowpass,
     highpass,
     smoothing,
     head_radius,
     taskid,
     params,
     omp_nthreads,
     template='MNI152NLin2009cAsym',
     num_bold=1,
     layout=None,
     name='bold_process_wf',
      ):
    TR = layout.get_metadata(bold_file)

    workflow = pe.Workflow(name=name)
   
    # get reference and mask
    mask_file,ref_file = _get_ref_mask(fname=bold_file)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file', 'ref_file','bold_mask','customs_conf','mni_to_t1w']),
        name='inputnode')
    
    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.ref_file = ref_file
    inputnode.inputs.bold_mask = mask_file
    inputnode.inputs.mni_to_t1w = mni_to_t1w


    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','alff_out','smoothed_alff', 
                'reho_out','sc207_ts', 'sc207_fc','sc207_ts','sc207_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc']),
        name='outputnode')

    
    # get the mem_bg size for each workflow
    
    mem_gbx = _create_mem_gb(bold_file)
    clean_data_wf = init_post_process_wf( mem_gb=mem_gbx['timeseries'], TR=TR,
                   head_radius=head_radius,lowpass=lowpass,highpass=highpass,
                   smoothing=smoothing,params=params,name='clean_data_wf') 
    
    
    fcon_ts_wf = init_fcon_ts_wf(mem_gb=mem_gbx['timeseries'],
                 t1w_to_native=_t12native(bold_file),
                 template=template,
                 name="fcons_ts_wf")
    
    alff_compute_wf = init_compute_alff_wf(mem_gb=mem_gbx['timeseries'], TR=TR,
                   lowpass=lowpass,highpass=highpass,smoothing=smoothing,
                    name="compue_alff_wf" )

    reho_compute_wf = init_3d_reho_wf(mem_gb=mem_gbx['timeseries'],smoothing=smoothing,
                       name="afni_reho_wf")

    workflow.connect([
        (inputnode,clean_data_wf,[('bold_file','bold'),('bold_mask','bold_mask'),
                                    ('customs_conf','customs_conf')]),

        (inputnode,fcon_ts_wf,[('bold_file','bold_file'),('ref_file','ref_file'),
                              ('mni_to_t1w','mni_to_t1w') ]),
        (clean_data_wf, fcon_ts_wf,[('processed_bold','clean_bold')]),

        (inputnode,alff_compute_wf,['boldmask','boldmask']),
        (clean_data_wf, alff_compute_wf,[('processed_bold','clean_bold')]),

        (inputnode,reho_compute_wf,['boldmask','boldmask']),
        (clean_data_wf, reho_compute_wf,[('processed_bold','clean_bold')]),
         
        #output
        'processed_bold', 
        (clean_data_wf,outputnode,[('processed_bold','processed_bold'),('smoothed_bold','smoothed_bold')]),
        (alff_compute_wf,outputnode,[('alff_out','alff_out'),('smoothed_alff','smoothed_alff')]),
        (reho_compute_wf,outputnode,[('reho_out','reho_out')]),
        (fcon_ts_wf,outputnode,[('sc207_ts','sc207_ts' ),('sc207_fc','sc207_fc'),
                        ('sc207_ts','sc207_ts'),('sc207_fc','sc207_fc'),
                        ('gs360_ts','gs360_ts'),('gs360_fc','gs360_fc'),
                        ('gd333_ts','gd333_ts'),('gd333_fc','gd333_fc')]),
        ])
    return workflow 

def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gbz = {
        'derivative': bold_size_gb,
        'resampled': bold_size_gb * 4,
        'timeseries': bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return mem_gbz

def _get_ref_mask(fname):
    directx = os.path.dirname(fname)
    filename = filename=os.path.basename(fname)
    filex = filename.split('preproc_bold.nii.gz')[0] + 'brain_mask.nii.gz'
    filez = filename.split('_desc-preproc_bold.nii.gz')[0] +'_boldref.nii.gz'
    mask = directx + '/' + filex
    ref = directx + '/' + filez
    return mask, ref

def _t12native(fname):
    directx = os.path.dirname(fname)
    filename = filename=os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]
    
    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'
    
    return t12ref