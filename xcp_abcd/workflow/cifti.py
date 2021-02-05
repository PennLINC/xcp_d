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

from  ..workflow import (init_cifti_conts_wf,
    init_post_process_wf,
    init_compute_alff_wf,
    init_surface_reho_wf)

LOGGER = logging.getLogger('nipype.workflow')


def init_ciftipostprocess_wf(
    cifti_file,
    lowpass,
    highpass,
    smoothing,
    head_radius,
    params,
    omp_nthreads,
    num_cifti=1,
    layout=None,
    name='cifti_process_wf'):
    


    

    workflow = pe.Workflow(name=name)
   
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['cifti_file','customs_conf',]),
        name='inputnode')
    
    inputnode.inputs.cifti_file = cifti_file


    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','alff_out','smoothed_alff', 
                'reho_lh','reho_rh','sc207_ts', 'sc207_fc','sc207_ts','sc207_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc']),
        name='outputnode')

    TR = layout.get_metadata(cifti_file)


    mem_gbx = _create_mem_gb(cifti_file)

    clean_data_wf = init_post_process_wf( mem_gb=mem_gbx['timeseries'], TR=TR,
                   head_radius=head_radius,lowpass=lowpass,highpass=highpass,
                   smoothing=smoothing,params=params,name='clean_data_wf')
    
    cifti_conts_wf = init_cifti_conts_wf(mem_gb=mem_gbx['timeseries'],
                      name='cifti_ts_con_wf')

    alff_compute_wf = init_compute_alff_wf(mem_gb=mem_gbx['timeseries'], TR=TR,
                   lowpass=lowpass,highpass=highpass,smoothing=smoothing,
                    name="compue_alff_wf" )

    reho_compute_wf = init_surface_reho_wf(mem_gb=mem_gbx['timeseries'],smoothing=smoothing,
                       name="afni_reho_wf")

    workflow.connect([
            (inputnode,clean_data_wf,[('cifti_file','bold'),
                                ('customs_conf','customs_conf')]),
            (clean_data_wf, cifti_conts_wf,[('processed_bold','clean_cifti')]),
            (clean_data_wf, alff_compute_wf,[('processed_bold','clean_bold')]),
            (clean_data_wf,reho_compute_wf,[('processed_bold','clean_bold')]),
        
            (clean_data_wf,outputnode,[('processed_bold','processed_bold'),
                                  ('smoothed_bold','smoothed_bold') ]),
            (alff_compute_wf,outputnode,[('alff_out','alff_out')]),
            (reho_compute_wf,outputnode,[('reho_lh','reho_lh'),('reho_rh','reho_rh')]),

            (cifti_conts_wf,outputnode,[('sc207_ts','sc207_ts' ),('sc207_fc','sc207_fc'),
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