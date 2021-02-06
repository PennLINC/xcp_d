# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Time series extractions 
functional connectvity matrix
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_fcon_ts_wf
.. autofunction:: init_cifti_conts_wf
"""
import numpy as np
import os 
from nipype.pipeline import engine as pe
from templateflow.api import get as get_template
from ..interfaces.connectivity import (nifticonnect,get_atlas_nifti, 
                      get_atlas_cifti,ApplyTransformsx)
from nipype.interfaces import utility as niu
from ..utils import CiftiCorrelation, CiftiParcellate
from pkg_resources import resource_filename as pkgrf
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_fcon_ts_wf(
    mem_gb,
    t1w_to_native,
    template,
    bold_file,
    name="fcons_ts_wf",
     ):
   

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold_file','clean_bold','ref_file',
                   'mni_to_t1w']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sc207_ts', 'sc207_fc','sc407_ts','sc407_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc' ]), 
                name='outputnode')

    inputnode.inputs.bold_file=bold_file

    # get atlases # ietration will be used later 
    sc207atlas = get_atlas_nifti(atlasname='schaefer200x7')
    sc407atlas = get_atlas_nifti(atlasname='schaefer400x7')
    gs360atlas = get_atlas_nifti(atlasname='glasser360')
    gd333atlas = get_atlas_nifti(atlasname='gordon333')


    file_base = os.path.basename(str(bold_file))
    if template in file_base:
        transformfile = 'identity'
    elif 'T1w' in file_base: 
        transformfile = str(inputnode.inputs.mni_to_t1w)
    elif not  template  or  'T1w' in file_base:
        transformfile = [str(inputnode.inputs.mni_to_t1w), str(t1w_to_native)]

    sc207_transform = pe.Node(ApplyTransformsx(input_image=sc207atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor'),
                       name="apply_tranform_sc27", mem_gb=mem_gb)
    
    sc407_transform = pe.Node(ApplyTransformsx(input_image=sc407atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor'),
                       name="apply_tranform_sc47", mem_gb=mem_gb)
    
    gs360_transform = pe.Node(ApplyTransformsx(input_image=gs360atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor'),
                       name="apply_tranform_gs36", mem_gb=mem_gb)
    gd333_transform = pe.Node(ApplyTransformsx(input_image=gd333atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor'),
                       name="apply_tranform_gd33", mem_gb=mem_gb)

    nifticonnect_sc27 = pe.Node(nifticonnect(), 
                    name="sc27_connect", mem_gb=mem_gb)
    nifticonnect_sc47 = pe.Node(nifticonnect(), 
                    name="sc47_connect", mem_gb=mem_gb)
    nifticonnect_gd33 = pe.Node(nifticonnect(), 
                    name="gd33_connect", mem_gb=mem_gb)
    nifticonnect_gs36 = pe.Node(nifticonnect(), 
                    name="gs36_connect", mem_gb=mem_gb)

    
    workflow.connect([
             ## tansform atlas to bold space 
             (inputnode,sc207_transform,[('ref_file','reference_image'),('bold_file','input_image')]),
             (inputnode,sc407_transform,[('ref_file','reference_image'),('bold_file','input_image')]),
             (inputnode,gs360_transform,[('ref_file','reference_image'),('bold_file','input_image')]),
             (inputnode,gd333_transform,[('ref_file','reference_image'),('bold_file','input_image')]),
             
             # load bold for timeseries extraction and connectivity
             (inputnode,nifticonnect_sc27, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc47, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_gd33, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_gs36, [('clean_bold','regressed_file'),]),

             # linked atlas
             (sc207_transform,nifticonnect_sc27,[(
                                         'output_image','atlas'),]),
             (sc407_transform,nifticonnect_sc47,[(
                                         'output_image','atlas'),]),
             (gd333_transform,nifticonnect_gd33,[(
                                         'output_image','atlas'),]),
             (gs360_transform,nifticonnect_gs36,[(
                                         'output_image','atlas'),]),
             
             # output file
             (nifticonnect_sc27,outputnode,[('time_series_tsv','sc207_ts'),
                                          ('fcon_matrix_tsv','sc207_fc')]),
             (nifticonnect_sc47,outputnode,[('time_series_tsv','sc407_ts'),
                                          ('fcon_matrix_tsv','sc407_fc')]),
             (nifticonnect_gs36,outputnode,[('time_series_tsv','gs360_ts'),
                                          ('fcon_matrix_tsv','gs360_fc')]),
             (nifticonnect_gs36,outputnode,[('time_series_tsv','gd333_ts'),
                                          ('fcon_matrix_tsv','gd333_fc')]),
           ])
    return workflow


def init_cifti_conts_wf(
    mem_gb,
    name="cifti_ts_con_wf", 
    ):
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_cifti']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sc207_ts', 'sc207_fc','sc407_ts','sc407_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc' ]), 
                name='outputnode')

    
    # get atlas list 
    sc207atlas = get_atlas_cifti(atlasname='schaefer200x7')
    sc407atlas = get_atlas_cifti(atlasname='schaefer400x7')
    gs360atlas = get_atlas_cifti(atlasname='glasser360')
    gd333atlas = get_atlas_cifti(atlasname='gordon333')
    
    # timeseries extraction
    sc207parcel = pe.Node(CiftiParcellate(atlas_label=sc207atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='sc207parcel')
    sc407parcel = pe.Node(CiftiParcellate(atlas_label=sc407atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc407parcel')
    gs360parcel = pe.Node(CiftiParcellate(atlas_label=gs360atlas,direction='COLUMN'),
                          mem_gb=mem_gb, name='gs360parcel')
    gd333parcel = pe.Node(CiftiParcellate(atlas_label=gd333atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='gd333parcel')

    # correlation
    sc207corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc207corr')
    sc407corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc407corr')
    gs360corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='gs360corr')
    gd333corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='gd333corr')

    workflow.connect([ 
                    (inputnode,sc207parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc407parcel,[('clean_cifti','in_file')]),
                    (inputnode,gd333parcel,[('clean_cifti','in_file')]),
                    (inputnode,gs360parcel,[('clean_cifti','in_file')]),

                    (sc207parcel,outputnode,[('out_file','sc207_ts',)]),
                    (sc407parcel,outputnode,[('out_file','sc407_ts',)]),
                    (gs360parcel,outputnode,[('out_file','gs360_ts',)]),
                    (gd333parcel,outputnode,[('out_file','gd333_ts',)]),
                     
                    (sc207parcel,sc207corr ,[('out_file','in_file',)]),
                    (sc407parcel,sc407corr ,[('out_file','in_file',)]),
                    (gs360parcel,gs360corr ,[('out_file','in_file',)]),
                    (gd333parcel,gd333corr ,[('out_file','in_file',)]),

                    (sc207parcel,outputnode,[('out_file','sc207_fc',)]),
                    (sc407parcel,outputnode,[('out_file','sc407_fc',)]),
                    (gs360parcel,outputnode,[('out_file','gs360_fc',)]),
                    (gd333parcel,outputnode,[('out_file','gd333_fc',)])
           ])


    return workflow
        