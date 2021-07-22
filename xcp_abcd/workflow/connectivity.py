# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Time series extractions
functional connectvity matrix
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_fcon_ts_wf
.. autofunction:: init_cifti_conts_wf
"""
import os
import numpy as np  
from nipype.pipeline import engine as pe
from templateflow.api import get as get_template
import nilearn as nl
from ..interfaces.connectivity import (nifticonnect,get_atlas_nifti,
                      get_atlas_cifti,ApplyTransformsx)
from ..interfaces import connectplot
from nipype.interfaces import utility as niu
from ..utils import CiftiCorrelation, CiftiParcellate,get_transformfile
from pkg_resources import resource_filename as pkgrf
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_fcon_ts_wf(
    mem_gb,
    t1w_to_native,
    mni_to_t1w,
    brain_template,
    bold_file,
    name="fcons_ts_wf",
     ):

    """
    This workflow is for bold timeseries extraction.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows import init_fcon_ts_wf
            wf = init_fcon_ts_wf(
                mem_gb,
                bold_file,
                tw1_to_native,
                template='MNI152NLin2009cAsym',
                name="fcons_ts_wf",
             )
    Parameters
    ----------
    bold_file: str
        bold file for post processing
    mem_gb: float
        memory size in gigabytes
    template: str
        template of bold
    tw1_to_native: str
        transformation files from tw1 to native space ( from fmriprep)
    Inputs
    ------
    bold_file
        bold file from frmiprep
    clean_bold
        clean bold after regressed out nuisscance and filtering
    ref_file
        reference file
    mni_tot1w
        MNI to T1w registration files from fmriprep

    Outputs
    -------
    sc217_ts
        schaefer 200 timeseries
    sc217_fc
        schaefer 200 func matrices
    sc417_ts
        schaefer 400 timeseries
    sc417_fc
        schaefer 400 func matrices
    gs360_ts
        glasser 360 timeseries
    gs360_fc
        glasser 360  func matrices
    gd333_ts
        gordon 333 timeseries
    gd333_fc
        gordon 333 func matrices
    qc_file
        quality control files

    """
    #from niworkflows.interfaces.nilearn import NILEARN_VERSION
    workflow = Workflow(name=name)

    workflow.__desc__ = """
Processed functional timeseries were extracted  from  the residual BOLD signal with  *Nilearn* {nilearnver}'s *NiftiLabelsMasker* for the following atlases
[@nilearn] :the Schaefer 200 and 400-parcel resolution atlas [@Schaefer_2017], Glasser atlas [@Glasser_2016], and Gordon atlas [@Gordon_2014] atlases. 
Corresponding pair-wise functional connectivity between all regions was computed for each atlas, which was operationalized as the Pearson’s correlation of each parcel’s (unsmoothed) timeseries
 """.format(nilearnver=nl.__version__)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold_file','clean_bold','ref_file',
                   ]), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sc217_ts', 'sc217_fc','sc417_ts','sc417_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc' ,
                'ts50_ts','ts50_fc','connectplot']),
                name='outputnode')

    inputnode.inputs.bold_file=bold_file


    # get atlases # ietration will be used later
    sc217atlas = get_atlas_nifti(atlasname='schaefer200x17')
    sc417atlas = get_atlas_nifti(atlasname='schaefer400x17')
    gs360atlas = get_atlas_nifti(atlasname='glasser360')
    gd333atlas = get_atlas_nifti(atlasname='gordon333')
    ts50atlas = get_atlas_nifti(atlasname='tiansubcortical')
    
    #get transfrom file
    transformfile = get_transformfile(bold_file=bold_file, mni_to_t1w=mni_to_t1w,
                 t1w_to_native=t1w_to_native)

    sc217_transform = pe.Node(ApplyTransformsx(input_image=sc217atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor',
                       input_image_type=3, dimension=3),
                       name="apply_tranform_sc27", mem_gb=mem_gb)

    sc417_transform = pe.Node(ApplyTransformsx(input_image=sc417atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor',
                       input_image_type=3, dimension=3),
                       name="apply_tranform_sc47", mem_gb=mem_gb)

    gs360_transform = pe.Node(ApplyTransformsx(input_image=gs360atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor',
                       input_image_type=3, dimension=3),
                       name="apply_tranform_gs36", mem_gb=mem_gb)
    gd333_transform = pe.Node(ApplyTransformsx(input_image=gd333atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor',
                       input_image_type=3, dimension=3),
                       name="apply_tranform_gd33", mem_gb=mem_gb)
    
    ts50_transform = pe.Node(ApplyTransformsx(input_image=ts50atlas,num_threads=2,
                       transforms=transformfile,interpolation='NearestNeighbor',
                       input_image_type=3, dimension=3),
                       name="apply_tranform_tian50", mem_gb=mem_gb)

    matrix_plot = pe.Node(connectplot(in_file=bold_file),name="matrix_plot_wf", mem_gb=mem_gb)

    nifticonnect_sc27 = pe.Node(nifticonnect(),
                    name="sc27_connect", mem_gb=mem_gb)
    nifticonnect_sc47 = pe.Node(nifticonnect(),
                    name="sc47_connect", mem_gb=mem_gb)
    nifticonnect_gd33 = pe.Node(nifticonnect(),
                    name="gd33_connect", mem_gb=mem_gb)
    nifticonnect_gs36 = pe.Node(nifticonnect(),
                    name="gs36_connect", mem_gb=mem_gb)
    nifticonnect_ts50 = pe.Node(nifticonnect(),
                    name="tiansub_connect", mem_gb=mem_gb)


    workflow.connect([
             ## tansform atlas to bold space
             (inputnode,sc217_transform,[('ref_file','reference_image'),]),
             (inputnode,sc417_transform,[('ref_file','reference_image'),]),
             (inputnode,gs360_transform,[('ref_file','reference_image'),]),
             (inputnode,gd333_transform,[('ref_file','reference_image'),]),
             (inputnode,ts50_transform,[('ref_file','reference_image'),]),

             # load bold for timeseries extraction and connectivity
             (inputnode,nifticonnect_sc27, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc47, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_gd33, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_gs36, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_ts50, [('clean_bold','regressed_file'),]),

             # linked atlas
             (sc217_transform,nifticonnect_sc27,[(
                                         'output_image','atlas'),]),
             (sc417_transform,nifticonnect_sc47,[(
                                         'output_image','atlas'),]),
             (gd333_transform,nifticonnect_gd33,[(
                                         'output_image','atlas'),]),
             (gs360_transform,nifticonnect_gs36,[(
                                         'output_image','atlas'),]),
             (ts50_transform,nifticonnect_ts50,[(
                                         'output_image','atlas'),]),

             # output file
             (nifticonnect_sc27,outputnode,[('time_series_tsv','sc217_ts'),
                                          ('fcon_matrix_tsv','sc217_fc')]),
             (nifticonnect_sc47,outputnode,[('time_series_tsv','sc417_ts'),
                                          ('fcon_matrix_tsv','sc417_fc')]),
             (nifticonnect_gs36,outputnode,[('time_series_tsv','gs360_ts'),
                                          ('fcon_matrix_tsv','gs360_fc')]),
             (nifticonnect_gd33,outputnode,[('time_series_tsv','gd333_ts'),
                                          ('fcon_matrix_tsv','gd333_fc')]),

             (nifticonnect_ts50,outputnode,[('time_series_tsv','ts50_ts'),
                                          ('fcon_matrix_tsv','ts50_fc')]),
              # to qcplot
             (nifticonnect_sc27,matrix_plot,[('time_series_tsv','sc217_timeseries')]),
             (nifticonnect_sc47,matrix_plot,[('time_series_tsv','sc417_timeseries')]),
             (nifticonnect_gs36,matrix_plot,[('time_series_tsv','gd333_timeseries')]),
             (nifticonnect_gd33,matrix_plot,[('time_series_tsv','gs360_timeseries')]),
             (matrix_plot,outputnode,[('connectplot','connectplot')])


           ])
    return workflow


def init_cifti_conts_wf(
    mem_gb,
    name="cifti_ts_con_wf",
    ):
    """
    This workflow is for cifti timeseries extraction.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows import init_fcon_ts_wf
            wf = init_fcon_ts_wf(
                mem_gb,
                bold_file,
                tw1_to_native,
                template='MNI152NLin2009cAsym',
                name="fcons_ts_wf",
             )
    Parameters
    ----------

    mem_gb: float
        memory size in gigabytes
    Inputs
    ------
    clean_cifti
        clean cifti after regressed out nuisscance and filtering
    Outputs
    -------
    sc217_ts
        schaefer 200 timeseries
    sc217_fc
        schaefer 200 func matrices
    sc417_ts
        schaefer 400 timeseries
    sc417_fc
        schaefer 400 func matrices
    gs360_ts
        glasser 360 timeseries
    gs360_fc
        glasser 360  func matrices
    gd333_ts
        gordon 333 timeseries
    gd333_fc
        gordon 333 func matrices
    qc_file
        quality control files

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """
Processed functional timeseries were extracted from residual BOLD  using Connectome Workbench[@hcppipelines]:
for the following atlases: Schaefer 200 and 400-parcel resolution atlas [@Schaefer_2017], Glasser atlas [@Glasser_2016] and Gordon atlas [@Gordon_2014]. 
Corresponding pair-wise functional connectivity between all regions was computed for each atlas, which was operationalized as the
 Pearson’s correlation of each parcel’s (unsmoothed) timeseries with the Connectome Workbench.
"""
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_cifti']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sc217_ts', 'sc217_fc','sc417_ts','sc417_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc',
                'ts50_ts','ts50_fc','connectplot' ]),
                name='outputnode')


    # get atlas list
    sc217atlas = get_atlas_cifti(atlasname='schaefer200x17')
    sc417atlas = get_atlas_cifti(atlasname='schaefer400x17')
    gs360atlas = get_atlas_cifti(atlasname='glasser360')
    gd333atlas = get_atlas_cifti(atlasname='gordon333')
    ts50atlas = get_atlas_cifti(atlasname='tiansubcortical')

    # timeseries extraction
    sc217parcel = pe.Node(CiftiParcellate(atlas_label=sc217atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='sc217parcel')
    sc417parcel = pe.Node(CiftiParcellate(atlas_label=sc417atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc417parcel')
    gs360parcel = pe.Node(CiftiParcellate(atlas_label=gs360atlas,direction='COLUMN'),
                          mem_gb=mem_gb, name='gs360parcel')
    gd333parcel = pe.Node(CiftiParcellate(atlas_label=gd333atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='gd333parcel')
    ts50parcel = pe.Node(CiftiParcellate(atlas_label=ts50atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='ts50parcel')

    matrix_plot = pe.Node(connectplot(),name="matrix_plot_wf", mem_gb=mem_gb)
    # correlation
    sc217corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc217corr')
    sc417corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc417corr')
    gs360corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='gs360corr')
    gd333corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='gd333corr')
    ts50corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='ts50corr')

    workflow.connect([
                    (inputnode,sc217parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc417parcel,[('clean_cifti','in_file')]),
                    (inputnode,gd333parcel,[('clean_cifti','in_file')]),
                    (inputnode,gs360parcel,[('clean_cifti','in_file')]),
                    (inputnode,ts50parcel,[('clean_cifti','in_file')]),

                    (sc217parcel,outputnode,[('out_file','sc217_ts',)]),
                    (sc417parcel,outputnode,[('out_file','sc417_ts',)]),
                    (gs360parcel,outputnode,[('out_file','gs360_ts',)]),
                    (gd333parcel,outputnode,[('out_file','gd333_ts',)]),
                    (ts50parcel,outputnode,[('out_file','ts50_ts',)]),

                    (sc217parcel,sc217corr ,[('out_file','in_file',)]),
                    (sc417parcel,sc417corr ,[('out_file','in_file',)]),
                    (gs360parcel,gs360corr ,[('out_file','in_file',)]),
                    (gd333parcel,gd333corr ,[('out_file','in_file',)]),
                    (ts50parcel,ts50corr ,[('out_file','in_file',)]),

                    (sc217corr,outputnode,[('out_file','sc217_fc',)]),
                    (sc417corr,outputnode,[('out_file','sc417_fc',)]),
                    (gs360corr,outputnode,[('out_file','gs360_fc',)]),
                    (gd333corr,outputnode,[('out_file','gd333_fc',)]),
                    (ts50corr,outputnode,[('out_file','ts50_fc',)]),

                    (inputnode,matrix_plot,[('clean_cifti','in_file')]),
                    (sc217parcel,matrix_plot,[('out_file','sc217_timeseries')]),
                    (sc417parcel,matrix_plot,[('out_file','sc417_timeseries')]),
                    (gd333parcel,matrix_plot,[('out_file','gd333_timeseries')]),
                    (gs360parcel,matrix_plot,[('out_file','gs360_timeseries')]),
                    (matrix_plot,outputnode,[('connectplot','connectplot')])
           ])


    return workflow




