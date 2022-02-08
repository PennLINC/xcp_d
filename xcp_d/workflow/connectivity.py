# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Time series extractions
functional connectvity matrix
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_fcon_ts_wf
.. autofunction:: init_cifti_conts_wf
"""
 
from nipype.pipeline import engine as pe
import nilearn as nl
from ..interfaces.connectivity import (nifticonnect,get_atlas_nifti,
                      get_atlas_cifti,ApplyTransformsx)
from ..interfaces import connectplot
from nipype.interfaces import utility as niu
from ..utils import CiftiCorrelation, CiftiParcellate,get_transformfile
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def init_fcon_ts_wf(
    mem_gb,
    t1w_to_native,
    mni_to_t1w,
    omp_nthreads,
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
            from xcp_d.workflows import init_fcon_ts_wf
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
    schaefer timeseries and func matrices, 100-1000 parcels

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
[@nilearn]: the Schaefer 200 and 400-parcel resolution atlas [@Schaefer_2017],the Glasser atlas [@Glasser_2016], and the Gordon atlas [@Gordon_2014] atlases. 
Corresponding pair-wise functional connectivity between all regions was computed for each atlas, which was operationalized as the Pearson’s correlation of each parcel’s (unsmoothed) timeseries.
 """.format(nilearnver=nl.__version__)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['bold_file','clean_bold','ref_file',
                   ]), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sc117_ts', 'sc117_fc','sc217_ts','sc217_fc',
                'sc317_ts', 'sc317_fc','sc417_ts','sc417_fc',
                'sc517_ts', 'sc517_fc','sc617_ts','sc617_fc',
                'sc717_ts', 'sc717_fc','sc817_ts','sc817_fc',
                'sc917_ts', 'sc917_fc','sc1017_ts','sc1017_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc',
                'ts50_ts','ts50_fc','connectplot']),
                 name='outputnode')

    inputnode.inputs.bold_file=bold_file


    # get atlases # ietration will be used later
    sc117atlas = get_atlas_nifti(atlasname='schaefer100x17')
    sc217atlas = get_atlas_nifti(atlasname='schaefer200x17')
    sc317atlas = get_atlas_nifti(atlasname='schaefer300x17')
    sc417atlas = get_atlas_nifti(atlasname='schaefer400x17')
    sc517atlas = get_atlas_nifti(atlasname='schaefer500x17')
    sc617atlas = get_atlas_nifti(atlasname='schaefer600x17')
    sc717atlas = get_atlas_nifti(atlasname='schaefer700x17')
    sc817atlas = get_atlas_nifti(atlasname='schaefer800x17')
    sc917atlas = get_atlas_nifti(atlasname='schaefer900x17')
    sc1017atlas = get_atlas_nifti(atlasname='schaefer1000x17')
    gs360atlas = get_atlas_nifti(atlasname='glasser360')
    gd333atlas = get_atlas_nifti(atlasname='gordon333')
    ts50atlas = get_atlas_nifti(atlasname='tiansubcortical')
    
    #get transfrom file
    transformfile = get_transformfile(bold_file=bold_file, mni_to_t1w=mni_to_t1w,
                 t1w_to_native=t1w_to_native)
    
    schaefer_transform = pe.Node(ApplyTransformsx(input_image=sc117atlas,
                       transforms=transformfile,interpolation="MultiLabel",
                       input_image_type=3, dimension=3),
                       name="apply_transform_schaefer", mem_gb=mem_gb,n_procs=omp_nthreads)

    gs360_transform = pe.Node(ApplyTransformsx(input_image=gs360atlas,
                       transforms=transformfile,interpolation="MultiLabel",
                       input_image_type=3,dimension=3),
                       name="apply_tranform_gs36", mem_gb=mem_gb,n_procs=omp_nthreads)
    gd333_transform = pe.Node(ApplyTransformsx(input_image=gd333atlas,
                       transforms=transformfile,interpolation="MultiLabel",
                       input_image_type=3, dimension=3),
                       name="apply_tranform_gd33", mem_gb=mem_gb,n_procs=omp_nthreads)
    
    ts50_transform = pe.Node(ApplyTransformsx(input_image=ts50atlas,
                       transforms=transformfile,interpolation="MultiLabel",
                       input_image_type=3, dimension=3),
                       name="apply_tranform_tian50", mem_gb=mem_gb,n_procs=omp_nthreads)

    matrix_plot = pe.Node(connectplot(in_file=bold_file),name="matrix_plot_wf", mem_gb=mem_gb)

    nifticonnect_sc17 = pe.Node(nifticonnect(),
                    name="sc17_connect", mem_gb=mem_gb)
    nifticonnect_sc27 = pe.Node(nifticonnect(),
                    name="sc27_connect", mem_gb=mem_gb)
    nifticonnect_sc37 = pe.Node(nifticonnect(),
                    name="sc37_connect", mem_gb=mem_gb)
    nifticonnect_sc47 = pe.Node(nifticonnect(),
                    name="sc47_connect", mem_gb=mem_gb)
    nifticonnect_sc57 = pe.Node(nifticonnect(),
                    name="sc57_connect", mem_gb=mem_gb)
    nifticonnect_sc67 = pe.Node(nifticonnect(),
                    name="sc67_connect", mem_gb=mem_gb)
    nifticonnect_sc77 = pe.Node(nifticonnect(),
                    name="sc77_connect", mem_gb=mem_gb)
    nifticonnect_sc87 = pe.Node(nifticonnect(),
                    name="sc87_connect", mem_gb=mem_gb)
    nifticonnect_sc97 = pe.Node(nifticonnect(),
                    name="sc97_connect", mem_gb=mem_gb)
    nifticonnect_sc107 = pe.Node(nifticonnect(),
                    name="sc107_connect", mem_gb=mem_gb)                    
    nifticonnect_gd33 = pe.Node(nifticonnect(),
                    name="gd33_connect", mem_gb=mem_gb)
    nifticonnect_gs36 = pe.Node(nifticonnect(),
                    name="gs36_connect", mem_gb=mem_gb)
    nifticonnect_ts50 = pe.Node(nifticonnect(),
                    name="tiansub_connect", mem_gb=mem_gb)


    workflow.connect([
             ## tansform atlas to bold space
             (inputnode,schaefer_transform,[('ref_file','reference_image'),]),
             (inputnode,gs360_transform,[('ref_file','reference_image'),]),
             (inputnode,gd333_transform,[('ref_file','reference_image'),]),
             (inputnode,ts50_transform,[('ref_file','reference_image'),]),

             # load bold for timeseries extraction and connectivity
             (inputnode,nifticonnect_sc17, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc27, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc37, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc47, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc57, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc67, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc77, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc87, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc97, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_sc107, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_gd33, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_gs36, [('clean_bold','regressed_file'),]),
             (inputnode,nifticonnect_ts50, [('clean_bold','regressed_file'),]),

             # linked atlas
             (schaefer_transform,nifticonnect_sc17,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc27,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc37,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc47,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc57,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc67,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc77,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc87,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc97,[(
                                         'output_image','atlas'),]),
             (schaefer_transform,nifticonnect_sc107,[(
                                         'output_image','atlas'),]),
             (gd333_transform,nifticonnect_gd33,[(
                                         'output_image','atlas'),]),
             (gs360_transform,nifticonnect_gs36,[(
                                         'output_image','atlas'),]),
             (ts50_transform,nifticonnect_ts50,[(
                                         'output_image','atlas'),]),

             # output file
             (nifticonnect_sc17,outputnode,[('time_series_tsv','sc117_ts'),
                                          ('fcon_matrix_tsv','sc117_fc')]),
             (nifticonnect_sc27,outputnode,[('time_series_tsv','sc217_ts'),
                                          ('fcon_matrix_tsv','sc217_fc')]),
             (nifticonnect_sc37,outputnode,[('time_series_tsv','sc317_ts'),
                                          ('fcon_matrix_tsv','sc317_fc')]),
             (nifticonnect_sc47,outputnode,[('time_series_tsv','sc417_ts'),
                                          ('fcon_matrix_tsv','sc417_fc')]),
             (nifticonnect_sc57,outputnode,[('time_series_tsv','sc517_ts'),
                                          ('fcon_matrix_tsv','sc517_fc')]),
             (nifticonnect_sc67,outputnode,[('time_series_tsv','sc617_ts'),
                                          ('fcon_matrix_tsv','sc617_fc')]),
             (nifticonnect_sc77,outputnode,[('time_series_tsv','sc717_ts'),
                                          ('fcon_matrix_tsv','sc717_fc')]),
             (nifticonnect_sc87,outputnode,[('time_series_tsv','sc817_ts'),
                                          ('fcon_matrix_tsv','sc817_fc')]),
             (nifticonnect_sc97,outputnode,[('time_series_tsv','sc917_ts'),
                                          ('fcon_matrix_tsv','sc917_fc')]),
             (nifticonnect_sc107,outputnode,[('time_series_tsv','sc1017_ts'),
                                          ('fcon_matrix_tsv','sc1017_fc')]),
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
    omp_nthreads,
    name="cifti_ts_con_wf",
    ):
    """
    This workflow is for cifti timeseries extraction.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_d.workflows import init_fcon_ts_wf
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
    schaefer time series and func matrices, 100-1000 nodes

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
for the following atlases: the Schaefer 200 and 400-parcel resolution atlas [@Schaefer_2017], the Glasser atlas [@Glasser_2016] and the Gordon atlas [@Gordon_2014]. 
Corresponding pair-wise functional connectivity between all regions was computed for each atlas, which was operationalized as the
 Pearson’s correlation of each parcel’s (unsmoothed) timeseries with the Connectome Workbench.
"""
    inputnode = pe.Node(niu.IdentityInterface(
            fields=['clean_cifti']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sc117_ts', 'sc117_fc','sc217_ts','sc217_fc',
                'sc317_ts', 'sc317_fc','sc417_ts','sc417_fc',
                'sc517_ts', 'sc517_fc','sc617_ts','sc617_fc',
                'sc717_ts', 'sc717_fc','sc817_ts','sc817_fc',
                'sc917_ts', 'sc917_fc','sc1017_ts','sc1017_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc',
                'ts50_ts','ts50_fc','connectplot' ]),
                name='outputnode')


    # get atlas list
    sc117atlas = get_atlas_cifti(atlasname='schaefer100x17')
    sc217atlas = get_atlas_cifti(atlasname='schaefer200x17')
    sc317atlas = get_atlas_cifti(atlasname='schaefer300x17')
    sc417atlas = get_atlas_cifti(atlasname='schaefer400x17')
    sc517atlas = get_atlas_cifti(atlasname='schaefer500x17')
    sc617atlas = get_atlas_cifti(atlasname='schaefer600x17')
    sc717atlas = get_atlas_cifti(atlasname='schaefer700x17')
    sc817atlas = get_atlas_cifti(atlasname='schaefer800x17')
    sc917atlas = get_atlas_cifti(atlasname='schaefer900x17')
    sc1017atlas = get_atlas_cifti(atlasname='schaefer1000x17')
    gs360atlas = get_atlas_cifti(atlasname='glasser360')
    gd333atlas = get_atlas_cifti(atlasname='gordon333')
    ts50atlas = get_atlas_cifti(atlasname='tiansubcortical')

    # timeseries extraction
    sc117parcel = pe.Node(CiftiParcellate(atlas_label=sc117atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='sc117parcel',n_procs=omp_nthreads)
    sc217parcel = pe.Node(CiftiParcellate(atlas_label=sc217atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='sc217parcel',n_procs=omp_nthreads)
    sc317parcel = pe.Node(CiftiParcellate(atlas_label=sc317atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='sc317parcel',n_procs=omp_nthreads)
    sc417parcel = pe.Node(CiftiParcellate(atlas_label=sc417atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc417parcel',n_procs=omp_nthreads)
    sc517parcel = pe.Node(CiftiParcellate(atlas_label=sc517atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc517parcel',n_procs=omp_nthreads)
    sc617parcel = pe.Node(CiftiParcellate(atlas_label=sc617atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc617parcel',n_procs=omp_nthreads)
    sc717parcel = pe.Node(CiftiParcellate(atlas_label=sc717atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc717parcel',n_procs=omp_nthreads)
    sc817parcel = pe.Node(CiftiParcellate(atlas_label=sc817atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc817parcel',n_procs=omp_nthreads)
    sc917parcel = pe.Node(CiftiParcellate(atlas_label=sc917atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc917parcel',n_procs=omp_nthreads)
    sc1017parcel = pe.Node(CiftiParcellate(atlas_label=sc1017atlas,direction='COLUMN'),
                           mem_gb=mem_gb, name='sc1017parcel',n_procs=omp_nthreads)
    gs360parcel = pe.Node(CiftiParcellate(atlas_label=gs360atlas,direction='COLUMN'),
                          mem_gb=mem_gb, name='gs360parcel',n_procs=omp_nthreads)
    gd333parcel = pe.Node(CiftiParcellate(atlas_label=gd333atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='gd333parcel',n_procs=omp_nthreads)
    ts50parcel = pe.Node(CiftiParcellate(atlas_label=ts50atlas,direction='COLUMN'),
                         mem_gb=mem_gb, name='ts50parcel',n_procs=omp_nthreads)

    matrix_plot = pe.Node(connectplot(),name="matrix_plot_wf", mem_gb=mem_gb)
    # correlation
    sc117corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc117corr',n_procs=omp_nthreads)
    sc217corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc217corr',n_procs=omp_nthreads)
    sc317corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc317corr',n_procs=omp_nthreads)
    sc417corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc417corr',n_procs=omp_nthreads)
    sc517corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc517corr',n_procs=omp_nthreads)
    sc617corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc617corr',n_procs=omp_nthreads)
    sc717corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc717corr',n_procs=omp_nthreads)
    sc817corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc817corr',n_procs=omp_nthreads)
    sc917corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc917corr',n_procs=omp_nthreads)
    sc1017corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='sc1017corr',n_procs=omp_nthreads)
    gs360corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='gs360corr',n_procs=omp_nthreads)
    gd333corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='gd333corr',n_procs=omp_nthreads)
    ts50corr = pe.Node(CiftiCorrelation(),mem_gb=mem_gb, name='ts50corr',n_procs=omp_nthreads)

    workflow.connect([
                    (inputnode,sc117parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc217parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc317parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc417parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc517parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc617parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc717parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc817parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc917parcel,[('clean_cifti','in_file')]),
                    (inputnode,sc1017parcel,[('clean_cifti','in_file')]),
                    (inputnode,gd333parcel,[('clean_cifti','in_file')]),
                    (inputnode,gs360parcel,[('clean_cifti','in_file')]),
                    (inputnode,ts50parcel,[('clean_cifti','in_file')]),

                    (sc117parcel,outputnode,[('out_file','sc117_ts',)]),
                    (sc217parcel,outputnode,[('out_file','sc217_ts',)]),
                    (sc317parcel,outputnode,[('out_file','sc317_ts',)]),
                    (sc417parcel,outputnode,[('out_file','sc417_ts',)]),
                    (sc517parcel,outputnode,[('out_file','sc517_ts',)]),
                    (sc617parcel,outputnode,[('out_file','sc617_ts',)]),
                    (sc717parcel,outputnode,[('out_file','sc717_ts',)]),
                    (sc817parcel,outputnode,[('out_file','sc817_ts',)]),
                    (sc917parcel,outputnode,[('out_file','sc917_ts',)]),
                    (sc1017parcel,outputnode,[('out_file','sc1017_ts',)]),
                    (gs360parcel,outputnode,[('out_file','gs360_ts',)]),
                    (gd333parcel,outputnode,[('out_file','gd333_ts',)]),
                    (ts50parcel,outputnode,[('out_file','ts50_ts',)]),

                    (sc117parcel,sc117corr ,[('out_file','in_file',)]),
                    (sc217parcel,sc217corr ,[('out_file','in_file',)]),
                    (sc317parcel,sc317corr ,[('out_file','in_file',)]),
                    (sc417parcel,sc417corr ,[('out_file','in_file',)]),
                    (sc517parcel,sc517corr ,[('out_file','in_file',)]),
                    (sc617parcel,sc617corr ,[('out_file','in_file',)]),
                    (sc717parcel,sc717corr ,[('out_file','in_file',)]),
                    (sc817parcel,sc817corr ,[('out_file','in_file',)]),
                    (sc917parcel,sc917corr ,[('out_file','in_file',)]),
                    (sc1017parcel,sc1017corr ,[('out_file','in_file',)]),
                    (gs360parcel,gs360corr ,[('out_file','in_file',)]),
                    (gd333parcel,gd333corr ,[('out_file','in_file',)]),
                    (ts50parcel,ts50corr ,[('out_file','in_file',)]),

                    (sc117corr,outputnode,[('out_file','sc117_fc',)]),
                    (sc217corr,outputnode,[('out_file','sc217_fc',)]),
                    (sc317corr,outputnode,[('out_file','sc317_fc',)]),
                    (sc417corr,outputnode,[('out_file','sc417_fc',)]),
                    (sc517corr,outputnode,[('out_file','sc517_fc',)]),
                    (sc617corr,outputnode,[('out_file','sc617_fc',)]),
                    (sc717corr,outputnode,[('out_file','sc717_fc',)]),
                    (sc817corr,outputnode,[('out_file','sc817_fc',)]),
                    (sc917corr,outputnode,[('out_file','sc917_fc',)]),
                    (sc1017corr,outputnode,[('out_file','sc1017_fc',)]),
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




