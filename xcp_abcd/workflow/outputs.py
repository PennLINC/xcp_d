# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import os 
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces import bids
from ..utils import bid_derivative

class DerivativesDataSink(bid_derivative):
    out_path_base = 'xcp_abcd'


def init_writederivatives_wf(
     bold_file,
     lowpass,
     highpass,
     smoothing,
     params,
     omp_nthreads,
     cifti,
     dummytime,
     output_dir,
     TR,
     name='write_derivatives_wf',
     ):
    """
    This workflow is for writing out the output in bids
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows import init_writederivatives_wf
            wf = init_writederivatives_wf(
                mem_gb,
                bold_file,
                lowpass,
                highpass,
                smoothing,
                params,
                omp_nthreads,
                scrub,
                cifti,
                dummytime,
                output_dir,
                TR,
                name="fcons_ts_wf",
             )
    Parameters
    ----------
    
    mem_gb: float
        memory size in gigabytes
    bold_file: str
        bold or cifti files 
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
    output_dir: str
        output directory
    TR: float
        repetition time in seconds 
    
    Inputs
    ------
    sc207_ts
        schaefer 200 timeseries
    sc207_fc
        schaefer 200 func matrices 
    sc407_ts
        schaefer 400 timeseries
    sc407_fc
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
    processed_bold
        clean bold after regression and filtering
    smoothed_bold
        smoothed clean bold
    alff_out
        alff niifti
    smoothed_alff
        smoothed alff 
    reho_lh
        reho left hemisphere
    reho_rh
        reho right hemisphere
    """
    workflow = Workflow(name=name)
    

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['processed_bold', 'smoothed_bold','alff_out','smoothed_alff', 
                'reho_out','sc207_ts', 'sc207_fc','sc407_ts','sc407_fc','reho_lh','reho_rh',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc','qc_file','fd']), name='inputnode')
    
    cleandata_dict= { 'RepetitionTime': TR, 'Freq Band': [highpass,lowpass],'nuissance parameters': params,  
                    'dummy vols' :  np.int(dummytime/TR)}
    smoothed_dict = { 'FWHM': smoothing }


    if not cifti:
        dv_cleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=cleandata_dict,dismiss_entities=['desc'], desc='residual',
                 extension='.nii.gz',source_file=bold_file,compression=True),
            name='dv_cleandata_wf', run_without_submitting=True, mem_gb=2)
            
        dv_alff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],compression=True,desc='alff',
                 extension='.nii.gz',source_file=bold_file),
            name='dv_alff_wf', run_without_submitting=True,mem_gb=1)
    
        dv_qcfile_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='qc',source_file=bold_file,
                 compression=True,extension='.tsv'),
            name='dv_qcfile_wf', run_without_submitting=True, mem_gb=1)

        dv_sc207ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer207',desc='timeseries',source_file=bold_file),
            name='dv_sc207ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc407ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer407',desc='timeseries',source_file=bold_file),
            name='dv_sc407ts_wf', run_without_submitting=True, mem_gb=1)
    
        dv_gs360ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Glasser',desc='timeseries',source_file=bold_file),
            name='dv_gs360ts_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Gordon',desc='timeseries',source_file=bold_file),
            name='dv_gd333_wf', run_without_submitting=True, mem_gb=1)

        dv_sc207fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer207',desc='connectivity',source_file=bold_file),
            name='dv_sc207fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc407fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer407',desc='connectivity',source_file=bold_file),
            name='dv_sc407fc_wf', run_without_submitting=True, mem_gb=1)
    
        dv_gs360fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Glaseer',desc='connectivity',source_file=bold_file),
            name='dv_gs333_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Gordon',desc='connectivity',source_file=bold_file),
            name='dv_gd333fc_wf', run_without_submitting=True, mem_gb=1)

        dv_reho_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,extension='.nii.gz', 
                 dismiss_entities=['desc'],compression=True,desc='reho',source_file=bold_file),
            name='dv_reho_wf', run_without_submitting=True, mem_gb=1)

        workflow.connect([
         (inputnode,dv_cleandata_wf,[('processed_bold','in_file')]),
         (inputnode,dv_alff_wf,[('alff_out','in_file')]),
         (inputnode,dv_reho_wf,[('reho_out','in_file')]),
         (inputnode,dv_qcfile_wf,[('qc_file','in_file')]),
         (inputnode,dv_sc207ts_wf,[('sc207_ts','in_file')]),
         (inputnode,dv_sc407ts_wf,[('sc407_ts','in_file')]),
         (inputnode,dv_gs360ts_wf,[('gs360_ts','in_file')]),
         (inputnode,dv_gd333ts_wf,[('gd333_ts','in_file')]),
         (inputnode,dv_sc207fc_wf,[('sc207_fc','in_file')]),
         (inputnode,dv_sc407fc_wf,[('sc407_fc','in_file')]),
         (inputnode,dv_gs360fc_wf,[('gs360_fc','in_file')]),
         (inputnode,dv_gd333fc_wf,[('gd333_fc','in_file')]),   
           ])
        if smoothing:
            dv_smoothcleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], desc='clean_smooth',source_file=bold_file),
            name='dv_smoothcleandata_wf', run_without_submitting=True, mem_gb=2)

            dv_smoothalff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], desc='alff_smooth',source_file=bold_file),
            name='dv_smoothalff_wf', run_without_submitting=True, mem_gb=1)
  
            workflow.connect([
                (inputnode,dv_smoothcleandata_wf,[('smoothed_bold','in_file')]),
                (inputnode,dv_smoothalff_wf,[('smoothed_alff','in_file')]),
            ])

    if cifti:
        dv_cleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=cleandata_dict,dismiss_entities=['desc'], desc='clean',
                 source_file=bold_file,density='91k',extension='.dtseries.nii'),
            name='dv_cleandata_wf', run_without_submitting=True, mem_gb=2)
            
        dv_alff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='alff',density='91k',extension='.dscalar.nii',
                 source_file=bold_file,check_hdr=False),
            name='dv_alff_wf', run_without_submitting=True, mem_gb=1)
    
        dv_qcfile_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='qc',source_file=bold_file,extension='.tsv',
                 density='91k'),
            name='dv_qcfile_wf', run_without_submitting=True, mem_gb=1)

        dv_sc207ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer207',check_hdr=False,density='91k',
                 extension='.ptseries.nii',source_file=bold_file),
            name='dv_sc207ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc407ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer407',extension='.ptseries.nii',
                 source_file=bold_file,density='91k',check_hdr=False),
            name='dv_sc407ts_wf',  run_without_submitting=True, mem_gb=1)
    
        dv_gs360ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Glasser',density='91k',extension='.ptseries.nii',
                 source_file=bold_file,check_hdr=False),
            name='dv_gs360ts_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Gordon',density='91k',extension='.ptseries.nii',
                 source_file=bold_file,check_hdr=False),
            name='dv_gd333_wf', run_without_submitting=True, mem_gb=1)

        dv_sc207fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer207',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc207fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc407fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer407',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc407fc_wf', run_without_submitting=True, mem_gb=1)
    
        dv_gs360fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Glasser',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_gs333_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,extension='.pconn.nii', 
                 check_hdr=False,dismiss_entities=['desc'],atlas='Gordon',density='91k',source_file=bold_file),
            name='dv_gd333fc_wf', run_without_submitting=True, mem_gb=1)
        
        dv_reholh_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,check_hdr=False,
                 dismiss_entities=['desc'],desc='reho',density='32k',hemi='L',extension='.func.gii',
                 source_file=bold_file),
            name='dv_reholh_wf', run_without_submitting=True, mem_gb=1)

        dv_rehorh_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,check_hdr=False,
                 dismiss_entities=['desc'],desc='reho',density='32k',hemi='R',extension='.func.gii',
                 source_file=bold_file),
            name='dv_rehorh_wf', run_without_submitting=True, mem_gb=1)
        
        dv_fd_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],desc='framewisedisplacement',extension='.tsv',
                 source_file=bold_file),
                 name='dv_fd_wf', run_without_submitting=True, mem_gb=1)
        
        workflow.connect([
         (inputnode,dv_cleandata_wf,[('processed_bold','in_file')]),
         (inputnode,dv_alff_wf,[('alff_out','in_file')]),
         (inputnode,dv_qcfile_wf,[('qc_file','in_file')]),
         (inputnode,dv_sc207ts_wf,[('sc207_ts','in_file')]),
         (inputnode,dv_sc407ts_wf,[('sc407_ts','in_file')]),
         (inputnode,dv_gs360ts_wf,[('gs360_ts','in_file')]),
         (inputnode,dv_gd333ts_wf,[('gd333_ts','in_file')]),
         (inputnode,dv_sc207fc_wf,[('sc207_fc','in_file')]),
         (inputnode,dv_sc407fc_wf,[('sc407_fc','in_file')]),
         (inputnode,dv_gs360fc_wf,[('gs360_fc','in_file')]),
         (inputnode,dv_gd333fc_wf,[('gd333_fc','in_file')]), 
         (inputnode,dv_reholh_wf,[('reho_lh','in_file')]), 
         (inputnode,dv_rehorh_wf,[('reho_rh','in_file')]), 
         (inputnode,dv_fd_wf,[('fd','in_file')]),  
           ])
        
        if smoothing:
            dv_smoothcleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], density='91k',
                 desc='clean_smooth',source_file=bold_file,extension='.ptseries',check_hdr=False),
            name='dv_smoothcleandata_wf', run_without_submitting=True, mem_gb=2)

            dv_smoothalff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], desc='alff_smooth',
                 density='91k',source_file=bold_file,extension='.dscalar.nii',check_hdr=False),
            name='dv_smoothalff_wf', run_without_submitting=True, mem_gb=1)
  
            workflow.connect([
                (inputnode,dv_smoothcleandata_wf,[('smoothed_bold','in_file')]),
                (inputnode,dv_smoothalff_wf,[('smoothed_alff','in_file')]),
            ])
    
    return workflow


