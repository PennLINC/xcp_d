# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import os 
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from . import DerivativesDataSink
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

def init_writederivatives_wf(
     bold_file,
     lowpass,
     highpass,
     smoothing,
     params,
     omp_nthreads,
     scrub,
     surface,
     dummytime,
     output_dir,
     TR,
     name='write_derivatives_wf',
     ):
    
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
            fields=['processed_bold', 'smoothed_bold','alff_out','smoothed_alff', 
                'reho_out','sc207_ts', 'sc207_fc','sc407_ts','sc407_fc','reho_lh','reho_rh',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc','qc_file']), name='inputnode')
    if tmask:
        nvolcensored = np.sum(np.loadtxt(tmask))
    else:
        nvolcensored = 0
    
    cleandata_dict= { 'RepetitionTime': TR, 'Freq Band': [highpass,lowpass],'nuissance parameters': params,  
                    'dummy vols' :  np.int(dummytime/TR),'nvolcensored':nvolcensored}
    smoothed_dict = { 'FWHM': smoothing }


    if not surface:
        dv_cleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=cleandata_dict,dismiss_entities=['desc'], desc='clean',source_file=bold_file),
            name='dv_cleandata_wf', run_without_submitting=True, mem_gb=2)
            
        dv_alff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='alff',source_file=bold_file),
            name='dv_alff_wf', run_without_submitting=True, mem_gb=1)
    
        dv_qcfile_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='qc',source_file=bold_file),
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

        dv_reho_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='reho',source_file=bold_file),
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

    if surface:
        dv_cleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=cleandata_dict,dismiss_entities=['desc'], desc='clean',
                 source_file=bold_file,density='91k'),
            name='dv_cleandata_wf', run_without_submitting=True, mem_gb=2)
            
        dv_alff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='alff',density='91k',
                 source_file=bold_file),
            name='dv_alff_wf', run_without_submitting=True, mem_gb=1)
    
        dv_qcfile_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='qc',source_file=bold_file,
                 density='91k'),
            name='dv_qcfile_wf', run_without_submitting=True, mem_gb=1)

        dv_sc207ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer207',desc='timeseries',source_file=bold_file),
            name='dv_sc207ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc407ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer407',
                 source_file=bold_file,density='91k'),
            name='dv_sc407ts_wf',  run_without_submitting=True, mem_gb=1)
    
        dv_gs360ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Glasser',density='91k',
                 source_file=bold_file),
            name='dv_gs360ts_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Gordon',density='91k',
                 source_file=bold_file),
            name='dv_gd333_wf', run_without_submitting=True, mem_gb=1)

        dv_sc207fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer207',
                 density='91k',source_file=bold_file),
            name='dv_sc207fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc407fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Schaefer407',
                 density='91k',source_file=bold_file),
            name='dv_sc407fc_wf', run_without_submitting=True, mem_gb=1)
    
        dv_gs360fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Glaseer',desc='connectivity',
                 density='91k',source_file=bold_file),
            name='dv_gs333_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],atlas='Gordon',density='91k',source_file=bold_file),
            name='dv_gd333fc_wf', run_without_submitting=True, mem_gb=1)
        
        dv_reholh_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='reho',density='32k',hemi='L',
                 source_file=bold_file),
            name='dv_reholh_wf', run_without_submitting=True, mem_gb=1)

        dv_rehorh_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 dismiss_entities=['desc'],desc='reho',density='32k',hemi='R',
                 source_file=bold_file),
            name='dv_rehorh_wf', run_without_submitting=True, mem_gb=1)
        
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
         (inputnode,dv_reholh_wf,[('reho_lh','in_file')]), 
         (inputnode,dv_rehorh_wf,[('reho_rh','in_file')]),   
           ])
        
        if smoothing:
            dv_smoothcleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], density='91k',
                 desc='clean_smooth',source_file=bold_file),
            name='dv_smoothcleandata_wf', run_without_submitting=True, mem_gb=2)

            dv_smoothalff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir, 
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], desc='alff_smooth',
                 density='91k',source_file=bold_file),
            name='dv_smoothalff_wf', run_without_submitting=True, mem_gb=1)
  
            workflow.connect([
                (inputnode,dv_smoothcleandata_wf,[('smoothed_bold','in_file')]),
                (inputnode,dv_smoothalff_wf,[('smoothed_alff','in_file')]),
            ])
    
    return workflow