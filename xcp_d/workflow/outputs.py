# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
import os
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ..utils import bid_derivative

class DerivativesDataSink(bid_derivative):
    out_path_base = 'xcp_d'


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
            from xcp_d.workflows import init_writederivatives_wf
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
                'sc117_ts', 'sc117_fc','sc217_ts', 'sc217_fc','sc317_ts', 'sc317_fc',
                'sc417_ts','sc417_fc',
                'sc517_ts', 'sc517_fc','sc617_ts', 'sc617_fc','sc717_ts','sc717_fc',
                'sc817_ts', 'sc817_fc','sc917_ts', 'sc917_fc','sc1017_ts','sc1017_fc',
                'reho_lh','reho_rh','reho_out',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc','ts50_ts', 'ts50_fc','qc_file','fd']), name='inputnode')

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
                 compression=True,extension='.csv'),
            name='dv_qcfile_wf', run_without_submitting=True, mem_gb=1)

        dv_sc117ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer117',desc='timeseries',source_file=bold_file),
            name='dv_sc117ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc217ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer217',desc='timeseries',source_file=bold_file),
            name='dv_sc217ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc317ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer317',desc='timeseries',source_file=bold_file),
            name='dv_sc317ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc417ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer417',desc='timeseries',source_file=bold_file),
            name='dv_sc417ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc517ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer517',desc='timeseries',source_file=bold_file),
            name='dv_sc517ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc617ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer617',desc='timeseries',source_file=bold_file),
            name='dv_sc617ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc717ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer717',desc='timeseries',source_file=bold_file),
            name='dv_sc717ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc817ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer817',desc='timeseries',source_file=bold_file),
            name='dv_sc817ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc917ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer917',desc='timeseries',source_file=bold_file),
            name='dv_sc917ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc1017ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer1017',desc='timeseries',source_file=bold_file),
            name='dv_sc1017ts_wf', run_without_submitting=True, mem_gb=1)

        dv_gs360ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Glasser',desc='timeseries',source_file=bold_file),
            name='dv_gs360ts_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Gordon',desc='timeseries',source_file=bold_file),
            name='dv_gd333_wf', run_without_submitting=True, mem_gb=1)

        dv_ts50ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='subcortical',desc='timeseries',source_file=bold_file),
            name='dv_ts50_wf', run_without_submitting=True, mem_gb=1)

        dv_sc117fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer117',desc='connectivity',source_file=bold_file),
            name='dv_sc117fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc217fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer217',desc='connectivity',source_file=bold_file),
            name='dv_sc217fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc317fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer317',desc='connectivity',source_file=bold_file),
            name='dv_sc317fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc417fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer417',desc='connectivity',source_file=bold_file),
            name='dv_sc417fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc517fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer517',desc='connectivity',source_file=bold_file),
            name='dv_sc517fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc617fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer617',desc='connectivity',source_file=bold_file),
            name='dv_sc617fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc717fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer717',desc='connectivity',source_file=bold_file),
            name='dv_sc717fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc817fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer817',desc='connectivity',source_file=bold_file),
            name='dv_sc817fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc917fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer917',desc='connectivity',source_file=bold_file),
            name='dv_sc917fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc1017fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Schaefer1017',desc='connectivity',source_file=bold_file),
            name='dv_sc1017fc_wf', run_without_submitting=True, mem_gb=1)

        dv_gs360fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Glaseer',desc='connectivity',source_file=bold_file),
            name='dv_gs333_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='Gordon',desc='connectivity',source_file=bold_file),
            name='dv_gd333fc_wf', run_without_submitting=True, mem_gb=1)
        
        dv_ts50fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],atlas='subcortical',desc='connectivity',source_file=bold_file),
            name='dv_ts50fc_wf', run_without_submitting=True, mem_gb=1)

        dv_reho_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,extension='.nii.gz',
                 dismiss_entities=['desc'],compression=True,desc='reho',source_file=bold_file),
            name='dv_reho_wf', run_without_submitting=True, mem_gb=1)

        dv_fd_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc'],desc='framewisedisplacement',extension='.tsv',
                 source_file=bold_file),
                 name='dv_fd_wf', run_without_submitting=True, mem_gb=1)

        workflow.connect([
         (inputnode,dv_cleandata_wf,[('processed_bold','in_file')]),
         (inputnode,dv_alff_wf,[('alff_out','in_file')]),
         (inputnode,dv_reho_wf,[('reho_out','in_file')]),
         (inputnode,dv_qcfile_wf,[('qc_file','in_file')]),
         (inputnode,dv_sc117ts_wf,[('sc117_ts','in_file')]),
         (inputnode,dv_sc217ts_wf,[('sc217_ts','in_file')]),
         (inputnode,dv_sc317ts_wf,[('sc317_ts','in_file')]),
         (inputnode,dv_sc417ts_wf,[('sc417_ts','in_file')]),
         (inputnode,dv_sc517ts_wf,[('sc517_ts','in_file')]),
         (inputnode,dv_sc617ts_wf,[('sc617_ts','in_file')]),
         (inputnode,dv_sc717ts_wf,[('sc717_ts','in_file')]),
         (inputnode,dv_sc817ts_wf,[('sc817_ts','in_file')]),
         (inputnode,dv_sc917ts_wf,[('sc917_ts','in_file')]),
         (inputnode,dv_sc1017ts_wf,[('sc1017_ts','in_file')]),
         (inputnode,dv_gs360ts_wf,[('gs360_ts','in_file')]),
         (inputnode,dv_gd333ts_wf,[('gd333_ts','in_file')]),
         (inputnode,dv_ts50ts_wf,[('ts50_ts','in_file')]),
         (inputnode,dv_sc117fc_wf,[('sc117_fc','in_file')]),
         (inputnode,dv_sc217fc_wf,[('sc217_fc','in_file')]),
         (inputnode,dv_sc317fc_wf,[('sc317_fc','in_file')]),
         (inputnode,dv_sc417fc_wf,[('sc417_fc','in_file')]),
         (inputnode,dv_sc517fc_wf,[('sc517_fc','in_file')]),
         (inputnode,dv_sc617fc_wf,[('sc617_fc','in_file')]),
         (inputnode,dv_sc717fc_wf,[('sc717_fc','in_file')]),
         (inputnode,dv_sc817fc_wf,[('sc817_fc','in_file')]),
         (inputnode,dv_sc917fc_wf,[('sc917_fc','in_file')]),
         (inputnode,dv_sc1017fc_wf,[('sc1017_fc','in_file')]),
         (inputnode,dv_gs360fc_wf,[('gs360_fc','in_file')]),
         (inputnode,dv_gd333fc_wf,[('gd333_fc','in_file')]),
         (inputnode,dv_ts50fc_wf,[('ts50_fc','in_file')]),
         (inputnode,dv_fd_wf,[('fd','in_file')]),
           ])
        if smoothing:
            dv_smoothcleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], desc='residual_smooth',source_file=bold_file,
                 extension='.nii.gz',compression=True),
            name='dv_smoothcleandata_wf', run_without_submitting=True, mem_gb=2)

            dv_smoothalff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 meta_dict=smoothed_dict,dismiss_entities=['desc'], desc='alff_smooth',source_file=bold_file
                 ,extension='.nii.gz',compression=True),
            name='dv_smoothalff_wf', run_without_submitting=True, mem_gb=1)

            workflow.connect([
                (inputnode,dv_smoothcleandata_wf,[('smoothed_bold','in_file')]),
                (inputnode,dv_smoothalff_wf,[('smoothed_alff','in_file')]),
            ])

    if cifti:
        dv_cleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 meta_dict=cleandata_dict,dismiss_entities=['desc','den'], desc='residual',
                 source_file=bold_file,density='91k',extension='.dtseries.nii'),
            name='dv_cleandata_wf', run_without_submitting=True, mem_gb=2)

        dv_alff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],desc='alff',density='91k',extension='.dtseries.nii',
                 source_file=bold_file,check_hdr=False),
            name='dv_alff_wf', run_without_submitting=True, mem_gb=1)

        dv_qcfile_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],desc='qc',source_file=bold_file,extension='.csv',
                 density='91k'),
            name='dv_qcfile_wf', run_without_submitting=True, mem_gb=1)

        dv_sc117ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer117',check_hdr=False,density='91k',
                 extension='.ptseries.nii',source_file=bold_file),
            name='dv_sc117ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc217ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer217',check_hdr=False,density='91k',
                 extension='.ptseries.nii',source_file=bold_file),
            name='dv_sc217ts_wf', run_without_submitting=True, mem_gb=1)
        
        dv_sc317ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer317',check_hdr=False,density='91k',
                 extension='.ptseries.nii',source_file=bold_file),
            name='dv_sc317ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc417ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer417',extension='.ptseries.nii',
                 source_file=bold_file,density='91k',check_hdr=False),
            name='dv_sc417ts_wf',  run_without_submitting=True, mem_gb=1)

        dv_sc517ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer517',extension='.ptseries.nii',
                 source_file=bold_file,density='91k',check_hdr=False),
            name='dv_sc517ts_wf',  run_without_submitting=True, mem_gb=1)

        dv_sc617ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer617',check_hdr=False,density='91k',
                 extension='.ptseries.nii',source_file=bold_file),
            name='dv_sc617ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc717ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer717',check_hdr=False,density='91k',
                 extension='.ptseries.nii',source_file=bold_file),
            name='dv_sc717ts_wf', run_without_submitting=True, mem_gb=1)
        
        dv_sc817ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer817',check_hdr=False,density='91k',
                 extension='.ptseries.nii',source_file=bold_file),
            name='dv_sc817ts_wf', run_without_submitting=True, mem_gb=1)

        dv_sc917ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer917',extension='.ptseries.nii',
                 source_file=bold_file,density='91k',check_hdr=False),
            name='dv_sc917ts_wf',  run_without_submitting=True, mem_gb=1)

        dv_sc1017ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer1017',extension='.ptseries.nii',
                 source_file=bold_file,density='91k',check_hdr=False),
            name='dv_sc1017ts_wf',  run_without_submitting=True, mem_gb=1)

        dv_gs360ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Glasser',density='91k',extension='.ptseries.nii',
                 source_file=bold_file,check_hdr=False),
            name='dv_gs360ts_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Gordon',density='91k',extension='.ptseries.nii',
                 source_file=bold_file,check_hdr=False),
            name='dv_gd333_wf', run_without_submitting=True, mem_gb=1)
        dv_ts50ts_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='subcortical',density='91k',extension='.ptseries.nii',
                 source_file=bold_file,check_hdr=False),
            name='dv_ts50_wf', run_without_submitting=True, mem_gb=1)

        dv_sc117fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer117',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc117fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc217fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer217',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc217fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc317fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer317',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc317fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc417fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer417',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc417fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc517fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer517',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc517fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc617fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer617',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc617fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc717fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer717',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc717fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc817fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer817',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc817fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc917fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer917',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc917fc_wf', run_without_submitting=True, mem_gb=1)

        dv_sc1017fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Schaefer1017',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_sc1017fc_wf', run_without_submitting=True, mem_gb=1)

        dv_gs360fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],atlas='Glasser',extension='.pconn.nii',
                 density='91k',source_file=bold_file,check_hdr=False),
            name='dv_gs333_wf', run_without_submitting=True, mem_gb=1)

        dv_gd333fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,extension='.pconn.nii',
                 check_hdr=False,dismiss_entities=['desc','den'],atlas='Gordon',density='91k',source_file=bold_file),
            name='dv_gd333fc_wf', run_without_submitting=True, mem_gb=1)
           
        dv_ts50fc_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,extension='.pconn.nii',
                 check_hdr=False,dismiss_entities=['desc','den'],atlas='subcortical',density='91k',source_file=bold_file),
            name='dv_ts50fc_wf', run_without_submitting=True, mem_gb=1)

        dv_reholh_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,check_hdr=False,
                 dismiss_entities=['desc','den'],desc='reho',density='32k',hemi='L',extension='.func.gii',
                 source_file=bold_file),
            name='dv_reholh_wf', run_without_submitting=True, mem_gb=1)

        dv_rehorh_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,check_hdr=False,
                 dismiss_entities=['desc','den'],desc='reho',density='32k',hemi='R',extension='.func.gii',
                 source_file=bold_file),
            name='dv_rehorh_wf', run_without_submitting=True, mem_gb=1)

        dv_fd_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 dismiss_entities=['desc','den'],desc='framewisedisplacement',extension='.tsv',
                 source_file=bold_file),
                 name='dv_fd_wf', run_without_submitting=True, mem_gb=1)

        workflow.connect([
         (inputnode,dv_cleandata_wf,[('processed_bold','in_file')]),
         (inputnode,dv_alff_wf,[('alff_out','in_file')]),
         (inputnode,dv_qcfile_wf,[('qc_file','in_file')]),
         (inputnode,dv_sc117ts_wf,[('sc117_ts','in_file')]),
         (inputnode,dv_sc217ts_wf,[('sc217_ts','in_file')]),
         (inputnode,dv_sc317ts_wf,[('sc317_ts','in_file')]),
         (inputnode,dv_sc417ts_wf,[('sc417_ts','in_file')]),
         (inputnode,dv_sc517ts_wf,[('sc517_ts','in_file')]),
         (inputnode,dv_sc617ts_wf,[('sc617_ts','in_file')]),
         (inputnode,dv_sc717ts_wf,[('sc717_ts','in_file')]),
         (inputnode,dv_sc817ts_wf,[('sc817_ts','in_file')]),
         (inputnode,dv_sc917ts_wf,[('sc917_ts','in_file')]),
         (inputnode,dv_sc1017ts_wf,[('sc1017_ts','in_file')]),
         (inputnode,dv_gs360ts_wf,[('gs360_ts','in_file')]),
         (inputnode,dv_gd333ts_wf,[('gd333_ts','in_file')]),
         (inputnode,dv_ts50ts_wf,[('ts50_ts','in_file')]),
         (inputnode,dv_sc117fc_wf,[('sc117_fc','in_file')]),
         (inputnode,dv_sc217fc_wf,[('sc217_fc','in_file')]),
         (inputnode,dv_sc317fc_wf,[('sc317_fc','in_file')]),
         (inputnode,dv_sc417fc_wf,[('sc417_fc','in_file')]),
         (inputnode,dv_sc517fc_wf,[('sc517_fc','in_file')]),
         (inputnode,dv_sc617fc_wf,[('sc617_fc','in_file')]),
         (inputnode,dv_sc717fc_wf,[('sc717_fc','in_file')]),
         (inputnode,dv_sc817fc_wf,[('sc817_fc','in_file')]),
         (inputnode,dv_sc917fc_wf,[('sc917_fc','in_file')]),
         (inputnode,dv_sc1017fc_wf,[('sc1017_fc','in_file')]),
         (inputnode,dv_gs360fc_wf,[('gs360_fc','in_file')]),
         (inputnode,dv_gd333fc_wf,[('gd333_fc','in_file')]),
         (inputnode,dv_ts50fc_wf,[('ts50_fc','in_file')]),
         (inputnode,dv_reholh_wf,[('reho_lh','in_file')]),
         (inputnode,dv_rehorh_wf,[('reho_rh','in_file')]),
         (inputnode,dv_fd_wf,[('fd','in_file')]),
           ])

        if smoothing:
            dv_smoothcleandata_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 meta_dict=smoothed_dict,dismiss_entities=['desc','den'], density='91k',
                 desc='residual_smooth',source_file=bold_file,extension='.dtseries.nii',check_hdr=False),
            name='dv_smoothcleandata_wf', run_without_submitting=True, mem_gb=2)

            dv_smoothalff_wf = pe.Node(DerivativesDataSink(base_directory=output_dir,
                 meta_dict=smoothed_dict,dismiss_entities=['desc','den'], desc='alff_smooth',
                 density='91k',source_file=bold_file,extension='.dtseries.nii',check_hdr=False),
            name='dv_smoothalff_wf', run_without_submitting=True, mem_gb=1)

            workflow.connect([
                (inputnode,dv_smoothcleandata_wf,[('smoothed_bold','in_file')]),
                (inputnode,dv_smoothalff_wf,[('smoothed_alff','in_file')]),
            ])

    return workflow
