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
from ..interfaces import computeqcplot
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from  ..utils import bid_derivative
from ..interfaces import  FunctionalSummary
from templateflow.api import get as get_template
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from nipype.interfaces.afni import Despike

from  ..workflow import (init_fcon_ts_wf,
    init_post_process_wf,
    init_compute_alff_wf,
    init_3d_reho_wf)
from .outputs import init_writederivatives_wf

LOGGER = logging.getLogger('nipype.workflow')

def init_boldpostprocess_wf(
     lower_bpf,
     upper_bpf,
     contigvol,
     bpf_order,
     motion_filter_order,
     motion_filter_type,
     band_stop_min,
     band_stop_max,
     smoothing,
     bold_file,
     head_radius,
     params,
     custom_conf,
     omp_nthreads,
     dummytime,
     output_dir,
     fd_thresh,
     num_bold,
     mni_to_t1w,
     despike,
     brain_template='MNI152NLin2009cAsym',
     layout=None,
     name='bold_postprocess_wf',
      ):

    """
    This workflow organizes bold processing workflow.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflow.bold import init_boldpostprocess_wf
            wf = init_boldpostprocess_wf(
                bold_file,
                lower_bpf,
                upper_bpf,
                contigvol,
                bpf_order,
                motion_filter_order,
                motion_filter_type,
                band_stop_min,
                band_stop_max,
                smoothing,
                head_radius,
                params,
                custom_conf,
                omp_nthreads,
                dummytime,
                output_dir,
                fd_thresh,
                num_bold,
                template='MNI152NLin2009cAsym',
                layout=None,
                name='bold_postprocess_wf',
             )
    Parameters
    ----------
    bold_file: str
        bold file for post processing
    lower_bpf : float
        Lower band pass filter
    upper_bpf : float
        Upper band pass filter
    layout : BIDSLayout object
        BIDS dataset layout
    contigvol: int
        number of contigious volumes
    despike: bool
        afni depsike
    motion_filter_order: int
        respiratory motion filter order
    motion_filter_type: str
        respiratory motion filter type: lp or notch
    band_stop_min: float
        respiratory minimum frequency in breathe per minutes(bpm)
    band_stop_max,: float
        respiratory maximum frequency in breathe per minutes(bpm)
    layout : BIDSLayout object
        BIDS dataset layout
    omp_nthreads : int
        Maximum number of threads an individual process may use
    output_dir : str
        Directory in which to save xcp_abcd output
    fd_thresh
        Criterion for flagging framewise displacement outliers
    head_radius : float
        radius of the head for FD computation
    params: str
        nuissance regressors to be selected from fmriprep regressors
    smoothing: float
        smooth the derivatives output with kernel size (fwhm)
    custom_conf: str
        path to cusrtom nuissance regressors
    dummytime: float
        the first vols in seconds to be removed before postprocessing

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    mni_to_t1w
        MNI to T1W ants Transformation file/h5
    ref_file
        Bold reference file from fmriprep
    bold_mask
        bold_mask from fmriprep
    cutstom_conf
        custom regressors

    Outputs
    -------
    processed_bold
        clean bold after regression and filtering
    smoothed_bold
        smoothed clean bold
    alff_out
        alff niifti
    smoothed_alff
        smoothed alff
    reho_out
        reho output computed by afni.3dreho
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


    TR = layout.get_tr(bold_file)

    workflow = Workflow(name=name)

    workflow.__desc__ = """
For each of the {num_bold} BOLD runs found per subject (across all
tasks and sessions), the following postprocessing was performed:
""".format(num_bold=num_bold)


    # get reference and mask
    mask_file,ref_file = _get_ref_mask(fname=bold_file)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_file','ref_file','bold_mask','cutstom_conf','mni_to_t1w']),
        name='inputnode')

    inputnode.inputs.bold_file = str(bold_file)
    inputnode.inputs.ref_file = str(ref_file)
    inputnode.inputs.bold_mask = str(mask_file)
    inputnode.inputs.custom_conf = str(custom_conf)


    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','alff_out','smoothed_alff',
                'reho_out','sc217_ts', 'sc217_fc','sc417_ts','sc417_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc','qc_file','fd']),
        name='outputnode')


    # get the mem_bg size for each workflow

    mem_gbx = _create_mem_gb(bold_file)
    clean_data_wf = init_post_process_wf(mem_gb=mem_gbx['timeseries'], TR=TR, bold_file=bold_file,
                    head_radius=head_radius,lower_bpf=lower_bpf,upper_bpf=upper_bpf,
                    bpf_order=bpf_order,band_stop_max=band_stop_max,band_stop_min=band_stop_min,
                    motion_filter_order=motion_filter_order,motion_filter_type=motion_filter_type,
                    smoothing=smoothing,params=params,contigvol=contigvol,
                    dummytime=dummytime,fd_thresh=fd_thresh,
                    name='clean_data_wf')


    fcon_ts_wf = init_fcon_ts_wf(mem_gb=mem_gbx['timeseries'],mni_to_t1w=mni_to_t1w,
                 t1w_to_native=_t12native(bold_file),bold_file=bold_file,
                 brain_template=brain_template,name="fcons_ts_wf")

    alff_compute_wf = init_compute_alff_wf(mem_gb=mem_gbx['timeseries'], TR=TR,
                   lowpass=upper_bpf,highpass=lower_bpf,smoothing=smoothing, cifti=False,
                    name="compute_alff_wf" )

    reho_compute_wf = init_3d_reho_wf(mem_gb=mem_gbx['timeseries'],smoothing=smoothing,
                       name="afni_reho_wf")

    write_derivative_wf = init_writederivatives_wf(smoothing=smoothing,bold_file=bold_file,
                    params=params,cifti=None,output_dir=output_dir,dummytime=dummytime,
                    lowpass=upper_bpf,highpass=lower_bpf,TR=TR,omp_nthreads=omp_nthreads,
                    name="write_derivative_wf")
    if despike:
        despike_wf = pe.Node(Despike(outputtype='NIFTI_GZ',args='-NEW'),name="despike_wf",mem_gb=mem_gbx['timeseries'])

        workflow.connect([
            (inputnode,despike_wf,[('bold_file','in_file')]),
            (despike_wf,clean_data_wf,[('out_file','inputnode.bold')])
            ])
    else:
        workflow.connect([
            (inputnode,clean_data_wf,[('bold_file','inputnode.bold')]),
            ])


    workflow.connect([
        (inputnode,clean_data_wf,[('bold_mask','inputnode.bold_mask')]),

        (inputnode,fcon_ts_wf,[
                               ('ref_file','inputnode.ref_file'),]),
        (clean_data_wf, fcon_ts_wf,[('outputnode.processed_bold','inputnode.clean_bold'),]),

        (inputnode,alff_compute_wf,[('bold_mask','inputnode.bold_mask')]),
        (clean_data_wf, alff_compute_wf,[('outputnode.processed_bold','inputnode.clean_bold')]),

        (inputnode,reho_compute_wf,[('bold_mask','inputnode.bold_mask'),]),
        (clean_data_wf, reho_compute_wf,[('outputnode.processed_bold','inputnode.clean_bold')]),
        (clean_data_wf,outputnode,[('outputnode.processed_bold','processed_bold'),
                                   ('outputnode.smoothed_bold','smoothed_bold'),
                                   ('outputnode.fd','fd')]),
        (alff_compute_wf,outputnode,[('outputnode.alff_out','alff_out'),
                                      ('outputnode.smoothed_alff','smoothed_alff')]),
        (reho_compute_wf,outputnode,[('outputnode.reho_out','reho_out')]),
        (fcon_ts_wf,outputnode,[('outputnode.sc217_ts','sc217_ts' ),('outputnode.sc217_fc','sc217_fc'),
                        ('outputnode.sc417_ts','sc417_ts'),('outputnode.sc417_fc','sc417_fc'),
                        ('outputnode.gs360_ts','gs360_ts'),('outputnode.gs360_fc','gs360_fc'),
                        ('outputnode.gd333_ts','gd333_ts'),('outputnode.gd333_fc','gd333_fc')]),
        ])
    if custom_conf:
        workflow.connect([
         (inputnode,clean_data_wf,[('custom_conf','inputnode.custom_conf')]),
        ])

    qcreport = pe.Node(computeqcplot(TR=TR,bold_file=bold_file,dummytime=dummytime,
                       head_radius=head_radius), name="qc_report")

    file_base = os.path.basename(str(bold_file))

    if brain_template in file_base:
        transformfile = 'identity'
    elif 'T1w' in file_base:
        transformfile = str(mni_to_t1w)
    else:
        transformfile = [str(mni_to_t1w), str(_t12native(bold_file))]

    resample_parc = pe.Node(ApplyTransforms(
        dimension=3,
        input_image=str(get_template(
            'MNI152NLin2009cAsym', resolution=1, desc='carpet',
            suffix='dseg', extension=['.nii', '.nii.gz'])),
        interpolation='MultiLabel',transforms=transformfile),
        name='resample_parc')

    workflow.connect([
        (inputnode,qcreport,[('bold_mask','mask_file')]),
        (clean_data_wf,qcreport,[('outputnode.processed_bold','cleaned_file'),
                            ('outputnode.tmask','tmask')]),
        (inputnode,resample_parc,[('ref_file','reference_image')]),
        (resample_parc,qcreport,[('output_image','seg_file')]),
        (qcreport,outputnode,[('qc_file','qc_file')]),
           ])

    workflow.connect([
        (clean_data_wf, write_derivative_wf,[('outputnode.processed_bold','inputnode.processed_bold'),
                                   ('outputnode.smoothed_bold','inputnode.smoothed_bold'),
                                   ('outputnode.fd','inputnode.fd')]),
        (alff_compute_wf,write_derivative_wf,[('outputnode.alff_out','inputnode.alff_out'),
                                      ('outputnode.smoothed_alff','inputnode.smoothed_alff')]),
        (reho_compute_wf,write_derivative_wf,[('outputnode.reho_out','inputnode.reho_out')]),
        (fcon_ts_wf,write_derivative_wf,[('outputnode.sc217_ts','inputnode.sc217_ts' ),
                                ('outputnode.sc217_fc','inputnode.sc217_fc'),
                                ('outputnode.sc417_ts','inputnode.sc417_ts'),
                                ('outputnode.sc417_fc','inputnode.sc417_fc'),
                                ('outputnode.gs360_ts','inputnode.gs360_ts'),
                                ('outputnode.gs360_fc','inputnode.gs360_fc'),
                                ('outputnode.gd333_ts','inputnode.gd333_ts'),
                                ('outputnode.gd333_fc','inputnode.gd333_fc')]),
        (qcreport,write_derivative_wf,[('qc_file','inputnode.qc_file')]),

         ])
    functional_qc = pe.Node(FunctionalSummary(bold_file=bold_file,tr=TR),
                name='qcsummary', run_without_submitting=True)

    ds_report_qualitycontrol = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='qualitycontrol',source_file=bold_file, datatype="figures"),
                  name='ds_report_qualitycontrol', run_without_submitting=True)

    ds_report_preprocessing = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='preprocessing',source_file=bold_file, datatype="figures"),
                  name='ds_report_preprocessing', run_without_submitting=True)
    ds_report_postprocessing = pe.Node(
        DerivativesDataSink(base_directory=output_dir,source_file=bold_file, desc='postprocessing', datatype="figures"),
                  name='ds_report_postprocessing', run_without_submitting=True)

    ds_report_connectivity = pe.Node(
        DerivativesDataSink(base_directory=output_dir,source_file=bold_file, desc='connectvityplot', datatype="figures"),
                  name='ds_report_connectivity', run_without_submitting=True)

    #ds_report_rehoplot = pe.Node(
        #DerivativesDataSink(base_directory=output_dir,source_file=bold_file, desc='rehoplot', datatype="figures"),
                  #name='ds_report_rehoplot', run_without_submitting=True)

    #ds_report_afniplot = pe.Node(
        #DerivativesDataSink(base_directory=output_dir,source_file=bold_file, desc='afniplot', datatype="figures"),
                  #name='ds_report_afniplot', run_without_submitting=True)

    workflow.connect([
        (qcreport,ds_report_preprocessing,[('raw_qcplot','in_file')]),
        (qcreport,ds_report_postprocessing ,[('clean_qcplot','in_file')]),
        (qcreport,functional_qc,[('qc_file','qc_file')]),
        (functional_qc,ds_report_qualitycontrol,[('out_report','in_file')]),
        (fcon_ts_wf,ds_report_connectivity,[('outputnode.connectplot','in_file')]),
        #(reho_compute_wf,ds_report_rehoplot,[('outputnode.rehohtml','in_file')]),
        #(alff_compute_wf,ds_report_afniplot ,[('outputnode.alffhtml','in_file')]),
    ])

    return workflow

def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gbz = {
        'derivative': bold_size_gb,
        'resampled': bold_size_gb * 4,
        'timeseries': bold_size_gb * (max(10/ 100, 1.0) + 5),
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
    filename = os.path.basename(fname)
    fileup = filename.split('desc-preproc_bold.nii.gz')[0].split('space-')[0]

    t12ref = directx + '/' + fileup + 'from-T1w_to-scanner_mode-image_xfm.txt'

    return t12ref


class DerivativesDataSink(bid_derivative):
    out_path_base = 'xcp_abcd'
