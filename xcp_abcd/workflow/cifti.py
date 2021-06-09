# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing the bold
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_ciftipostprocess_wf

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
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ..interfaces import computeqcplot
from  ..utils import bid_derivative
from ..interfaces import  FunctionalSummary,ciftidespike
from  ..workflow import (init_cifti_conts_wf,
    init_post_process_wf,
    init_compute_alff_wf,
    init_surface_reho_wf)


from .outputs import init_writederivatives_wf

LOGGER = logging.getLogger('nipype.workflow')


def init_ciftipostprocess_wf(
    cifti_file,
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
    output_dir,
    custom_conf,
    omp_nthreads,
    dummytime,
    fd_thresh,
    despike,
    num_cifti,
    layout=None,
    name='cifti_process_wf'):

    """
    This workflow organizes cifti processing workflow.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflow.cifti import init_ciftipostprocess_wf
            wf = init_ciftipostprocess_wf(
                bold_file,
                lower_bpf,
                upper_bpf,
                contigvol,
                bpf_order,
                motion_filter_order,
                motion_filter_type,
                band_stop_min,
                band_stop_max,
                despike,
                smoothing,
                head_radius,
                params,
                custom_conf,
                omp_nthreads,
                dummytime,
                output_dir,
                fd_thresh,
                num_cifti,
                template='MNI152NLin2009cAsym',
                layout=None,
                name='cifti_postprocess_wf',
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
    scrub: bool
        remove the censored volumes
    dummytime: float
        the first vols in seconds to be removed before postprocessing

    Inputs
    ------
    cifti_file
        CIFTI file
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
    reho_lh
        reho left hemisphere
    reho_rh
        reho right hemisphere
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
For each of the {num_cifti} CIFTI runs found per subject (across all
tasks and sessions), the following postprocessing was performed:
""".format(num_cifti=num_cifti)


    inputnode = pe.Node(niu.IdentityInterface(
        fields=['cifti_file','custom_conf']),
        name='inputnode')

    inputnode.inputs.cifti_file = cifti_file
    inputnode.inputs.custom_conf = str(custom_conf)


    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','alff_out','smoothed_alff',
                'reho_lh','reho_rh','sc217_ts', 'sc217_fc','sc417_ts','sc417_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc','qc_file','fd']),
        name='outputnode')

    TR = layout.get_tr(cifti_file)



    mem_gbx = _create_mem_gb(cifti_file)

    clean_data_wf = init_post_process_wf(mem_gb=mem_gbx['timeseries'], TR=TR,
                    head_radius=head_radius,lower_bpf=lower_bpf,upper_bpf=upper_bpf,
                    bpf_order=bpf_order,band_stop_max=band_stop_max,band_stop_min=band_stop_min,
                    motion_filter_order=motion_filter_order,motion_filter_type=motion_filter_type,
                    smoothing=smoothing,params=params,contigvol=contigvol,
                    dummytime=dummytime,fd_thresh=fd_thresh,cifti=True,bold_file=cifti_file,
                   name='clean_data_wf')

    cifti_conts_wf = init_cifti_conts_wf(mem_gb=mem_gbx['timeseries'],
                      name='cifti_ts_con_wf')

    alff_compute_wf = init_compute_alff_wf(mem_gb=mem_gbx['timeseries'],TR=TR,
                   lowpass=lower_bpf,highpass=upper_bpf,smoothing=smoothing,cifti=True,
                    name="compute_alff_wf" )

    reho_compute_wf = init_surface_reho_wf(mem_gb=mem_gbx['timeseries'],smoothing=smoothing,
                       name="surface_reho_wf")

    write_derivative_wf = init_writederivatives_wf(mem_gb=2,smoothing=smoothing,bold_file=cifti_file,
                    params=params,cifti=True,output_dir=output_dir,dummytime=dummytime,
                    lowpass=upper_bpf,highpass=lower_bpf,TR=TR,omp_nthreads=omp_nthreads,
                    name="write_derivative_wf",)


    if despike:
        despike_wf = pe.Node(ciftidespike(tr=TR),name="cifti_depike_wf", mem_gb=mem_gbx['timeseries'])
        workflow.connect([
             (inputnode,despike_wf,[('cifti_file','in_file'),]),
             (despike_wf,clean_data_wf,[('des_file','inputnode.bold'),]),

        ])
    else:
        workflow.connect([
        (inputnode,clean_data_wf,[('cifti_file','inputnode.bold'),]),
        ])

    workflow.connect([
            (clean_data_wf, cifti_conts_wf,[('outputnode.processed_bold','inputnode.clean_cifti')]),
            (clean_data_wf, alff_compute_wf,[('outputnode.processed_bold','inputnode.clean_bold')]),
            (clean_data_wf,reho_compute_wf,[('outputnode.processed_bold','inputnode.clean_bold')]),

            (clean_data_wf,outputnode,[('outputnode.processed_bold','processed_bold'),
                                       ('outputnode.fd','fd'),

                                  ('outputnode.smoothed_bold','smoothed_bold') ]),

            (alff_compute_wf,outputnode,[('outputnode.alff_out','alff_out')]),
            (reho_compute_wf,outputnode,[('outputnode.lh_reho','reho_lh'),('outputnode.rh_reho','reho_rh')]),

            (cifti_conts_wf,outputnode,[('outputnode.sc217_ts','sc217_ts' ),('outputnode.sc217_fc','sc217_fc'),
                        ('outputnode.sc417_ts','sc417_ts'),('outputnode.sc417_fc','sc417_fc'),
                        ('outputnode.gs360_ts','gs360_ts'),('outputnode.gs360_fc','gs360_fc'),
                        ('outputnode.gd333_ts','gd333_ts'),('outputnode.gd333_fc','gd333_fc')]),


      ])
    if custom_conf:
        workflow.connect([
         (inputnode,clean_data_wf,[('custom_conf','inputnode.custom_conf')]),
        ])

    qcreport = pe.Node(computeqcplot(TR=TR,bold_file=cifti_file,dummytime=dummytime,
                       head_radius=head_radius), name="qc_report")
    workflow.connect([
        (clean_data_wf,qcreport,[('outputnode.processed_bold','cleaned_file'),
                            ('outputnode.tmask','tmask')]),
        (qcreport,outputnode,[('qc_file','qc_file')]),
           ])

    workflow.connect([
        (clean_data_wf, write_derivative_wf,[('outputnode.processed_bold','inputnode.processed_bold'),
                                    ('outputnode.fd','inputnode.fd'),
                                   ('outputnode.smoothed_bold','inputnode.smoothed_bold')]),
        (alff_compute_wf,write_derivative_wf,[('outputnode.alff_out','inputnode.alff_out'),
                                      ('outputnode.smoothed_alff','inputnode.smoothed_alff')]),
        (reho_compute_wf,write_derivative_wf,[('outputnode.rh_reho','inputnode.reho_rh'),
                                     ('outputnode.lh_reho','inputnode.reho_lh')]),
        (cifti_conts_wf,write_derivative_wf,[('outputnode.sc217_ts','inputnode.sc217_ts' ),
                                ('outputnode.sc217_fc','inputnode.sc217_fc'),
                                ('outputnode.sc417_ts','inputnode.sc417_ts'),
                                ('outputnode.sc417_fc','inputnode.sc417_fc'),
                                ('outputnode.gs360_ts','inputnode.gs360_ts'),
                                ('outputnode.gs360_fc','inputnode.gs360_fc'),
                                ('outputnode.gd333_ts','inputnode.gd333_ts'),
                                ('outputnode.gd333_fc','inputnode.gd333_fc')]),
        (qcreport,write_derivative_wf,[('qc_file','inputnode.qc_file')]),

         ])
    functional_qc = pe.Node(FunctionalSummary(bold_file=cifti_file,tr=TR),
                name='qcsummary', run_without_submitting=True)
    ds_report_qualitycontrol = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='qualitycontrol',source_file=cifti_file, datatype="figures"),
                  name='ds_report_qualitycontrol', run_without_submitting=True)
    ds_report_preprocessing = pe.Node(
        DerivativesDataSink(base_directory=output_dir, source_file=cifti_file, desc='preprocessing', datatype="figures"),
                  name='ds_report_preprocessing', run_without_submitting=True)
    ds_report_postprocessing = pe.Node(
        DerivativesDataSink(base_directory=output_dir,source_file=cifti_file, desc='postprocessing', datatype="figures"),
                  name='ds_report_postprocessing', run_without_submitting=True)

    ds_report_connectivity = pe.Node(
        DerivativesDataSink(base_directory=output_dir,source_file=cifti_file, desc='connectvityplot', datatype="figures"),
                  name='ds_report_connectivity', run_without_submitting=True)

    workflow.connect([
        (qcreport,ds_report_preprocessing,[('raw_qcplot','in_file')]),
        (qcreport,ds_report_postprocessing ,[('clean_qcplot','in_file')]),
        (qcreport,functional_qc,[('qc_file','qc_file')]),
        (functional_qc,ds_report_qualitycontrol,[('out_report','in_file')]),
        (cifti_conts_wf,ds_report_connectivity,[('outputnode.connectplot',"in_file")]),

     ])
    return workflow



def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    #bold_tlen = nb.load(bold_fname).shape[-1]
    bold_tlen = 100
    mem_gbz = {
        'derivative': bold_size_gb,
        'resampled': bold_size_gb * 4,
        'timeseries': 6 + bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    return mem_gbz


class DerivativesDataSink(bid_derivative):
    out_path_base = 'xcp_abcd'
