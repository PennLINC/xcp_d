# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing the bold
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_ciftipostprocess_wf

"""
import os
import sklearn
import numpy as np
import nibabel as nb
from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ..interfaces import computeqcplot
from  ..utils import bid_derivative,stringforparams
from ..interfaces import  FunctionalSummary,ciftidespike
from .connectivity import init_cifti_conts_wf
from .restingstate import init_compute_alff_wf,init_surface_reho_wf
from .execsummary import init_execsummary_wf
from ..interfaces import interpolate
from ..interfaces import (ConfoundMatrix,FilteringData,regress)
from .postprocessing import init_censoring_wf, init_resd_smoohthing
from num2words import num2words
from .outputs import init_writederivatives_wf

LOGGER = logging.getLogger('nipype.workflow')


def init_ciftipostprocess_wf(
    cifti_file,
    lower_bpf,
    upper_bpf,
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
    mni_to_t1w,
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

            from xcp_d.workflow.cifti import init_ciftipostprocess_wf
            wf = init_ciftipostprocess_wf(
                bold_file,
                lower_bpf,
                upper_bpf,
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
                name='cifti_postprocess_wf',)

                
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
        Directory in which to save xcp_d output
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
tasks and sessions), the following post-processing was performed:
""".format(num_cifti=num2words(num_cifti))
    
    
    TR = get_ciftiTR(cifti_file)
    if TR is None:
        metadata = layout.get_metadata(cifti_file)
        TR = metadata['RepetitionTime']
    
    #TR = get_ciftiTR(cifti_file=cifti_file)

    if dummytime > 0:
        nvolx = str(np.floor(dummytime / TR))
        workflow.__desc__ = workflow.__desc__ + """ \
before nuisance regression and filtering of the data,  the first {nvol} were discarded,
. Both the nuisance regressors and volumes were demean and detrended. Furthermore, any volumes with framewise-displacement greater than 
{fd_thresh} mm [@power_fd_dvars;@satterthwaite_2013] were  flagged as outliers
 and excluded from nuisance regression.
""".format(nvol=num2words(nvolx),fd_thresh=fd_thresh)

    else:
        workflow.__desc__ = workflow.__desc__ + """ \
before nuissance regression and filtering,both the nuisance regressors and volumes were demean and detrended. Volumes 
 with framewise-displacement greater than {fd_thresh} mm [@power_fd_dvars;@satterthwaite_2013] were  flagged as outlier
 and excluded from nuissance regression.
""".format(fd_thresh=fd_thresh)

    workflow.__desc__ = workflow.__desc__ +  """ \
{regressors} [@mitigating_2018;@benchmarkp;@satterthwaite_2013]. These nuisance regressors were 
regressed from the BOLD data using linear regression - as implemented in Scikit-Learn {sclver} [@scikit-learn].
Residual timeseries from this regression were then band-pass filtered to retain signals within the 
{highpass}-{lowpass} Hz frequency band. 
 """.format(regressors=stringforparams(params=params),sclver=sklearn.__version__,
             lowpass=upper_bpf,highpass=lower_bpf)



    inputnode = pe.Node(niu.IdentityInterface(
        fields=['cifti_file','custom_conf','t1w','t1seg']),
        name='inputnode')

    inputnode.inputs.cifti_file = cifti_file


    outputnode = pe.Node(niu.IdentityInterface(
        fields=['processed_bold', 'smoothed_bold','alff_out','smoothed_alff',
                'reho_lh','reho_rh','sc117_ts', 'sc117_fc','sc217_ts', 'sc217_fc','sc317_ts','sc317_fc',
                'sc517_ts', 'sc517_fc','sc517_ts', 'sc517_fc','sc617_ts','sc617_fc',
                'sc717_ts', 'sc717_fc','sc817_ts', 'sc817_fc','sc917_ts','sc917_fc','sc1017_ts','sc1017_fc',
                'gs360_ts', 'gs360_fc','gd333_ts', 'gd333_fc','ts50_ts','ts50_fc','qc_file','fd']),
        name='outputnode')

    



    mem_gbx = _create_mem_gb(cifti_file)

    

    cifti_conts_wf = init_cifti_conts_wf(mem_gb=mem_gbx['timeseries'],
                      name='cifti_ts_con_wf',omp_nthreads=omp_nthreads)

    alff_compute_wf = init_compute_alff_wf(mem_gb=mem_gbx['timeseries'],TR=TR,
                   lowpass=upper_bpf,highpass=lower_bpf,smoothing=smoothing,cifti=True,
                    name="compute_alff_wf",omp_nthreads=omp_nthreads)

    reho_compute_wf = init_surface_reho_wf(mem_gb=mem_gbx['timeseries'],smoothing=smoothing,
                       name="surface_reho_wf",omp_nthreads=omp_nthreads)

    write_derivative_wf = init_writederivatives_wf(smoothing=smoothing,bold_file=cifti_file,
                    params=params,cifti=True,output_dir=output_dir,dummytime=dummytime,
                    lowpass=upper_bpf,highpass=lower_bpf,TR=TR,omp_nthreads=omp_nthreads,
                    name="write_derivative_wf",)
    
    confoundmat_wf = pe.Node(ConfoundMatrix(head_radius=head_radius, params=params,
                filtertype=motion_filter_type,cutoff=band_stop_max,
                low_freq=band_stop_max,high_freq=band_stop_min,TR=TR,
                filterorder=motion_filter_order),
                  name="ConfoundMatrix_wf", mem_gb=mem_gbx['timeseries'],n_procs=omp_nthreads)

    censorscrub_wf = init_censoring_wf(mem_gb=mem_gbx['timeseries'],custom_conf=custom_conf,TR=TR,head_radius=head_radius,
                dummytime=dummytime,fd_thresh=fd_thresh,name='censoring',omp_nthreads=omp_nthreads)
    
    resdsmoothing_wf = init_resd_smoohthing(mem_gb=mem_gbx['timeseries'],smoothing=smoothing,cifti=True,
                name="resd_smoothing_wf",omp_nthreads=omp_nthreads)
    
    filtering_wf  = pe.Node(FilteringData(tr=TR,lowpass=upper_bpf,highpass=lower_bpf,
                filter_order=bpf_order),
                    name="filtering_wf", mem_gb=mem_gbx['timeseries'],n_procs=omp_nthreads)

    regression_wf = pe.Node(regress(tr=TR),
               name="regression_wf",mem_gb = mem_gbx['timeseries'],n_procs=omp_nthreads)

    interpolate_wf = pe.Node(interpolate(TR=TR),
                  name="interpolation_wf",mem_gb = mem_gbx['timeseries'],n_procs=omp_nthreads)

    qcreport = pe.Node(computeqcplot(TR=TR,bold_file=cifti_file,dummytime=dummytime,
                       head_radius=head_radius), name="qc_report",mem_gb = mem_gbx['resampled'],n_procs=omp_nthreads)

    
    executivesummary_wf =init_execsummary_wf(tr=TR,bold_file=cifti_file,layout=layout,
                      output_dir=output_dir,mni_to_t1w=mni_to_t1w,omp_nthreads=omp_nthreads,mem_gb=mem_gbx['timeseries'])


    workflow.connect([
             # connect bold confound matrix to extract confound matrix 
            (inputnode, confoundmat_wf, [('cifti_file', 'in_file'),]),
         ])
    
    # if there is despiking
    if despike:
        despike_wf = pe.Node(ciftidespike(tr=TR),name="cifti_depike_wf", mem_gb=mem_gbx['timeseries'],n_procs=omp_nthreads)
        workflow.connect([
             (inputnode,despike_wf,[('cifti_file','in_file'),]),
             (despike_wf,censorscrub_wf,[('des_file','inputnode.bold'),]),

        ])
    else:
        workflow.connect([
        (inputnode,censorscrub_wf,[('cifti_file','inputnode.bold'),]),
        ])

    # add neccessary input for censoring if there is one
    workflow.connect([
	     (inputnode,censorscrub_wf,[('cifti_file','inputnode.bold_file')]),
	     (confoundmat_wf,censorscrub_wf,[('confound_file','inputnode.confound_file')])
     ])

    # regression workflow 
    workflow.connect([
	      (censorscrub_wf,regression_wf,[('outputnode.bold_censored','in_file'),
	             ('outputnode.fmriprepconf_censored','confounds'), 
		      ('outputnode.customconf_censored','custom_conf')])
        ])
    # interpolation workflow
    workflow.connect([
	      (inputnode,interpolate_wf,[('cifti_file','bold_file')]),
	      (censorscrub_wf,interpolate_wf,[('outputnode.tmask','tmask')]),
	      (regression_wf,interpolate_wf,[('res_file','in_file')]),     
	])
    # add filtering workflow 
    workflow.connect([
	         (interpolate_wf,filtering_wf,[('bold_interpolated','in_file')]),

    ])
    # residual smoothing 
    workflow.connect([
	   (filtering_wf,resdsmoothing_wf,[('filt_file','inputnode.bold_file')]) 
    ])
    
    #functional connect workflow
    workflow.connect([
         (filtering_wf,cifti_conts_wf,[('filt_file','inputnode.clean_cifti'),]),
      ])
   # reho and alff
    workflow.connect([ 
	 (filtering_wf, alff_compute_wf,[('filt_file','inputnode.clean_bold')]),
	 (filtering_wf, reho_compute_wf,[('filt_file','inputnode.clean_bold')]),
      ])

   # qc report
    workflow.connect([
        (filtering_wf,qcreport,[('filt_file','cleaned_file')]),
        (censorscrub_wf,qcreport,[('outputnode.tmask','tmask')]),
        (qcreport,outputnode,[('qc_file','qc_file')]),
           ])

    workflow.connect([
	    (filtering_wf,outputnode,[('filt_file','processed_bold')]),
	    (censorscrub_wf,outputnode,[('outputnode.fd','fd')]),
	    (resdsmoothing_wf,outputnode,[('outputnode.smoothed_bold','smoothed_bold')]),
	    (alff_compute_wf,outputnode,[('outputnode.alff_out','alff_out')]),
        (reho_compute_wf,outputnode,[('outputnode.lh_reho','reho_lh'),('outputnode.rh_reho','reho_rh')]),
	    (cifti_conts_wf,outputnode,[('outputnode.sc117_ts','sc117_ts' ),('outputnode.sc117_fc','sc117_fc'),
                        ('outputnode.sc217_ts','sc217_ts'),('outputnode.sc217_fc','sc217_fc'),
                        ('outputnode.sc317_ts','sc317_ts'),('outputnode.sc317_fc','sc317_fc'),
                        ('outputnode.sc417_ts','sc417_ts'),('outputnode.sc417_fc','sc417_fc'),
                        ('outputnode.sc517_ts','sc517_ts'),('outputnode.sc517_fc','sc517_fc'),
                        ('outputnode.sc617_ts','sc617_ts'),('outputnode.sc617_fc','sc617_fc'),
                        ('outputnode.sc717_ts','sc717_ts'),('outputnode.sc717_fc','sc717_fc'),
                        ('outputnode.sc817_ts','sc817_ts'),('outputnode.sc817_fc','sc817_fc'),
                        ('outputnode.sc917_ts','sc917_ts'),('outputnode.sc917_fc','sc917_fc'),
                        ('outputnode.sc1017_ts','sc1017_ts'),('outputnode.sc1017_fc','sc1017_fc'),
                        ('outputnode.gs360_ts','gs360_ts'),('outputnode.gs360_fc','gs360_fc'),
                        ('outputnode.gd333_ts','gd333_ts'),('outputnode.gd333_fc','gd333_fc'),
                        ('outputnode.ts50_ts','ts50_ts'),('outputnode.ts50_fc','ts50_fc')]),

       ])



    # write derivatives 
    workflow.connect([
          (filtering_wf,write_derivative_wf,[('filt_file','inputnode.processed_bold')]),
	      (resdsmoothing_wf,write_derivative_wf,[('outputnode.smoothed_bold','inputnode.smoothed_bold')]),
          (censorscrub_wf,write_derivative_wf,[('outputnode.fd','inputnode.fd')]),
          (alff_compute_wf,write_derivative_wf,[('outputnode.alff_out','inputnode.alff_out'),
                                   ('outputnode.smoothed_alff','inputnode.smoothed_alff')]),
          (reho_compute_wf,write_derivative_wf,[('outputnode.rh_reho','inputnode.reho_rh'),
                                     ('outputnode.lh_reho','inputnode.reho_lh')]),
          (cifti_conts_wf,write_derivative_wf,[('outputnode.sc117_ts','inputnode.sc117_ts' ),
                                ('outputnode.sc117_fc','inputnode.sc117_fc'),
                                ('outputnode.sc217_ts','inputnode.sc217_ts'),
                                ('outputnode.sc217_fc','inputnode.sc217_fc'),
                                ('outputnode.sc317_ts','inputnode.sc317_ts'),
                                ('outputnode.sc317_fc','inputnode.sc317_fc'),
                                ('outputnode.sc417_ts','inputnode.sc417_ts'),
                                ('outputnode.sc417_fc','inputnode.sc417_fc'),
                                ('outputnode.sc517_ts','inputnode.sc517_ts'),
                                ('outputnode.sc517_fc','inputnode.sc517_fc'),
                                ('outputnode.sc617_ts','inputnode.sc617_ts'),
                                ('outputnode.sc617_fc','inputnode.sc617_fc'),
                                ('outputnode.sc717_ts','inputnode.sc717_ts'),
                                ('outputnode.sc717_fc','inputnode.sc717_fc'),
                                ('outputnode.sc817_ts','inputnode.sc817_ts'),
                                ('outputnode.sc817_fc','inputnode.sc817_fc'),
                                ('outputnode.sc917_ts','inputnode.sc917_ts'),
                                ('outputnode.sc917_fc','inputnode.sc917_fc'),
                                ('outputnode.sc1017_ts','inputnode.sc1017_ts'),
                                ('outputnode.sc1017_fc','inputnode.sc1017_fc'),
                                ('outputnode.gs360_ts','inputnode.gs360_ts'),
                                ('outputnode.gs360_fc','inputnode.gs360_fc'),
                                ('outputnode.gd333_ts','inputnode.gd333_ts'),
                                ('outputnode.gd333_fc','inputnode.gd333_fc'),
                                ('outputnode.ts50_ts','inputnode.ts50_ts'),
                                ('outputnode.ts50_fc','inputnode.ts50_fc')]),
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

     ## exexetive summary workflow
    workflow.connect([
        (inputnode,executivesummary_wf,[('t1w','inputnode.t1w'),('t1seg','inputnode.t1seg'),
        ('cifti_file','inputnode.bold_file'),]),
        (regression_wf,executivesummary_wf,[('res_file','inputnode.regdata'),]),
        (filtering_wf,executivesummary_wf,[('filt_file','inputnode.resddata')]),
        (censorscrub_wf,executivesummary_wf,[('outputnode.fd','inputnode.fd')]),
    ]),

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

class DerivativesDataSink(bid_derivative):
    out_path_base = 'xcp_d'

def get_ciftiTR(cifti_file):
    import nibabel as nb
    ciaxis = nb.load(cifti_file).header.get_axis(0)
    return ciaxis.step

