# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
 xcp_abcd  postprocessing workflows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

import sys
import os
from copy import deepcopy
from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from ..__about__ import __version__

from ..utils import collect_data

from  ..workflow import( init_ciftipostprocess_wf, 
            init_boldpostprocess_wf)
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ..interfaces import SubjectSummary, AboutSummary
from  ..utils import bid_derivative



def init_xcpabcd_wf(layout,
                   lowpass,
                   highpass,
                   fmriprep_dir,
                   omp_nthreads,
                   surface,
                   task_id,
                   head_radius,
                   params,
                   template,
                   subject_list,
                   smoothing,
                   custom_conf,
                   bids_filters,
                   output_dir,
                   work_dir,
                   scrub,
                   dummytime,
                   fd_thresh,
                   name):
    
    """
    coming to fix this 

    """

    xcpabcd_wf = Workflow(name='xcpabcd_wf')
    xcpabcd_wf.base_dir = work_dir

    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
                            layout=layout,
                            lowpass=lowpass,
                            highpass=highpass,
                            fmriprep_dir=fmriprep_dir,
                            omp_nthreads=omp_nthreads,
                            subject_id=subject_id,
                            surface=surface,
                            head_radius=head_radius,
                            params=params,
                            task_id=task_id,
                            template=template,
                            smoothing=smoothing,
                            custom_conf=custom_conf,
                            bids_filters=bids_filters,
                            output_dir=output_dir,
                            scrub=scrub,
                            dummytime=dummytime,
                            fd_thresh=fd_thresh,
                            name="single_subject_" + subject_id + "_wf")

        single_subject_wf.config['execution']['crashdump_dir'] = (
            os.path.join(output_dir, "xcp_abcd", "sub-" + subject_id, 'log')
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        xcpabcd_wf.add_nodes([single_subject_wf])

    return xcpabcd_wf


def init_single_subject_wf(
    layout,
    lowpass,
    highpass,
    fmriprep_dir,
    omp_nthreads,
    subject_id,
    surface,
    head_radius,
    params,
    scrub,
    dummytime,
    fd_thresh,
    task_id,
    template,
    smoothing,
    custom_conf,
    bids_filters,
    output_dir,
    name
    ):
    """
    

    """
    layout,subject_data,regfile = collect_data(bids_dir=fmriprep_dir,participant_label=subject_id, 
                                               task=task_id,bids_validate=False, 
                                               bids_filters=bids_filters,template=template)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['custom_conf','mni_to_t1w']),
        name='inputnode')
    inputnode.inputs.custom_conf = custom_conf
    inputnode.inputs.mni_to_t1w = regfile[0]
    
    workflow = Workflow(name=name)
    summary = pe.Node(SubjectSummary(subject_id=subject_id,bold=subject_data[0]),
                      name='summary', run_without_submitting=True)

    about = pe.Node(AboutSummary(version='trial_Ax',
                                 command=' '.join(sys.argv)),
                    name='about', run_without_submitting=True)

    ds_report_summary = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='summary', datatype="figures"),
                  name='ds_report_summary', run_without_submitting=True)

    ds_report_about = pe.Node(
        DerivativesDataSink(base_directory=output_dir, desc='about', datatype="figures",),
        name='ds_report_about', run_without_submitting=True)

    if surface:
        ii=0
        for cifti_file in subject_data[1]:
            ii = ii+1
            cifti_postproc_wf = init_ciftipostprocess_wf(cifti_file=cifti_file,
                                                        lowpass=lowpass,
                                                        highpass=highpass,
                                                        smoothing=smoothing,
                                                        head_radius=head_radius,
                                                        params=params,
                                                        custom_conf=custom_conf,
                                                        omp_nthreads=omp_nthreads,
                                                        num_cifti=1,
                                                        scrub=scrub,
                                                        dummytime=dummytime,
                                                        fd_thresh=fd_thresh,
                                                        layout=layout,
                                                        output_dir=output_dir,
                                                        name='cifti_postprocess_'+ str(ii) + '_wf')
            workflow.connect([
                  (inputnode,cifti_postproc_wf,[('custom_conf','inputnode.custom_conf')]),
            ])

            
    else:
        ii = 0
        for bold_file in subject_data[0]:
            ii = ii+1
            mni_to_t1w = regfile[0]
            inputnode.inputs.mni_to_t1w = mni_to_t1w
            bold_postproc_wf = init_boldpostprocess_wf(bold_file=bold_file,
                                                       lowpass=lowpass,
                                                       highpass=highpass,
                                                       smoothing=smoothing,
                                                       head_radius=head_radius,
                                                       params=params,
                                                       omp_nthreads=omp_nthreads,
                                                       template='MNI152NLin2009cAsym',
                                                       num_bold=1,
                                                       custom_conf=custom_conf,
                                                       layout=layout,
                                                       scrub=scrub,
                                                       dummytime=dummytime,
                                                       fd_thresh=fd_thresh,
                                                       output_dir=output_dir,
                                                       name='bold_postprocess_'+ str(ii) + '_wf')
            workflow.connect([
                  (inputnode,bold_postproc_wf,[ ('mni_to_t1w','inputnode.mni_to_t1w')]),
            ])
    workflow.connect([ 
        (summary,ds_report_summary,[('out_report','in_file')]),
        (about, ds_report_about, [('out_report', 'in_file')]),
         
       ])
    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = 'xcp_abcd'

    return workflow


def _prefix(subid):
    if subid.startswith('sub-'):
        return subid
    return '-'.join(('sub', subid))


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist

class DerivativesDataSink(bid_derivative):
    out_path_base = 'xcp_abcd'