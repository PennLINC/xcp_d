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


def init_xcpabcd_wf(layout,
                   lowpass,
                   highpass,
                   fmriprep_dir,
                   omp_nthreads,
                   surface,
                   head_radius,
                   params,
                   template,
                   subject_list,
                   smoothing,
                   customs_conf,
                   bids_filters,
                   output_dir,
                   work_dir,
                   name):
    
    """
    coming to fix this 

    """

    xcpabcd_wf = pe.Workflow(name='xcpabcd_wf')
    xcpabcd_wf.base_dir = work_dir

    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
                            layout=layout,
                            lowpass=lowpass,
                            highpass=highpass,
                            fmriprep_dir=fmriprep_dir,
                            omp_nthreads=omp_nthreads,
                            subject_id=subject_id,
                            task_id=task_id,
                            surface=surface,
                            head_radius=head_radius,
                            params=params,
                            template=template,
                            smoothing=smoothing,
                            customs_conf=customs_conf,
                            bids_filters=bids_filters,
                            name="single_subject_" + subject_id + "_wf")

        single_subject_wf.config['execution']['crashdump_dir'] = (
            os.path.join(output_dir, "xcpabcd", "sub-" + subject_id, 'log')
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
    task_id,
    surface,
    head_radius,
    params,
    template,
    smoothing,
    customs_conf,
    bids_filters,
    name
    ):
    """
    

    """
    layout,subject_data,regfile = collect_data(bids_dir=fmriprep_dir,participant_label=subject_id, 
                                               task=task_id,bids_validate=False, 
                                               bids_filters=bids_filters,template=template)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['customs_conf','mni_to_t1w']),
        name='inputnode')
    inputnode.inputs.customs_conf = customs_conf
    inputnode.inputs.mni_to_t1w = regfile[0]
    
    workflow = pe.Workflow(name=name)
    if surface:
        for cifti_file in subject_data[1]:
            cifti_postproc_wf = init_ciftipostprocess_wf(cifti_file=cifti_file,
                                                        lowpass=lowpass,
                                                        highpass=highpass,
                                                        smoothing=smoothing,
                                                        head_radius=head_radius,
                                                        params=params,
                                                        omp_nthreads=omp_nthreads,
                                                        num_cifti=1,
                                                        layout=layout,
                                                        name='cifti_process_wf')
            workflow.connect([
                  (inputnode,cifti_postproc_wf,[('customs_conf','inputnode.customs_conf')]),
            ])

            
    else:
        for bold_file in subject_data[0]:
            mni_to_t1w = regfile[0]
            bold_postproc_wf = init_boldpostprocess_wf(bold_file=bold_file,
                                                       lowpass=lowpass,
                                                       highpass=highpass,
                                                       smoothing=smoothing,
                                                       head_radius=head_radius,
                                                       params=params,
                                                       omp_nthreads=omp_nthreads,
                                                       template='MNI152NLin2009cAsym',
                                                       num_bold=1,
                                                       layout=layout,
                                                       name='bold_postprocess_wf')
            workflow.connect([
                  (inputnode,bold_postproc_wf,[('customs_conf','inputnode.customs_conf'),
                                                ('mni_to_t1w','inputnode.mni_to_t1w')]),
            ])


    return workflow


def _prefix(subid):
    if subid.startswith('sub-'):
        return subid
    return '-'.join(('sub', subid))


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist

