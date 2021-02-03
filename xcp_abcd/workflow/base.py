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


def init_xcpabcd_wf():
    
    """
    coming to fix this 

    """

    xcpabcd_wf = pe.Workflow(name='xcpabcd_wf')
    xcpabcd_wf.base_dir = work_dir

    for subject_id in subject_list:
        single_subject_wf = init_single_subject_wf(
            layout=layout,
            name="single_subject_" + subject_id + "_wf",
            omp_nthreads=omp_nthreads,
            output_dir=output_dir,
            subject_id=subject_id,
            task_id=task_id,
            bids_filters=bids_filters,
            run_uuid=run_uuid
        )

        single_subject_wf.config['execution']['crashdump_dir'] = (
            os.path.join(output_dir, "xcpabcd", "sub-" + subject_id, 'log', run_uuid)
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)
        xcpabcd_wf.add_nodes([single_subject_wf])

    return xcpabcd_wf


def init_single_subject_wf(
    layout,
    fmriprep_dir
    low_mem,
    name,
    omp_nthreads,
    subject_id,
    task_id,
    bids_filters,
    name='single_subject' + subject_id + '_wf'
   ):
    """
    

    """
    layout,subject_data,regfile = collect_data(fmriprep_dir,subtject_id, task_id,bids_validate=False, 
                                    bids_filters=bids_filters,template=template)
    
    
    workflow = pe.workflowWorkflow(name=name)
    if surface:
        for cifti_file in subject_data[1]:
            func_postproc_wf = init_ciftipostprocess_wf()
    else:
        for bold_file in subject_data[0]:
            mni_to_t1w = regfile[0]
            func_postproc_wf = init_boldpostprocess_wf(bold_file=bold_file,
                                                       mni_to_t1w=mni_to_t1w,
                                                       lowpass=lowpass,
                                                       highpass=highpass,
                                                       smoothing=smoothing,
                                                       head_radius=head_radius,
                                                       params=params,
                                                       omp_nthreads=omp_nthreads,
                                                       template='MNI152NLin2009cAsym',
                                                       num_bold=1,
                                                       layout=layout,
                                                       name='bold_process_wf')


    return workflow


def _prefix(subid):
    if subid.startswith('sub-'):
        return subid
    return '-'.join(('sub', subid))


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist

