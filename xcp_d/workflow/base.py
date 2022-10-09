# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The primary workflows for xcp_d."""

import glob
import json
import os
import sys
from copy import deepcopy

import nibabel as nb
import numpy as np
import scipy
import templateflow
from nipype import Function
from nipype import __version__ as nipype_ver
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.__about__ import __version__
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.report import AboutSummary, SubjectSummary
from xcp_d.utils.bids import (
    collect_data,
    extract_t1w_seg,
    select_cifti_bold,
    select_registrationfile,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import get_customfile
from xcp_d.workflow.anatomical import init_anatomical_wf, init_t1w_wf
from xcp_d.workflow.bold import init_boldpostprocess_wf
from xcp_d.workflow.cifti import init_ciftipostprocess_wf


@fill_doc
def init_xcpd_wf(
    layout,
    lower_bpf,
    upper_bpf,
    despike,
    bpf_order,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    bandpass_filter,
    fmri_dir,
    omp_nthreads,
    cifti,
    task_id,
    head_radius,
    params,
    subject_list,
    analysis_level,
    smoothing,
    custom_confounds,
    output_dir,
    work_dir,
    dummytime,
    fd_thresh,
    process_surfaces=False,
    input_type='fmriprep',
    name='xcpd_wf',
):
    """Build and organize execution of xcp_d pipeline.

    It also connects the subworkflows under the xcp_d workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.base import init_xcpd_wf
            wf = init_xcpd_wf(
                layout=None,
                lower_bpf=0.009,
                upper_bpf=0.08,
                despike=False,
                bpf_order=2,
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=0.,
                band_stop_max=0.,
                bandpass_filter=True,
                fmri_dir=".",
                omp_nthreads=1,
                cifti=True,
                task_id="rest",
                head_radius=50.,
                params="36P",
                subject_list=["sub-01", "sub-02"],
                analysis_level="participant",
                smoothing=6,
                custom_confounds=None,
                output_dir=".",
                work_dir=".",
                dummytime=0,
                fd_thresh=0.2,
                process_surfaces=True,
                input_type='fmriprep',
                name='xcpd_wf',
            )

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        BIDS dataset layout
    %(bandpass_filter)s
    %(lower_bpf)s
    %(upper_bpf)s
    despike: bool
        afni depsike
    %(bpf_order)s
    %(analysis_level)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    fmriprep_dir : Path
        fmriprep output directory
    %(omp_nthreads)s
    %(cifti)s
    task_id : str or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    low_mem : bool
        Write uncompressed .nii files in some cases to reduce memory usage
    %(output_dir)s
    %(fd_thresh)s
    run_uuid : str
        Unique identifier for execution instance
    subject_list : list
        List of subject labels
    %(work_dir)s
    %(head_radius)s
    %(params)s
    %(smoothing)s
    custom_confounds: str
        path to cusrtom nuisance regressors
    dummytime: float
        the first vols in seconds to be removed before postprocessing
    %(process_surfaces)s
    %(input_type)s
    %(name)s
    """
    xcpd_wf = Workflow(name='xcpd_wf')
    xcpd_wf.base_dir = work_dir
    print("Begin the " + name + " workflow")
    for subject_id in subject_list:
        single_subj_wf = init_subject_wf(
            layout=layout,
            lower_bpf=lower_bpf,
            upper_bpf=upper_bpf,
            bpf_order=bpf_order,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            bandpass_filter=bandpass_filter,
            fmri_dir=fmri_dir,
            omp_nthreads=omp_nthreads,
            subject_id=subject_id,
            cifti=cifti,
            despike=despike,
            head_radius=head_radius,
            params=params,
            task_id=task_id,
            smoothing=smoothing,
            output_dir=output_dir,
            dummytime=dummytime,
            custom_confounds=custom_confounds,
            fd_thresh=fd_thresh,
            process_surfaces=process_surfaces,
            input_type=input_type,
            name="single_subject_" + subject_id + "_wf")

        single_subj_wf.config['execution']['crashdump_dir'] = (os.path.join(
            output_dir, "xcp_d", "sub-" + subject_id, 'log'))
        for node in single_subj_wf._get_all_nodes():
            node.config = deepcopy(single_subj_wf.config)
        print("Analyzing data at the " + str(analysis_level) + " level")
        xcpd_wf.add_nodes([single_subj_wf])

    return xcpd_wf


@fill_doc
def init_subject_wf(
    layout,
    lower_bpf,
    upper_bpf,
    bpf_order,
    motion_filter_order,
    motion_filter_type,
    bandpass_filter,
    band_stop_min,
    band_stop_max,
    fmri_dir,
    omp_nthreads,
    subject_id,
    cifti,
    despike,
    head_radius,
    params,
    dummytime,
    fd_thresh,
    task_id,
    smoothing,
    custom_confounds,
    process_surfaces,
    output_dir,
    input_type,
    name,
):
    """Organize the postprocessing pipeline for a single subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.base import init_subject_wf
            wf = init_subject_wf(
                layout=None,
                bandpass_filter=True,
                lower_bpf=0.009,
                upper_bpf=0.08,
                bpf_order=2,
                motion_filter_type=None,
                band_stop_min=0,
                band_stop_max=0,
                motion_filter_order=4,
                fmri_dir=".",
                omp_nthreads=1,
                subject_id="sub-01",
                cifti=False,
                despike=False,
                head_radius=50,
                params="36P",
                dummytime=0,
                fd_thresh=0.2,
                task_id="rest",
                smoothing=6.,
                custom_confounds=None,
                process_surfaces=True,
                output_dir=".",
                input_type="fmriprep",
                name="single_subject_sub-01_wf",
            )

    Parameters
    ----------
    layout : BIDSLayout object
        BIDS dataset layout
    %(bandpass_filter)s
    %(lower_bpf)s
    %(upper_bpf)s
    despike: bool
        afni depsike
    %(bpf_order)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(fmri_dir)s
    %(omp_nthreads)s
    %(cifti)s
    task_id : str or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    low_mem : bool
        Write uncompressed .nii files in some cases to reduce memory usage
    %(output_dir)s
    %(fd_thresh)s
    %(head_radius)s
    %(params)s
    %(smoothing)s
    custom_confounds: str
        path to custom nuisance regressors
    dummytime: float
        the first vols in seconds to be removed before postprocessing
    %(process_surfaces)s
    %(input_type)s
    %(name)s
    """
    layout, subj_data = collect_data(bids_dir=fmri_dir,
                                     participant_label=subject_id,
                                     task=task_id,
                                     bids_validate=False)

    preproc_nifti_files, preproc_cifti_files = select_cifti_bold(subj_data=subj_data)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['custom_confounds', 'subj_data']),
        name='inputnode',
    )
    inputnode.inputs.custom_confounds = custom_confounds
    inputnode.inputs.subj_data = subj_data

    workflow = Workflow(name=name)

    workflow.__desc__ = f"""
### Post-processing of {input_type} outputs
The eXtensible Connectivity Pipeline (XCP) [@mitigating_2018;@satterthwaite_2013]
was used to post-process the outputs of fMRIPrep version {getfmriprepv(fmri_dir=fmri_dir)}
[@esteban2019fmriprep;esteban2020analysis, RRID:SCR_016216].
XCP was built with *Nipype* {nipype_ver} [@nipype1, RRID:SCR_002502].
"""

    workflow.__postdesc__ = f"""

Many internal operations of *XCP* use
*TemplateFlow* version {templateflow.__version__} [@ciric2022templateflow],
*Nibabel* version {nb.__version__} [@brett_matthew_2022_6658382],
*numpy* version {np.__version__} [@harris2020array],
and *scipy* version {scipy.__version__} [@2020SciPy-NMeth].
For more details, see the *xcp_d* website https://xcp-d.readthedocs.io.


#### Copyright Waiver

The above methods description text was automatically generated by *XCP*
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0](https://creativecommons.org/publicdomain/zero/1.0/) license.

#### References

"""

    summary = pe.Node(SubjectSummary(subject_id=subject_id,
                                     bold=preproc_nifti_files),
                      name='summary')

    about = pe.Node(AboutSummary(version=__version__,
                                 command=' '.join(sys.argv)),
                    name='about')

    ds_report_summary = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        source_file=preproc_nifti_files[0],
        desc='summary',
        datatype="figures"),
        name='ds_report_summary')

    t1w_file_grabber = pe.Node(
        Function(
            input_names=["subj_data"],
            output_names=["t1w", "t1seg"],
            function=extract_t1w_seg,
        ),
        name="t1w_file_grabber",
    )

    transform_file_grabber = pe.Node(
        Function(
            input_names=["subj_data"],
            output_names=["mni_to_t1w", "t1w_to_mni"],
            function=select_registrationfile,
        ),
        name="transform_file_grabber",
    )

    t1w_wf = init_t1w_wf(
        output_dir=output_dir,
        input_type=input_type,
        omp_nthreads=omp_nthreads,
        mem_gb=5,  # RF: need to change memory size
    )

    workflow.connect([
        (inputnode, t1w_file_grabber, [('subj_data', 'subj_data')]),
        (inputnode, transform_file_grabber, [('subj_data', 'subj_data')]),
        (t1w_file_grabber, t1w_wf, [('t1w', 'inputnode.t1w'), ('t1seg', 'inputnode.t1seg')]),
        (transform_file_grabber, t1w_wf, [('t1w_to_mni', 'inputnode.t1w_to_mni')]),
    ])

    if process_surfaces:
        anatomical_wf = init_anatomical_wf(
            omp_nthreads=omp_nthreads,
            fmri_dir=fmri_dir,
            subject_id=subject_id,
            output_dir=output_dir,
            input_type=input_type,
            mem_gb=5,  # RF: need to change memory size
        )

        workflow.connect([
            (t1w_file_grabber, anatomical_wf, [('t1w', 'inputnode.t1w'),
                                               ('t1seg', 'inputnode.t1seg')]),
        ])

    # determine the appropriate post-processing workflow
    postproc_wf_function = init_ciftipostprocess_wf if cifti else init_boldpostprocess_wf
    preproc_files = preproc_cifti_files if cifti else preproc_nifti_files

    # loop over each bold run to be postprocessed
    for i_run, bold_file in enumerate(preproc_files):
        custom_confounds_file = get_customfile(
            custom_confounds=custom_confounds,
            bold_file=bold_file,
        )

        bold_postproc_wf = postproc_wf_function(
            bold_file=bold_file,
            lower_bpf=lower_bpf,
            upper_bpf=upper_bpf,
            bpf_order=bpf_order,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            bandpass_filter=bandpass_filter,
            smoothing=smoothing,
            params=params,
            head_radius=head_radius,
            omp_nthreads=omp_nthreads,
            n_runs=len(preproc_files),
            custom_confounds=custom_confounds_file,
            layout=layout,
            despike=despike,
            dummytime=dummytime,
            fd_thresh=fd_thresh,
            output_dir=output_dir,
            name=f"{'cifti' if cifti else 'nifti'}_postprocess_{i_run}_wf",
        )

        # NOTE: TS- Why is the data sink initialized separately for each run?
        # If it's run-specific, shouldn't the name reflect the run?
        ds_report_about = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                desc='about',
                datatype="figures",
            ),
            name='ds_report_about',
            run_without_submitting=True,
        )

        workflow.connect(
            [
                (t1w_file_grabber, bold_postproc_wf, [('t1w', 'inputnode.t1w'),
                                                      ('t1seg', 'inputnode.t1seg')]),
                (transform_file_grabber, bold_postproc_wf, [
                    ('mni_to_t1w', 'inputnode.mni_to_t1w'),
                ]),
            ],
        )

    try:
        workflow.connect([(summary, ds_report_summary, [('out_report', 'in_file')
                                                        ]),
                          (about, ds_report_about, [('out_report', 'in_file')])])
    except Exception as exc:
        if cifti:
            exc = "No cifti files ending with 'bold.dtseries.nii' found for one or more" \
                " participants."
            print(exc)
            sys.exit()

    for node in workflow.list_node_names():
        if node.split('.')[-1].startswith('ds_'):
            workflow.get_node(node).interface.out_path_base = 'xcp_d'

    return workflow


def _pop(inlist):
    """Make a list of lists into a list."""
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def getfmriprepv(fmri_dir):
    """Get fmriprep/nibabies/dcan/hcp version."""
    datax = glob.glob(fmri_dir + '/dataset_description.json')

    if datax:
        datax = datax[0]
        with open(datax) as f:
            datay = json.load(f)

        fvers = datay['GeneratedBy'][0]['Version']
    else:
        fvers = str('Unknown vers')

    return fvers
