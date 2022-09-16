# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The primary workflows for xcp_d.

post processing
^^^^^^^^^^^^^^^

.. autofunction:: init_xcpd_wf

"""

import sys
import glob
import json
import os
from copy import deepcopy
from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from xcp_d.__about__ import __version__
from xcp_d.utils import (collect_data, get_customfile, select_cifti_bold,
                         select_registrationfile, extract_t1w_seg)
from xcp_d.workflow.bold import init_boldpostprocess_wf
from xcp_d.workflow.cifti import init_ciftipostprocess_wf
from xcp_d.workflow.anatomical import init_anatomical_wf
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from xcp_d.interfaces import SubjectSummary, AboutSummary
from xcp_d.utils import bid_derivative


def init_xcpd_wf(layout,
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
                 input_type='fmriprep',
                 name='xcpd_wf'):
    """
    This workflow builds and organizes  execution of  xcp_d  pipeline.
    It is also connect the subworkflows under the xcp_d
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_d.workflow.base import init_xcpd_wf
            wf = init_xcpd_wf(
                layout,
                lower_bpf,
                upper_bpf,
                despike,
                bpf_order,
                motion_filter_type,
                motion_filter_order,
                band_stop_min,
                band_stop_max,
                fmriprep_dir,
                omp_nthreads,
                cifti,
                task_id,
                head_radius,
                params,
                brain_template,
                subject_list,
                smoothing,
                analysis_level,
                custom_confounds,
                output_dir,
                work_dir,
                dummytime,
                fd_thresh,
            )

    Parameters
    ----------
    lower_bpf : float
        Lower band pass filter
    upper_bpf : float
        Upper band pass filter
    layout : BIDSLayout object
        BIDS dataset layout
    despike: bool
        afni depsike
    analysis_level: str
        only option is 'participant'
    motion_filter_type: str
        respiratory motion filter type: lp or notch
    motion_filter_order: str
        respiratory motion order
    band_stop_min: float
        respiratory minimum frequency in breathe per minutes(bpm)
    band_stop_max,: float
        respiratory maximum frequency in breathe per minutes(bpm)
    fmriprep_dir : Path
        fmriprep output directory
    omp_nthreads : int
        Maximum number of threads an individual process may use
    cifti : bool
        To postprocessed cifti files instead of nifti
    task_id : str or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    low_mem : bool
        Write uncompressed .nii files in some cases to reduce memory usage
    output_dir : str
        Directory in which to save xcp_d output
    fd_thresh
        Criterion for flagging framewise displacement outliers
    run_uuid : str
        Unique identifier for execution instance
    subject_list : list
        List of subject labels
    work_dir : str
        Directory in which to store workflow execution state and temporary files
    head_radius : float
        radius of the head for FD computation
    params: str
        nuissance regressors to be selected from fmriprep regressors
    smoothing: float
        smooth the derivatives output with kernel size (fwhm)
    custom_confounds: str
        path to cusrtom nuissance regressors
    dummytime: float
        the first vols in seconds to be removed before postprocessing
s
    """

    xcpd_wf = Workflow(name='xcpd_wf')
    xcpd_wf.base_dir = work_dir

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
            input_type=input_type,
            name="single_subject_" + subject_id + "_wf")

        single_subj_wf.config['execution']['crashdump_dir'] = (os.path.join(
            output_dir, "xcp_d", "sub-" + subject_id, 'log'))
        for node in single_subj_wf._get_all_nodes():
            node.config = deepcopy(single_subj_wf.config)
        xcpd_wf.add_nodes([single_subj_wf])

    return xcpd_wf


def init_subject_wf(layout, lower_bpf, upper_bpf, bpf_order, motion_filter_order,
                    motion_filter_type, bandpass_filter,
                    band_stop_min, band_stop_max, fmri_dir, omp_nthreads,
                    subject_id, cifti, despike, head_radius, params, dummytime,
                    fd_thresh, task_id, smoothing, custom_confounds, output_dir,
                    input_type, name):
    """This workflow organizes the postprocessing pipeline for a single subject

    # RF: this is the wrong function
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_d.workflows.base import init_single_bold_wf
            wf = init_single_bold_wf(
                layout,
                lower_bpf,
                upper_bpf,
                bpf_order,
                motion_filter_type,
                motion_filter_order,
                band_stop_min,
                band_stop_max,
                fmriprep_dir,
                omp_nthreads,
                subject_id,
                cifti,
                head_radius,
                params,
                scrub,
                dummytime,
                fd_thresh,
                task_id,
                template,
                smoothing,
                custom_confounds,
                bids_filters,
                output_dir
             )
    Parameters
    ----------
    lower_bpf : float
        Lower band pass filter
    upper_bpf : float
        Upper band pass filter
    layout : BIDSLayout object
        BIDS dataset layout
    despike: bool
        afni depsike
    motion_filter_type: str
        respiratory motion filter type: lp or notch
    motion_filter_order: int
        order for motion filter
    band_stop_min: float
        respiratory minimum frequency in breathe per minutes(bpm)
    band_stop_max,: float
        respiratory maximum frequency in breathe per minutes(bpm)
    fmriprep_dir : Path
        fmriprep output directory
    omp_nthreads : int
        Maximum number of threads an individual process may use
    cifti : bool
        To postprocessed cifti files instead of nifti
    task_id : str or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    low_mem : bool
        Write uncompressed .nii files in some cases to reduce memory usage
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
    custom_confounds: str
        path to custom nuissance regressors
    dummytime: float
        the first vols in seconds to be removed before postprocessing

    """

    layout, subj_data = collect_data(bids_dir=fmri_dir,
                                     participant_label=subject_id,
                                     task=task_id,
                                     bids_validate=False)

    regfile = select_registrationfile(subj_data=subj_data)
    subject_data = select_cifti_bold(subj_data=subj_data)
    t1wseg = extract_t1w_seg(subj_data=subj_data)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['custom_confounds', 'mni_to_t1w', 't1w', 't1seg']),
        name='inputnode')
    inputnode.inputs.custom_confounds = custom_confounds
    inputnode.inputs.t1w = t1wseg[0]
    inputnode.inputs.t1seg = t1wseg[1]
    mni_to_t1w = regfile[0]
    inputnode.inputs.mni_to_t1w = mni_to_t1w

    workflow = Workflow(name=name)

    workflow.__desc__ = """
### Post-processing of {input_type} outputs
The eXtensible Connectivity Pipeline (XCP) [@mitigating_2018;@satterthwaite_2013]
was used to post-process the outputs of fMRIPrep version {fvers} [@fmriprep1].
XCP was built with *Nipype* {nipype_ver} [@nipype1].
""".format(input_type=input_type,
           nipype_ver=nipype_ver,
           fvers=getfmriprepv(fmri_dir=fmri_dir))

    workflow.__postdesc__ = """

Many internal operations of *XCP* use *Nibabel* [@nilearn], *numpy*
[@harris2020array], and  *scipy* [@2020SciPy-NMeth]. For more details,
see the *xcp_d* website https://xcp-d.readthedocs.io.


#### Copyright Waiver
The above methods descrip text was automatically generated by *XCP*
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

#### References

"""

    summary = pe.Node(SubjectSummary(subject_id=subject_id,
                                     bold=subject_data[0]),
                      name='summary')

    about = pe.Node(AboutSummary(version=__version__,
                                 command=' '.join(sys.argv)),
                    name='about')

    ds_report_summary = pe.Node(DerivativesDataSink(
        base_directory=output_dir,
        source_file=subject_data[0][0],
        desc='summary',
        datatype="figures"),
        name='ds_report_summary')

    anatomical_wf = init_anatomical_wf(
        omp_nthreads=omp_nthreads,
        fmri_dir=fmri_dir,
        subject_id=subject_id,
        output_dir=output_dir,
        t1w_to_mni=regfile[1],
        input_type=input_type,
        mem_gb=5)  # RF: need to chnage memory size

    # send t1w and t1seg to anatomical workflow
    workflow.connect([
        (inputnode, anatomical_wf, [('t1w', 'inputnode.t1w'),
                                    ('t1seg', 'inputnode.t1seg')]),
    ])

    # loop over each bold data to be postprocessed
    # RF: get rid of ii's
    if cifti:
        ii = 0
        for cifti_file in subject_data[1]:
            ii = ii + 1
            custom_confoundsx = get_customfile(custom_confounds=custom_confounds,
                                               bold_file=cifti_file)
            cifti_postproc_wf = init_ciftipostprocess_wf(
                cifti_file=cifti_file,
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
                custom_confounds=custom_confoundsx,
                omp_nthreads=omp_nthreads,
                num_cifti=len(subject_data[1]),
                dummytime=dummytime,
                fd_thresh=fd_thresh,
                despike=despike,
                layout=layout,
                mni_to_t1w=regfile[0],
                output_dir=output_dir,
                name='cifti_postprocess_' + str(ii) + '_wf')

            ds_report_about = pe.Node(DerivativesDataSink(
                base_directory=output_dir,
                source_file=cifti_file,
                desc='about',
                datatype="figures",
            ),
                name='ds_report_about',
                run_without_submitting=True)

            workflow.connect([(inputnode, cifti_postproc_wf,
                               [('custom_confounds', 'inputnode.custom_confounds'),
                                ('t1w', 'inputnode.t1w'),
                                ('t1seg', 'inputnode.t1seg')])])

    else:
        ii = 0
        for bold_file in subject_data[0]:
            ii = ii + 1
            custom_confoundsx = get_customfile(custom_confounds=custom_confounds,
                                               bold_file=bold_file)
            bold_postproc_wf = init_boldpostprocess_wf(
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
                brain_template='MNI152NLin2009cAsym',
                num_bold=len(subject_data[0]),
                custom_confounds=custom_confoundsx,
                layout=layout,
                despike=despike,
                dummytime=dummytime,
                fd_thresh=fd_thresh,
                output_dir=output_dir,
                mni_to_t1w=mni_to_t1w,
                name='bold_postprocess_' + str(ii) + '_wf')

            ds_report_about = pe.Node(DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                desc='about',
                datatype="figures"),
                name='ds_report_about',
                run_without_submitting=True)

            workflow.connect([(inputnode, bold_postproc_wf,
                               [('mni_to_t1w', 'inputnode.mni_to_t1w'),
                                ('t1w', 'inputnode.t1w'),
                                ('t1seg', 'inputnode.t1seg')])])

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


def _prefix(subid):
    """
    Prefix for subject id
    """
    if subid.startswith('sub-'):
        return subid
    return '-'.join(('sub', subid))


def _pop(inlist):
    """
    make a list of lists into a list
    """
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


# RF: this shouldn't be in this file
class DerivativesDataSink(bid_derivative):
    """
    defines the data sink for the workflow
    """
    out_path_base = 'xcp_d'


def getfmriprepv(fmri_dir):
    """
    get fmriprep/nibabies/dcan/hcp version
    """

    datax = glob.glob(fmri_dir + '/dataset_description.json')

    if datax:
        datax = datax[0]
        with open(datax) as f:
            datay = json.load(f)

        fvers = datay['GeneratedBy'][0]['Version']
    else:
        fvers = str('Unknown vers')

    return fvers


def _getsesid(filename):
    """
    get session id from filename if available
    """
    ses_id = None
    filex = os.path.basename(filename)

    file_id = filex.split('_')
    for k in file_id:
        if 'ses' in k:
            ses_id = k.split('-')[1]
            break

    return ses_id
