# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
post processing 
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_xcpabcd_wf

"""

import sys
import glob
import json
import os
from copy import deepcopy
from nipype import __version__ as nipype_ver
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from ..__about__ import __version__
from ..utils import collect_data, get_customfile,select_cifti_bold,select_registrationfile,extract_t1w_seg
from .bold import init_boldpostprocess_wf
from .cifti import init_ciftipostprocess_wf
from .anatomical import init_anatomical_wf
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ..interfaces import SubjectSummary, AboutSummary
from  ..utils import bid_derivative



def init_xcpabcd_wf(layout,
                   lower_bpf,
                   upper_bpf,
                   contigvol,
                   despike,
                   bpf_order,
                   motion_filter_order,
                   motion_filter_type,
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
                   custom_conf,
                   output_dir,
                   work_dir,
                   dummytime,
                   fd_thresh,
                   name):
    
    """
    This workflow builds and organizes  execution of  xcp_abcd  pipeline.
    It is also connect the subworkflows under the xcp_abcd
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflow.base import init_xcpabcd_wf
            wf = init_xcpabcd_wf(
                layout,
                lower_bpf,
                upper_bpf,
                contigvol,
                despike,
                bpf_order,
                motion_filter_order,
                motion_filter_type,
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
                custom_conf,
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
        Directory in which to save xcp_abcd output
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
    custom_conf: str
        path to cusrtom nuissance regressors 
    dummytime: float
        the first vols in seconds to be removed before postprocessing
    
    """

    xcpabcd_wf = Workflow(name='xcpabcd_wf')
    xcpabcd_wf.base_dir = work_dir

    for subject_id in subject_list:
        single_subj_wf = init_subject_wf(
                            layout=layout,
                            lower_bpf=lower_bpf,
                            upper_bpf=upper_bpf,
                            contigvol=contigvol,
                            bpf_order=bpf_order,
                            motion_filter_order=motion_filter_order,
                            motion_filter_type=motion_filter_type,
                            band_stop_min=band_stop_min,
                            band_stop_max=band_stop_max,
                            fmriprep_dir=fmriprep_dir,
                            omp_nthreads=omp_nthreads,
                            subject_id=subject_id,
                            cifti=cifti,
                            despike=despike,
                            head_radius=head_radius,
                            params=params,
                            task_id=task_id,
                            brain_template=brain_template,
                            smoothing=smoothing,
                            output_dir=output_dir,
                            dummytime=dummytime,
                            custom_conf=custom_conf,
                            fd_thresh=fd_thresh,
                            name="single_subject_" + subject_id + "_wf")

        single_subj_wf.config['execution']['crashdump_dir'] = (
            os.path.join(output_dir, "xcp_abcd", "sub-" + subject_id, 'log')
        )
        for node in single_subj_wf._get_all_nodes():
            node.config = deepcopy(single_subj_wf.config)
        xcpabcd_wf.add_nodes([single_subj_wf])

    return xcpabcd_wf


def init_subject_wf(
    layout,
    lower_bpf,
    upper_bpf,
    contigvol,
    bpf_order,
    motion_filter_order,
    motion_filter_type,
    band_stop_min,
    band_stop_max,
    fmriprep_dir,
    omp_nthreads,
    subject_id,
    cifti,
    despike,
    head_radius,
    params,
    dummytime,
    fd_thresh,
    task_id,
    brain_template,
    smoothing,
    custom_conf,
    output_dir,
    name
    ):
    """
    This workflow organizes the postprocessing pipeline for a single bold or cifti.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from xcp_abcd.workflows.base import init_single_bold_wf
            wf = init_single_bold_wf(
                layout,
                lower_bpf,
                upper_bpf,
                contigvol,
                bpf_order,
                motion_filter_order,
                motion_filter_type,
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
                custom_conf,
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

    """
    layout,subj_data= collect_data(bids_dir=fmriprep_dir,participant_label=subject_id, 
                                               task=task_id,bids_validate=False, 
                                               template=brain_template)
    regfile = select_registrationfile(subj_data=subj_data,template=brain_template)
    subject_data = select_cifti_bold(subj_data=subj_data)
    t1wseg =extract_t1w_seg(subj_data=subj_data)
    
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['custom_conf','mni_to_t1w','t1w','t1seg']),
        name='inputnode')
    inputnode.inputs.custom_conf = custom_conf
    inputnode.inputs.t1w = t1wseg[0]
    inputnode.inputs.t1seg = t1wseg[1]
    mni_to_t1w = regfile[0]
    inputnode.inputs.mni_to_t1w = mni_to_t1w

    workflow = Workflow(name=name)
    
    workflow.__desc__ = """
### Post-processing of fMRIPrep outputs
The eXtensible Connectivity Pipeline (XCP) [@mitigating_2018;@satterthwaite_2013]
was used to post-process the outputs of fMRIPrep version {fvers} [@fmriprep1].
XCP was built with *Nipype* {nipype_ver} [@nipype1].
""".format(nipype_ver=nipype_ver,fvers=getfmriprepv(fmriprepdir=fmriprep_dir))


    workflow.__postdesc__ = """


Many internal operations of *XCP* use *Nibabel* [@nilearn], *numpy* 
[@harris2020array], and  *scipy* [@2020SciPy-NMeth]. For more details, 
see the *xcp_abcd* website https://xcp-abcd.readthedocs.io.


#### Copyright Waiver
The above methods descrip text was automatically generated by *XCP*
with the express intention that users should copy and paste this
text into their manuscripts *unchanged*.
It is released under the [CC0]\
(https://creativecommons.org/publicdomain/zero/1.0/) license.

#### References

"""

    summary = pe.Node(SubjectSummary(subject_id=subject_id,bold=subject_data[0]),
                      name='summary', run_without_submitting=True)

    about = pe.Node(AboutSummary(version=__version__,
                                 command=' '.join(sys.argv)),
                    name='about', run_without_submitting=True)

    
    ds_report_summary = pe.Node(
             DerivativesDataSink(base_directory=output_dir,source_file=subject_data[0][0],desc='summary', datatype="figures"),
                  name='ds_report_summary', run_without_submitting=True)

    
    anatomical_wf = init_anatomical_wf(omp_nthreads=omp_nthreads,bids_dir=fmriprep_dir,
                                        subject_id=subject_id,output_dir=output_dir,
                                        t1w_to_mni=regfile[1])

    ## send t1w and t1seg to anatomical workflow

    workflow.connect([ 
          (inputnode,anatomical_wf,[('t1w','inputnode.t1w'),('t1seg','inputnode.t1seg')]),
      ])

    if cifti:
        ii = 0
        for cifti_file in subject_data[1]:
            ii = ii+1
            custom_confx = get_customfile(custom_conf=custom_conf,bold_file=cifti_file)
            cifti_postproc_wf = init_ciftipostprocess_wf(cifti_file=cifti_file,
                                                        lower_bpf=lower_bpf,
                                                        upper_bpf=upper_bpf,
                                                        contigvol=contigvol,
                                                        bpf_order=bpf_order,
                                                        motion_filter_order=motion_filter_order,
                                                        motion_filter_type=motion_filter_type,
                                                        band_stop_min=band_stop_min,
                                                        band_stop_max=band_stop_max,
                                                        smoothing=smoothing,
                                                        params=params,
                                                        head_radius=head_radius,
                                                        custom_conf=custom_confx,
                                                        omp_nthreads=omp_nthreads,
                                                        num_cifti=len(subject_data[1]),
                                                        dummytime=dummytime,
                                                        fd_thresh=fd_thresh,
                                                        despike=despike,
                                                        layout=layout,
                                                        mni_to_t1w=regfile[0],
                                                        output_dir=output_dir,
                                                        name='cifti_postprocess_'+ str(ii) + '_wf')
            ds_report_about = pe.Node(
            DerivativesDataSink(base_directory=output_dir, source_file=cifti_file, desc='about', datatype="figures",),
              name='ds_report_about', run_without_submitting=True)
            workflow.connect([
                  (inputnode,cifti_postproc_wf,[('custom_conf','inputnode.custom_conf'),
                              ('t1w','inputnode.t1w'),('t1seg','inputnode.t1seg')]),
            
            ])

            
    else:
        ii = 0
        for bold_file in subject_data[0]:
            ii = ii+1
            custom_confx = get_customfile(custom_conf=custom_conf,bold_file=bold_file)
            bold_postproc_wf = init_boldpostprocess_wf(bold_file=bold_file,
                                                       lower_bpf=lower_bpf,
                                                       upper_bpf=upper_bpf,
                                                       contigvol=contigvol,
                                                       bpf_order=bpf_order,
                                                       motion_filter_order=motion_filter_order,
                                                       motion_filter_type=motion_filter_type,
                                                       band_stop_min=band_stop_min,
                                                       band_stop_max=band_stop_max,
                                                       smoothing=smoothing,
                                                       params=params,
                                                       head_radius=head_radius,
                                                       omp_nthreads=omp_nthreads,
                                                       brain_template='MNI152NLin2009cAsym',
                                                       num_bold=len(subject_data[0]),
                                                       custom_conf=custom_confx,
                                                       layout=layout,
                                                       despike=despike,
                                                       dummytime=dummytime,
                                                       fd_thresh=fd_thresh,
                                                       output_dir=output_dir,
                                                       mni_to_t1w = mni_to_t1w,
                                                       name='bold_postprocess_'+ str(ii) + '_wf')
            ds_report_about = pe.Node(
             DerivativesDataSink(base_directory=output_dir, source_file=bold_file, desc='about', datatype="figures",),
              name='ds_report_about', run_without_submitting=True)
            workflow.connect([
                  (inputnode,bold_postproc_wf,[ ('mni_to_t1w','inputnode.mni_to_t1w'),
                                   ('t1w','inputnode.t1w'),('t1seg','inputnode.t1seg')]),
                           
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

def getfmriprepv(fmriprepdir):

    datax = glob.glob(fmriprepdir+'/dataset_description.json')[0]

    if datax:
        with open(datax) as f:
            datay = json.load(f)
        
        fvers = datay['GeneratedBy'][0]['Version']
    else:
        fvers = str('Unknown vers')
    
    return fvers
        