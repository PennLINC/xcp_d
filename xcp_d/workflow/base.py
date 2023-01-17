# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The primary workflows for xcp_d."""
import os
import sys
from copy import deepcopy

import nibabel as nb
import numpy as np
import scipy
import templateflow
from nipype import __version__ as nipype_ver
from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.__about__ import __version__
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.report import AboutSummary, SubjectSummary
from xcp_d.utils.bids import (
    collect_data,
    collect_surface_data,
    get_preproc_pipeline_info,
    write_dataset_description,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.workflow.anatomical import init_anatomical_wf, init_t1w_wf
from xcp_d.workflow.bold import init_boldpostprocess_wf
from xcp_d.workflow.cifti import init_ciftipostprocess_wf
from xcp_d.workflow.execsummary import init_brainsprite_figures_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_xcpd_wf(
    fmri_dir,
    output_dir,
    work_dir,
    subject_list,
    analysis_level,
    task_id,
    bids_filters,
    bandpass_filter,
    lower_bpf,
    upper_bpf,
    bpf_order,
    fd_thresh,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    despike,
    head_radius,
    params,
    smoothing,
    custom_confounds_folder,
    dummytime,
    dummy_scans,
    cifti,
    omp_nthreads,
    layout=None,
    process_surfaces=False,
    dcan_qc=False,
    input_type="fmriprep",
    name="xcpd_wf",
):
    """Build and organize execution of xcp_d pipeline.

    It also connects the subworkflows under the xcp_d workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            import os
            import tempfile

            from xcp_d.workflow.base import init_xcpd_wf
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()
            out_dir = tempfile.mkdtemp()

            # Create xcp_d derivatives folder.
            os.mkdir(os.path.join(out_dir, "xcp_d"))

            wf = init_xcpd_wf(
                fmri_dir=fmri_dir,
                output_dir=out_dir,
                work_dir=".",
                subject_list=["01"],
                analysis_level="participant",
                task_id="imagery",
                bids_filters=None,
                bandpass_filter=True,
                lower_bpf=0.01,
                upper_bpf=0.08,
                bpf_order=2,
                fd_thresh=0.2,
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                despike=True,
                head_radius=50.,
                params="36P",
                smoothing=6,
                custom_confounds_folder=None,
                dummytime=0,
                dummy_scans=0,
                cifti=False,
                omp_nthreads=1,
                layout=None,
                process_surfaces=False,
                dcan_qc=False,
                input_type="fmriprep",
                name="xcpd_wf",
            )

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        BIDS dataset layout or None.
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
    bids_filters : dict or None
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
    custom_confounds_folder : str or None
        Path to custom nuisance regressors.
        Must be a folder containing confounds files,
        in which case the file with the name matching the fMRIPrep confounds file will be selected.
    %(dummytime)s
    %(dummy_scans)s
    %(process_surfaces)s
    dcan_qc : bool
        Whether to run DCAN QC or not.
    %(input_type)s
    %(name)s

    References
    ----------
    .. footbibliography::
    """
    xcpd_wf = Workflow(name="xcpd_wf")
    xcpd_wf.base_dir = work_dir
    LOGGER.info(f"Beginning the {name} workflow")

    write_dataset_description(fmri_dir, os.path.join(output_dir, "xcp_d"))

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
            bids_filters=bids_filters,
            smoothing=smoothing,
            output_dir=output_dir,
            dummytime=dummytime,
            dummy_scans=dummy_scans,
            custom_confounds_folder=custom_confounds_folder,
            fd_thresh=fd_thresh,
            process_surfaces=process_surfaces,
            dcan_qc=dcan_qc,
            input_type=input_type,
            name=f"single_subject_{subject_id}_wf",
        )

        single_subj_wf.config["execution"]["crashdump_dir"] = os.path.join(
            output_dir, "xcp_d", "sub-" + subject_id, "log"
        )
        for node in single_subj_wf._get_all_nodes():
            node.config = deepcopy(single_subj_wf.config)
        print(f"Analyzing data at the {analysis_level} level")
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
    dummy_scans,
    fd_thresh,
    task_id,
    bids_filters,
    smoothing,
    custom_confounds_folder,
    process_surfaces,
    dcan_qc,
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
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()

            wf = init_subject_wf(
                fmri_dir=fmri_dir,
                output_dir=".",
                subject_id="01",
                task_id="imagery",
                bids_filters=None,
                bandpass_filter=True,
                lower_bpf=0.01,
                upper_bpf=0.08,
                bpf_order=2,
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                cifti=False,
                despike=True,
                head_radius=50,
                params="36P",
                dummytime=0,
                dummy_scans=0,
                fd_thresh=0.2,
                smoothing=6.,
                custom_confounds_folder=None,
                process_surfaces=False,
                omp_nthreads=1,
                layout=None,
                dcan_qc=False,
                input_type="fmriprep",
                name="single_subject_sub-01_wf",
            )

    Parameters
    ----------
    layout : BIDSLayout object
        BIDS dataset layout or None.
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
    bids_filters : dict or None
    low_mem : bool
        Write uncompressed .nii files in some cases to reduce memory usage
    %(output_dir)s
    %(fd_thresh)s
    %(head_radius)s
    %(params)s
    %(smoothing)s
    custom_confounds_folder : str or None
        Path to custom nuisance regressors.
        Must be a folder containing confounds files,
        in which case the file with the name matching the fMRIPrep confounds file will be selected.
    %(dummytime)s
    %(dummy_scans)s
    %(process_surfaces)s
    dcan_qc : bool
        Whether to run DCAN QC or not.
    %(subject_id)s
    %(input_type)s
    %(name)s

    References
    ----------
    .. footbibliography::
    """
    layout, subj_data = collect_data(
        bids_dir=fmri_dir,
        input_type=input_type,
        participant_label=subject_id,
        task=task_id,
        bids_filters=bids_filters,
        bids_validate=False,
        cifti=cifti,
        layout=layout,
    )

    surface_data, _, surfaces_found = collect_surface_data(
        layout=layout,
        participant_label=subject_id,
    )

    # determine the appropriate post-processing workflow
    postproc_wf_function = init_ciftipostprocess_wf if cifti else init_boldpostprocess_wf
    preproc_files = subj_data["bold"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subj_data",  # not currently used, but will be in future
                "t1w",
                "t1w_mask",  # not used by cifti workflow
                "t1w_seg",
                "template_to_t1w_xform",  # not used by cifti workflow
                "t1w_to_template_xform",
                # surface files
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_smoothwm_surf",
                "rh_smoothwm_surf",
                # hcp-style surface files
                "lh_midthickness_surf",
                "rh_midthickness_surf",
                "lh_inflated_surf",
                "rh_inflated_surf",
                "lh_vinflated_surf",
                "rh_vinflated_surf",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.subj_data = subj_data
    inputnode.inputs.t1w = subj_data["t1w"]
    inputnode.inputs.t1w_mask = subj_data["t1w_mask"]
    inputnode.inputs.t1w_seg = subj_data["t1w_seg"]
    inputnode.inputs.template_to_t1w_xform = subj_data["template_to_t1w_xform"]
    inputnode.inputs.t1w_to_template_xform = subj_data["t1w_to_template_xform"]

    # surface files (required for brainsprite/warp workflows)
    inputnode.inputs.lh_pial_surf = surface_data["lh_pial_surf"]
    inputnode.inputs.rh_pial_surf = surface_data["rh_pial_surf"]
    inputnode.inputs.lh_smoothwm_surf = surface_data["lh_smoothwm_surf"]
    inputnode.inputs.rh_smoothwm_surf = surface_data["rh_smoothwm_surf"]

    # optional surface files
    inputnode.inputs.lh_midthickness_surf = surface_data["lh_midthickness_surf"]
    inputnode.inputs.rh_midthickness_surf = surface_data["rh_midthickness_surf"]
    inputnode.inputs.lh_inflated_surf = surface_data["lh_inflated_surf"]
    inputnode.inputs.rh_inflated_surf = surface_data["rh_inflated_surf"]
    inputnode.inputs.lh_inflated_surf = surface_data["lh_vinflated_surf"]
    inputnode.inputs.rh_inflated_surf = surface_data["rh_vinflated_surf"]

    workflow = Workflow(name=name)

    info_dict = get_preproc_pipeline_info(input_type=input_type, fmri_dir=fmri_dir)

    workflow.__desc__ = f"""
### Post-processing of {input_type} outputs
The eXtensible Connectivity Pipeline (XCP) [@mitigating_2018;@satterthwaite_2013]
was used to post-process the outputs of {info_dict["name"]} version {info_dict["version"]}
{info_dict["references"]}.
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

    summary = pe.Node(
        SubjectSummary(subject_id=subject_id, bold=preproc_files),
        name="summary",
    )

    about = pe.Node(
        AboutSummary(version=__version__, command=" ".join(sys.argv)),
        name="about",
    )

    ds_report_summary = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=preproc_files[0],
            desc="summary",
            datatype="figures",
        ),
        name="ds_report_summary",
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=preproc_files[0],
            desc="about",
            datatype="figures",
        ),
        name="ds_report_about",
        run_without_submitting=True,
    )

    t1w_wf = init_t1w_wf(
        output_dir=output_dir,
        input_type=input_type,
        omp_nthreads=omp_nthreads,
        mem_gb=5,  # RF: need to change memory size
    )

    # fmt:off
    workflow.connect([
        (inputnode, t1w_wf, [('t1w', 'inputnode.t1w'),
                             ('t1w_seg', 'inputnode.t1seg'),
                             ('t1w_to_template_xform', 'inputnode.t1w_to_template')]),
    ])
    # fmt:on

    if surfaces_found:
        # Plot the white and pial surfaces on the brain in a brainsprite figure.
        brainsprite_wf = init_brainsprite_figures_wf(
            output_dir=output_dir,
            t2w_available=False,
            omp_nthreads=omp_nthreads,
            mem_gb=5,
        )

    if process_surfaces and surfaces_found:
        anatomical_wf = init_anatomical_wf(
            layout=layout,
            fmri_dir=fmri_dir,
            subject_id=subject_id,
            output_dir=output_dir,
            input_type=input_type,
            omp_nthreads=omp_nthreads,
            mem_gb=5,  # RF: need to change memory size
        )

        # Use standard-space T1w and surfaces for brainsprite.
        # fmt:off
        workflow.connect([
            (inputnode, anatomical_wf, [
                ("t1w", "inputnode.t1w"),
                ("t1w_seg", "inputnode.t1seg"),
            ]),
            (t1w_wf, brainsprite_wf, [
                ("outputnode.t1w", "inputnode.t1w"),
            ]),
            (anatomical_wf, brainsprite_wf, [
                ("outputnode.lh_wm_surf", "inputnode.lh_smoothwm_surf"),
                ("outputnode.rh_wm_surf", "inputnode.rh_smoothwm_surf"),
                ("outputnode.lh_pial_surf", "inputnode.lh_pial_surf"),
                ("outputnode.rh_pial_surf", "inputnode.rh_pial_surf"),
            ]),
        ])
        # fmt:on

    elif surfaces_found:
        # Use native-space T1w and surfaces for brainsprite.
        # fmt:off
        workflow.connect([
            (inputnode, brainsprite_wf, [
                ("t1w", "inputnode.t1w"),
                ("lh_smoothwm_surf", "inputnode.lh_smoothwm_surf"),
                ("rh_smoothwm_surf", "inputnode.rh_smoothwm_surf"),
                ("lh_pial_surf", "inputnode.lh_pial_surf"),
                ("rh_pial_surf", "inputnode.rh_pial_surf"),
            ]),
        ])
        # fmt:on

    elif process_surfaces:
        raise ValueError(
            "No surfaces found. "
            "Surfaces are required if `--warp-surfaces-native2std` is enabled."
        )

    # loop over each bold run to be postprocessed
    # NOTE: Look at https://miykael.github.io/nipype_tutorial/notebooks/basic_iteration.html
    # for hints on iteration
    for i_run, bold_file in enumerate(preproc_files):
        bold_postproc_wf = postproc_wf_function(
            input_type=input_type,
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
            custom_confounds_folder=custom_confounds_folder,
            layout=layout,
            despike=despike,
            dummytime=dummytime,
            dummy_scans=dummy_scans,
            fd_thresh=fd_thresh,
            dcan_qc=dcan_qc,
            output_dir=output_dir,
            name=f"{'cifti' if cifti else 'nifti'}_postprocess_{i_run}_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, bold_postproc_wf, [
                ('t1w', 'inputnode.t1w'),
                ('t1w_seg', 'inputnode.t1seg'),
                ('t1w_mask', 'inputnode.t1w_mask'),
            ]),
        ])
        if not cifti:
            workflow.connect([
                (inputnode, bold_postproc_wf, [
                    ('template_to_t1w_xform', 'inputnode.template_to_t1w'),
                ]),
            ])
        # fmt:on

    # fmt:off
    workflow.connect([(summary, ds_report_summary, [('out_report', 'in_file')]),
                      (about, ds_report_about, [('out_report', 'in_file')])])
    # fmt:on

    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).interface.out_path_base = "xcp_d"

    return workflow
