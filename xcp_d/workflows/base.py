# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""The primary workflows for xcp_d."""
import os
import sys
from copy import deepcopy

import bids
import matplotlib
import nibabel as nb
import nilearn
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
    _get_tr,
    collect_data,
    collect_run_data,
    collect_surface_data,
    get_entity,
    get_preproc_pipeline_info,
    group_across_runs,
    write_dataset_description,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.modified_data import flag_bad_run
from xcp_d.utils.utils import estimate_brain_radius
from xcp_d.workflows.anatomical import (
    init_postprocess_anat_wf,
    init_postprocess_surfaces_wf,
)
from xcp_d.workflows.bold import init_postprocess_nifti_wf
from xcp_d.workflows.cifti import init_postprocess_cifti_wf
from xcp_d.workflows.concatenation import init_concatenate_data_wf

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
    high_pass,
    low_pass,
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
    dummy_scans,
    cifti,
    omp_nthreads,
    layout=None,
    process_surfaces=False,
    dcan_qc=False,
    input_type="fmriprep",
    min_coverage=0.5,
    min_time=100,
    combineruns=False,
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

            from xcp_d.workflows.base import init_xcpd_wf
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
                high_pass=0.01,
                low_pass=0.08,
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
                dummy_scans=0,
                cifti=False,
                omp_nthreads=1,
                layout=None,
                process_surfaces=False,
                dcan_qc=False,
                input_type="fmriprep",
                min_coverage=0.5,
                min_time=100,
                combineruns=False,
                name="xcpd_wf",
            )

    Parameters
    ----------
    %(layout)s
    %(bandpass_filter)s
    %(high_pass)s
    %(low_pass)s
    %(despike)s
    %(bpf_order)s
    %(analysis_level)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(omp_nthreads)s
    %(cifti)s
    task_id : :obj:`str` or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    bids_filters : dict or None
    %(output_dir)s
    %(fd_thresh)s
    run_uuid : :obj:`str`
        Unique identifier for execution instance
    subject_list : list
        List of subject labels
    %(work_dir)s
    %(head_radius)s
    %(params)s
    %(smoothing)s
    %(custom_confounds_folder)s
    %(dummy_scans)s
    %(process_surfaces)s
    %(dcan_qc)s
    %(input_type)s
    %(min_coverage)s
    %(min_time)s
    combineruns
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
            high_pass=high_pass,
            low_pass=low_pass,
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
            dummy_scans=dummy_scans,
            custom_confounds_folder=custom_confounds_folder,
            fd_thresh=fd_thresh,
            process_surfaces=process_surfaces,
            dcan_qc=dcan_qc,
            input_type=input_type,
            min_coverage=min_coverage,
            min_time=min_time,
            combineruns=combineruns,
            name=f"single_subject_{subject_id}_wf",
        )

        single_subj_wf.config["execution"]["crashdump_dir"] = os.path.join(
            output_dir,
            "xcp_d",
            f"sub-{subject_id}",
            "log",
        )
        for node in single_subj_wf._get_all_nodes():
            node.config = deepcopy(single_subj_wf.config)
        print(f"Analyzing data at the {analysis_level} level")
        xcpd_wf.add_nodes([single_subj_wf])

    return xcpd_wf


@fill_doc
def init_subject_wf(
    fmri_dir,
    subject_id,
    input_type,
    process_surfaces,
    combineruns,
    cifti,
    task_id,
    bids_filters,
    bandpass_filter,
    high_pass,
    low_pass,
    bpf_order,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    smoothing,
    head_radius,
    params,
    output_dir,
    custom_confounds_folder,
    dummy_scans,
    fd_thresh,
    despike,
    dcan_qc,
    min_coverage,
    min_time,
    omp_nthreads,
    layout,
    name,
):
    """Organize the postprocessing pipeline for a single subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.base import init_subject_wf
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()

            wf = init_subject_wf(
                fmri_dir=fmri_dir,
                subject_id="01",
                input_type="fmriprep",
                process_surfaces=False,
                combineruns=False,
                cifti=False,
                task_id="imagery",
                bids_filters=None,
                bandpass_filter=True,
                high_pass=0.01,
                low_pass=0.08,
                bpf_order=2,
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                smoothing=6.,
                head_radius=50,
                params="36P",
                output_dir=".",
                custom_confounds_folder=None,
                dummy_scans=0,
                fd_thresh=0.2,
                despike=True,
                dcan_qc=False,
                min_coverage=0.5,
                min_time=100,
                omp_nthreads=1,
                layout=None,
                name="single_subject_sub-01_wf",
            )

    Parameters
    ----------
    %(fmri_dir)s
    %(subject_id)s
    %(input_type)s
    %(process_surfaces)s
    combineruns
    %(cifti)s
    task_id : :obj:`str` or None
        Task ID of BOLD  series to be selected for postprocess , or ``None`` to postprocess all
    bids_filters : dict or None
    %(bandpass_filter)s
    %(high_pass)s
    %(low_pass)s
    %(bpf_order)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(smoothing)s
    %(head_radius)s
    %(params)s
    %(output_dir)s
    %(custom_confounds_folder)s
    %(dummy_scans)s
    %(fd_thresh)s
    %(despike)s
    %(dcan_qc)s
    %(min_coverage)s
    %(min_time)s
    %(omp_nthreads)s
    %(layout)s
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
    t1w_available = subj_data["t1w"] is not None
    t2w_available = subj_data["t2w"] is not None
    primary_anat = "T1w" if subj_data["t1w"] else "T2w"

    mesh_available, shape_available, standard_space_mesh, surface_data = collect_surface_data(
        layout=layout,
        participant_label=subject_id,
    )

    # determine the appropriate post-processing workflow
    init_postprocess_bold_wf = init_postprocess_cifti_wf if cifti else init_postprocess_nifti_wf
    preproc_files = subj_data["bold"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "subj_data",  # not currently used, but will be in future
                "t1w",
                "t2w",  # optional
                "anat_brainmask",  # not used by cifti workflow
                "anat_dseg",
                "template_to_anat_xfm",  # not used by cifti workflow
                "anat_to_template_xfm",
                # mesh files
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
                # shape files
                "lh_sulcal_depth",
                "rh_sulcal_depth",
                "lh_sulcal_curv",
                "rh_sulcal_curv",
                "lh_cortical_thickness",
                "rh_cortical_thickness",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.subj_data = subj_data
    inputnode.inputs.t1w = subj_data["t1w"]
    inputnode.inputs.t2w = subj_data["t2w"]
    inputnode.inputs.anat_brainmask = subj_data["anat_brainmask"]
    inputnode.inputs.anat_dseg = subj_data["anat_dseg"]
    inputnode.inputs.template_to_anat_xfm = subj_data["template_to_anat_xfm"]
    inputnode.inputs.anat_to_template_xfm = subj_data["anat_to_template_xfm"]

    # surface mesh files (required for brainsprite/warp workflows)
    inputnode.inputs.lh_pial_surf = surface_data["lh_pial_surf"]
    inputnode.inputs.rh_pial_surf = surface_data["rh_pial_surf"]
    inputnode.inputs.lh_wm_surf = surface_data["lh_wm_surf"]
    inputnode.inputs.rh_wm_surf = surface_data["rh_wm_surf"]

    # optional surface shape files (used by surface-warping workflow)
    inputnode.inputs.lh_sulcal_depth = surface_data["lh_sulcal_depth"]
    inputnode.inputs.rh_sulcal_depth = surface_data["rh_sulcal_depth"]
    inputnode.inputs.lh_sulcal_curv = surface_data["lh_sulcal_curv"]
    inputnode.inputs.rh_sulcal_curv = surface_data["rh_sulcal_curv"]
    inputnode.inputs.lh_cortical_thickness = surface_data["lh_cortical_thickness"]
    inputnode.inputs.rh_cortical_thickness = surface_data["rh_cortical_thickness"]

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
*AFNI* [@cox1996afni;@cox1997software],
{"*Connectome Workbench* [@marcus2011informatics], " if cifti else ""}*ANTS* [@avants2009advanced],
*TemplateFlow* version {templateflow.__version__} [@ciric2022templateflow],
*matplotlib* version {matplotlib.__version__} [@hunter2007matplotlib],
*Nibabel* version {nb.__version__} [@brett_matthew_2022_6658382],
*Nilearn* version {nilearn.__version__} [@abraham2014machine],
*numpy* version {np.__version__} [@harris2020array],
*pybids* version {bids.__version__} [@yarkoni2019pybids],
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
            datatype="figures",
            desc="summary",
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

    # Extract target volumetric space for T1w image
    target_space = get_entity(subj_data["anat_to_template_xfm"], "to")

    postprocess_anat_wf = init_postprocess_anat_wf(
        output_dir=output_dir,
        input_type=input_type,
        t1w_available=t1w_available,
        t2w_available=t2w_available,
        target_space=target_space,
        dcan_qc=dcan_qc,
        omp_nthreads=omp_nthreads,
        mem_gb=1,
        name="postprocess_anat_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, postprocess_anat_wf, [
            ("t1w", "inputnode.t1w"),
            ("t2w", "inputnode.t2w"),
            ("anat_dseg", "inputnode.anat_dseg"),
            ("anat_to_template_xfm", "inputnode.anat_to_template_xfm"),
        ]),
    ])
    # fmt:on

    if process_surfaces or (dcan_qc and mesh_available):
        # Run surface post-processing workflow if we want to warp meshes to standard space *or*
        # generate brainsprite.
        postprocess_surfaces_wf = init_postprocess_surfaces_wf(
            fmri_dir=fmri_dir,
            subject_id=subject_id,
            dcan_qc=dcan_qc,
            mesh_available=mesh_available,
            standard_space_mesh=standard_space_mesh,
            shape_available=shape_available,
            process_surfaces=process_surfaces,
            output_dir=output_dir,
            t1w_available=t1w_available,
            t2w_available=t2w_available,
            mem_gb=1,
            omp_nthreads=omp_nthreads,
            name="postprocess_surfaces_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, postprocess_surfaces_wf, [
                ("lh_pial_surf", "inputnode.lh_pial_surf"),
                ("rh_pial_surf", "inputnode.rh_pial_surf"),
                ("lh_wm_surf", "inputnode.lh_wm_surf"),
                ("rh_wm_surf", "inputnode.rh_wm_surf"),
                ("anat_to_template_xfm", "inputnode.anat_to_template_xfm"),
                ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
                ("lh_sulcal_depth", "inputnode.lh_sulcal_depth"),
                ("rh_sulcal_depth", "inputnode.rh_sulcal_depth"),
                ("lh_sulcal_curv", "inputnode.lh_sulcal_curv"),
                ("rh_sulcal_curv", "inputnode.rh_sulcal_curv"),
                ("lh_cortical_thickness", "inputnode.lh_cortical_thickness"),
                ("rh_cortical_thickness", "inputnode.rh_cortical_thickness"),
            ]),
            (postprocess_anat_wf, postprocess_surfaces_wf, [
                ("outputnode.t1w", "inputnode.t1w"),
                ("outputnode.t2w", "inputnode.t2w"),
            ]),
        ])
        # fmt:on

    # Estimate head radius, if necessary
    head_radius = estimate_brain_radius(
        mask_file=subj_data["anat_brainmask"],
        head_radius=head_radius,
    )

    n_runs = len(preproc_files)
    preproc_files = group_across_runs(preproc_files)
    run_counter = 0
    for ent_set, task_files in enumerate(preproc_files):
        # Assuming TR is constant across runs for a given combination of entities.
        TR = _get_tr(nb.load(task_files[0]))

        n_task_runs = len(task_files)
        if combineruns and (n_task_runs > 1):
            merge_elements = [
                "name_source",
                "preprocessed_bold",
                "fmriprep_confounds_file",
                "filtered_motion",
                "temporal_mask",
                "uncensored_denoised_bold",
                "interpolated_filtered_bold",
                "censored_denoised_bold",
                "smoothed_denoised_bold",
                "anat_to_native_xfm",
                "bold_mask",
                "boldref",
                "atlas_names",  # this will be exactly the same across runs
                "timeseries",
                "timeseries_ciftis",
            ]
            merge_dict = {
                io_name: pe.Node(
                    niu.Merge(n_task_runs, no_flatten=True),
                    name=f"collect_{io_name}_{ent_set}",
                )
                for io_name in merge_elements
            }

        for j_run, bold_file in enumerate(task_files):
            run_data = collect_run_data(
                layout,
                input_type,
                bold_file,
                cifti=cifti,
                primary_anat=primary_anat,
            )

            post_scrubbing_duration = flag_bad_run(
                fmriprep_confounds_file=run_data["confounds"],
                dummy_scans=dummy_scans,
                TR=run_data["bold_metadata"]["RepetitionTime"],
                motion_filter_type=motion_filter_type,
                motion_filter_order=motion_filter_order,
                band_stop_min=band_stop_min,
                band_stop_max=band_stop_max,
                head_radius=head_radius,
                fd_thresh=fd_thresh,
            )
            if (min_time >= 0) and (post_scrubbing_duration < min_time):
                LOGGER.warning(
                    f"Less than {min_time} seconds in {bold_file} survive high-motion outlier "
                    f"scrubbing ({post_scrubbing_duration}). "
                    "This run will not be processed."
                )
                continue

            postprocess_bold_wf = init_postprocess_bold_wf(
                bold_file=bold_file,
                bandpass_filter=bandpass_filter,
                high_pass=high_pass,
                low_pass=low_pass,
                bpf_order=bpf_order,
                motion_filter_type=motion_filter_type,
                motion_filter_order=motion_filter_order,
                band_stop_min=band_stop_min,
                band_stop_max=band_stop_max,
                smoothing=smoothing,
                head_radius=head_radius,
                params=params,
                output_dir=output_dir,
                custom_confounds_folder=custom_confounds_folder,
                dummy_scans=dummy_scans,
                fd_thresh=fd_thresh,
                despike=despike,
                dcan_qc=dcan_qc,
                run_data=run_data,
                t1w_available=t1w_available,
                t2w_available=t2w_available,
                n_runs=n_runs,
                min_coverage=min_coverage,
                omp_nthreads=omp_nthreads,
                layout=layout,
                name=f"{'cifti' if cifti else 'nifti'}_postprocess_{run_counter}_wf",
            )
            run_counter += 1

            # fmt:off
            workflow.connect([
                (postprocess_anat_wf, postprocess_bold_wf, [
                    ("outputnode.t1w", "inputnode.t1w"),
                    ("outputnode.t2w", "inputnode.t2w"),
                ]),
            ])
            # fmt:on

            if not cifti:
                # fmt:off
                workflow.connect([
                    (inputnode, postprocess_bold_wf, [
                        ("anat_brainmask", "inputnode.anat_brainmask"),
                        ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
                    ]),
                ])
                # fmt:on

            if combineruns and (n_task_runs > 1):
                for io_name, node in merge_dict.items():
                    # fmt:off
                    workflow.connect([
                        (postprocess_bold_wf, node, [(f"outputnode.{io_name}", f"in{j_run + 1}")]),
                    ])
                    # fmt:on

        if combineruns and (n_task_runs > 1):
            concatenate_data_wf = init_concatenate_data_wf(
                output_dir=output_dir,
                motion_filter_type=motion_filter_type,
                mem_gb=1,
                omp_nthreads=omp_nthreads,
                TR=TR,
                head_radius=head_radius,
                smoothing=smoothing,
                cifti=cifti,
                dcan_qc=dcan_qc,
                name=f"concatenate_entity_set_{ent_set}_wf",
            )

            # fmt:off
            workflow.connect([
                (inputnode, concatenate_data_wf, [
                    ("anat_brainmask", "inputnode.anat_brainmask"),
                    ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
                ]),
            ])
            # fmt:on

            for io_name, node in merge_dict.items():
                # fmt:off
                workflow.connect([(node, concatenate_data_wf, [("out", f"inputnode.{io_name}")])])
                # fmt:on

    # fmt:off
    workflow.connect([
        (summary, ds_report_summary, [("out_report", "in_file")]),
        (about, ds_report_about, [("out_report", "in_file")]),
    ])
    # fmt:on

    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).interface.out_path_base = "xcp_d"

    return workflow
