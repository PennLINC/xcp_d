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
from packaging.version import Version

from xcp_d import config
from xcp_d.__about__ import __version__
from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.report import AboutSummary, SubjectSummary
from xcp_d.utils.bids import (
    _get_tr,
    collect_data,
    collect_mesh_data,
    collect_morphometry_data,
    collect_run_data,
    get_entity,
    get_preproc_pipeline_info,
    group_across_runs,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.modified_data import calculate_exact_scans, flag_bad_run
from xcp_d.utils.utils import estimate_brain_radius
from xcp_d.workflows.anatomical.parcellation import init_parcellate_surfaces_wf
from xcp_d.workflows.anatomical.surface import init_postprocess_surfaces_wf
from xcp_d.workflows.anatomical.volume import init_postprocess_anat_wf
from xcp_d.workflows.bold.cifti import init_postprocess_cifti_wf
from xcp_d.workflows.bold.concatenation import init_concatenate_data_wf
from xcp_d.workflows.bold.nifti import init_postprocess_nifti_wf
from xcp_d.workflows.parcellation import init_load_atlases_wf

LOGGER = logging.getLogger("nipype.workflow")


def init_xcpd_wf():
    """Build XCP-D's pipeline.

    This workflow organizes the execution of XCP-D, with a sub-workflow for
    each subject.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.base import init_xcpd_wf

            with mock_config():
                wf = init_xcpd_wf()

    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    ver = Version(config.environment.version)

    xcpd_wf = Workflow(name=f"xcp_d_{ver.major}_{ver.minor}_wf")
    xcpd_wf.base_dir = config.execution.work_dir

    for subject_id in config.execution.participant_label:
        single_subject_wf = init_single_subject_wf(subject_id)

        single_subject_wf.config["execution"]["crashdump_dir"] = str(
            config.execution.xcp_d_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        for node in single_subject_wf._get_all_nodes():
            node.config = deepcopy(single_subject_wf.config)

        xcpd_wf.add_nodes([single_subject_wf])

        # Dump a copy of the config file into the log directory
        log_dir = (
            config.execution.xcp_d_dir / f"sub-{subject_id}" / "log" / config.execution.run_uuid
        )
        log_dir.mkdir(exist_ok=True, parents=True)
        config.to_filename(log_dir / "xcp_d.toml")

    return xcpd_wf


@fill_doc
def init_single_subject_wf(subject_id: str):
    """Organize the postprocessing pipeline for a single subject.

    It collects and reports information about the subject, and prepares
    sub-workflows to perform anatomical and functional postprocessing.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.base import init_single_subject_wf

            with mock_config():
                wf = init_single_subject_wf("01")

    Parameters
    ----------
    subject_id : :obj:`str`
        Subject label for this single-subject workflow.
    """
    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    subj_data = collect_data(
        layout=config.execution.layout,
        participant_label=subject_id,
        bids_filters=config.execution.bids_filters,
        input_type=config.workflow.input_type,
        file_format=config.workflow.file_format,
    )
    t1w_available = subj_data["t1w"] is not None
    t2w_available = subj_data["t2w"] is not None
    anat_mod = "t1w" if t1w_available else "t2w"

    mesh_available, standard_space_mesh, software, mesh_files = collect_mesh_data(
        layout=config.execution.layout,
        participant_label=subject_id,
        bids_filters=config.execution.bids_filters,
    )
    morph_file_types, morphometry_files = collect_morphometry_data(
        layout=config.execution.layout,
        participant_label=subject_id,
        bids_filters=config.execution.bids_filters,
    )

    # determine the appropriate post-processing workflow
    workflows = {
        "nifti": init_postprocess_nifti_wf,
        "cifti": init_postprocess_cifti_wf,
    }
    init_postprocess_bold_wf = workflows[config.workflow.file_format]
    preproc_files = subj_data["bold"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t2w",  # optional
                "anat_brainmask",  # used to estimate head radius and for QC metrics
                "anat_dseg",
                "template_to_anat_xfm",  # not used by cifti workflow
                "anat_to_template_xfm",
                # mesh files
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
                "lh_subject_sphere",
                "rh_subject_sphere",
                # morphometry files
                "sulcal_depth",
                "sulcal_curv",
                "cortical_thickness",
                "cortical_thickness_corr",
                "myelin",
                "myelin_smoothed",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.t1w = subj_data["t1w"]
    inputnode.inputs.t2w = subj_data["t2w"]
    inputnode.inputs.anat_brainmask = subj_data["anat_brainmask"]
    inputnode.inputs.anat_dseg = subj_data["anat_dseg"]
    inputnode.inputs.template_to_anat_xfm = subj_data["template_to_anat_xfm"]
    inputnode.inputs.anat_to_template_xfm = subj_data["anat_to_template_xfm"]

    # surface mesh files (required for brainsprite/warp workflows)
    inputnode.inputs.lh_pial_surf = mesh_files["lh_pial_surf"]
    inputnode.inputs.rh_pial_surf = mesh_files["rh_pial_surf"]
    inputnode.inputs.lh_wm_surf = mesh_files["lh_wm_surf"]
    inputnode.inputs.rh_wm_surf = mesh_files["rh_wm_surf"]
    inputnode.inputs.lh_subject_sphere = mesh_files["lh_subject_sphere"]
    inputnode.inputs.rh_subject_sphere = mesh_files["rh_subject_sphere"]

    # optional surface shape files (used by surface-warping workflow)
    inputnode.inputs.sulcal_depth = morphometry_files["sulcal_depth"]
    inputnode.inputs.sulcal_curv = morphometry_files["sulcal_curv"]
    inputnode.inputs.cortical_thickness = morphometry_files["cortical_thickness"]
    inputnode.inputs.cortical_thickness_corr = morphometry_files["cortical_thickness_corr"]
    inputnode.inputs.myelin = morphometry_files["myelin"]
    inputnode.inputs.myelin_smoothed = morphometry_files["myelin_smoothed"]

    workflow = Workflow(name=f"sub_{subject_id}_wf")

    info_dict = get_preproc_pipeline_info(
        input_type=config.workflow.input_type,
        fmri_dir=config.execution.fmri_dir,
    )

    workflow.__desc__ = f"""
### Post-processing of {config.workflow.input_type} outputs
The eXtensible Connectivity Pipeline- DCAN (XCP-D) [@mitigating_2018;@satterthwaite_2013]
was used to post-process the outputs of *{info_dict["name"]}* version {info_dict["version"]}
{info_dict["references"]}.
XCP-D was built with *Nipype* version {nipype_ver} [@nipype1, RRID:SCR_002502].
"""

    cw_str = (
        "*Connectome Workbench* [@marcus2011informatics], "
        if config.workflow.file_format == "cifti"
        else ""
    )
    workflow.__postdesc__ = f"""

Many internal operations of *XCP-D* use
*AFNI* [@cox1996afni;@cox1997software],{cw_str}
*ANTS* [@avants2009advanced],
*TemplateFlow* version {templateflow.__version__} [@ciric2022templateflow],
*matplotlib* version {matplotlib.__version__} [@hunter2007matplotlib],
*Nibabel* version {nb.__version__} [@brett_matthew_2022_6658382],
*Nilearn* version {nilearn.__version__} [@abraham2014machine],
*numpy* version {np.__version__} [@harris2020array],
*pybids* version {bids.__version__} [@yarkoni2019pybids],
and *scipy* version {scipy.__version__} [@2020SciPy-NMeth].
For more details, see the *XCP-D* website (https://xcp-d.readthedocs.io).


#### Copyright Waiver

The above methods description text was automatically generated by *XCP-D*
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
            base_directory=config.execution.xcp_d_dir,
            source_file=preproc_files[0],
            datatype="figures",
            desc="summary",
        ),
        name="ds_report_summary",
    )

    ds_report_about = pe.Node(
        DerivativesDataSink(
            base_directory=config.execution.xcp_d_dir,
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
        t1w_available=t1w_available,
        t2w_available=t2w_available,
        target_space=target_space,
    )

    workflow.connect([
        (inputnode, postprocess_anat_wf, [
            ("t1w", "inputnode.t1w"),
            ("t2w", "inputnode.t2w"),
            ("anat_dseg", "inputnode.anat_dseg"),
            ("anat_to_template_xfm", "inputnode.anat_to_template_xfm"),
        ]),
    ])  # fmt:skip

    # Load the atlases, warping to the same space as the BOLD data if necessary.
    if config.execution.atlases:
        load_atlases_wf = init_load_atlases_wf()
        load_atlases_wf.inputs.inputnode.name_source = preproc_files[0]
        load_atlases_wf.inputs.inputnode.bold_file = preproc_files[0]

    if config.workflow.process_surfaces or (config.workflow.abcc_qc and mesh_available):
        # Run surface post-processing workflow if we want to warp meshes to standard space *or*
        # generate brainsprite.
        postprocess_surfaces_wf = init_postprocess_surfaces_wf(
            mesh_available=mesh_available,
            standard_space_mesh=standard_space_mesh,
            morphometry_files=morph_file_types,
            t1w_available=t1w_available,
            t2w_available=t2w_available,
            software=software,
        )

        workflow.connect([
            (inputnode, postprocess_surfaces_wf, [
                ("lh_pial_surf", "inputnode.lh_pial_surf"),
                ("rh_pial_surf", "inputnode.rh_pial_surf"),
                ("lh_wm_surf", "inputnode.lh_wm_surf"),
                ("rh_wm_surf", "inputnode.rh_wm_surf"),
                ("lh_subject_sphere", "inputnode.lh_subject_sphere"),
                ("rh_subject_sphere", "inputnode.rh_subject_sphere"),
                ("anat_to_template_xfm", "inputnode.anat_to_template_xfm"),
                ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
            ]),
        ])  # fmt:skip

        for morph_file in morph_file_types:
            workflow.connect([
                (inputnode, postprocess_surfaces_wf, [(morph_file, f"inputnode.{morph_file}")]),
            ])  # fmt:skip

        if config.workflow.process_surfaces or standard_space_mesh:
            # Use standard-space structurals
            workflow.connect([
                (postprocess_anat_wf, postprocess_surfaces_wf, [
                    ("outputnode.t1w", "inputnode.t1w"),
                    ("outputnode.t2w", "inputnode.t2w"),
                ]),
            ])  # fmt:skip

        else:
            # Use native-space structurals
            workflow.connect([
                (inputnode, postprocess_surfaces_wf, [
                    ("t1w", "inputnode.t1w"),
                    ("t2w", "inputnode.t2w"),
                ]),
            ])  # fmt:skip

        if morph_file_types and config.execution.atlases:
            # Parcellate the morphometry files
            parcellate_surfaces_wf = init_parcellate_surfaces_wf(
                files_to_parcellate=morph_file_types,
            )

            for morph_file_type in morph_file_types:
                workflow.connect([
                    (inputnode, parcellate_surfaces_wf, [
                        (morph_file_type, f"inputnode.{morph_file_type}"),
                    ]),
                ])  # fmt:skip

    # Estimate head radius, if necessary
    # Need to warp the standard-space brain mask to the anatomical space to estimate head radius
    warp_brainmask = ApplyTransforms(
        input_image=subj_data["anat_brainmask"],
        transforms=[subj_data["template_to_anat_xfm"]],
        reference_image=subj_data[anat_mod],
        num_threads=2,
        interpolation="GenericLabel",
        input_image_type=3,
        dimension=3,
    )
    os.makedirs(config.execution.work_dir / workflow.fullname, exist_ok=True)
    warp_brainmask_results = warp_brainmask.run(
        cwd=(config.execution.work_dir / workflow.fullname),
    )
    anat_brainmask_in_anat_space = warp_brainmask_results.outputs.output_image

    head_radius = estimate_brain_radius(
        mask_file=anat_brainmask_in_anat_space,
        head_radius=config.workflow.head_radius,
    )

    n_runs = len(preproc_files)
    # group files across runs and directions, to facilitate concatenation
    preproc_files = group_across_runs(preproc_files)
    run_counter = 0
    for ent_set, task_files in enumerate(preproc_files):
        # Assuming TR is constant across runs for a given combination of entities.
        TR = _get_tr(nb.load(task_files[0]))

        n_task_runs = len(task_files)
        if config.workflow.combine_runs and (n_task_runs > 1):
            merge_elements = [
                "name_source",
                "preprocessed_bold",
                "fmriprep_confounds_file",
                "filtered_motion",
                "temporal_mask",
                "denoised_bold",
                "denoised_interpolated_bold",
                "censored_denoised_bold",
                "smoothed_denoised_bold",
                "bold_mask",
                "boldref",
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
                layout=config.execution.layout,
                bold_file=bold_file,
                file_format=config.workflow.file_format,
                target_space=target_space,
            )

            post_scrubbing_duration = flag_bad_run(
                fmriprep_confounds_file=run_data["confounds"],
                dummy_scans=config.workflow.dummy_scans,
                TR=run_data["bold_metadata"]["RepetitionTime"],
                motion_filter_type=config.workflow.motion_filter_type,
                motion_filter_order=config.workflow.motion_filter_order,
                band_stop_min=config.workflow.band_stop_min,
                band_stop_max=config.workflow.band_stop_max,
                head_radius=head_radius,
                fd_thresh=config.workflow.fd_thresh,
            )

            if (config.workflow.min_time >= 0) and (
                post_scrubbing_duration < config.workflow.min_time
            ):
                LOGGER.warning(
                    f"Less than {config.workflow.min_time} seconds in "
                    f"{os.path.basename(bold_file)} survive "
                    f"high-motion outlier scrubbing ({post_scrubbing_duration}). "
                    "This run will not be processed."
                )
                continue

            # Reduce exact_times to only include values greater than the post-scrubbing duration.
            exact_scans = []
            if config.workflow.dcan_correlation_lengths:
                exact_scans = calculate_exact_scans(
                    exact_times=config.workflow.dcan_correlation_lengths,
                    scan_length=post_scrubbing_duration,
                    t_r=run_data["bold_metadata"]["RepetitionTime"],
                    bold_file=bold_file,
                )

            postprocess_bold_wf = init_postprocess_bold_wf(
                bold_file=bold_file,
                head_radius=head_radius,
                run_data=run_data,
                t1w_available=t1w_available,
                t2w_available=t2w_available,
                n_runs=n_runs,
                exact_scans=exact_scans,
                name=f"{config.workflow.file_format}_postprocess_{run_counter}_wf",
            )
            run_counter += 1

            workflow.connect([
                (postprocess_anat_wf, postprocess_bold_wf, [
                    ("outputnode.t1w", "inputnode.t1w"),
                    ("outputnode.t2w", "inputnode.t2w"),
                ]),
            ])  # fmt:skip

            if config.workflow.process_surfaces or (config.workflow.abcc_qc and mesh_available):
                workflow.connect([
                    (postprocess_surfaces_wf, postprocess_bold_wf, [
                        ("outputnode.lh_midthickness", "inputnode.lh_midthickness"),
                        ("outputnode.rh_midthickness", "inputnode.rh_midthickness"),
                    ]),
                ])  # fmt:skip

            if config.execution.atlases:
                workflow.connect([
                    (load_atlases_wf, postprocess_bold_wf, [
                        ("outputnode.atlas_files", "inputnode.atlas_files"),
                        ("outputnode.atlas_labels_files", "inputnode.atlas_labels_files"),
                    ]),
                ])  # fmt:skip

            if config.workflow.file_format == "nifti":
                workflow.connect([
                    (inputnode, postprocess_bold_wf, [
                        ("anat_brainmask", "inputnode.anat_brainmask"),
                        ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
                    ]),
                ])  # fmt:skip

                # The post-processing workflow needs a native anatomical-space image as a reference
                workflow.connect([
                    (inputnode, postprocess_bold_wf, [(anat_mod, "inputnode.anat_native")]),
                ])  # fmt:skip

            if config.workflow.combine_runs and (n_task_runs > 1):
                for io_name, node in merge_dict.items():
                    workflow.connect([
                        (postprocess_bold_wf, node, [(f"outputnode.{io_name}", f"in{j_run + 1}")]),
                    ])  # fmt:skip

        if config.workflow.combine_runs and (n_task_runs > 1):
            concatenate_data_wf = init_concatenate_data_wf(
                TR=TR,
                head_radius=head_radius,
                name=f"concatenate_entity_set_{ent_set}_wf",
            )

            workflow.connect([
                (inputnode, concatenate_data_wf, [
                    ("anat_brainmask", "inputnode.anat_brainmask"),
                    ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
                    (anat_mod, "inputnode.anat_native"),
                ]),
            ])  # fmt:skip

            for io_name, node in merge_dict.items():
                workflow.connect([(node, concatenate_data_wf, [("out", f"inputnode.{io_name}")])])

    if run_counter == 0:
        raise RuntimeError(
            f"No runs survived high-motion outlier scrubbing for subject {subject_id}. "
            "Quitting workflow."
        )

    workflow.connect([
        (summary, ds_report_summary, [("out_report", "in_file")]),
        (about, ds_report_about, [("out_report", "in_file")]),
    ])  # fmt:skip

    for node in workflow.list_node_names():
        if node.split(".")[-1].startswith("ds_"):
            workflow.get_node(node).interface.out_path_base = ""

    return workflow
