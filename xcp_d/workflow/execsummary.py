# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for generating the executive summary."""
import fnmatch
import os

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.surfplotting import PlotImage
from xcp_d.interfaces.workbench import ShowScene
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.execsummary import (
    get_n_frames,
    get_png_image_names,
    make_mosaic,
    modify_brainsprite_scene_template,
    modify_pngs_scene_template,
)

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_brainsprite_figures_wf(
    output_dir,
    t2w_available,
    mem_gb,
    omp_nthreads,
    name="init_brainsprite_figures_wf",
):
    """Create mosaic and PNG files for executive summary brainsprite.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.execsummary import init_brainsprite_figures_wf

            wf = init_brainsprite_figures_wf(
                output_dir=".",
                t2w_available=True,
                mem_gb=0.1,
                omp_nthreads=1,
                name="brainsprite_figures_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    t2w_available : bool
        True if a T2w image is available.
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "init_brainsprite_figures_wf".

    Inputs
    ------
    t1w
        Path to T1w image.
    t2w
        Path to T2w image. Optional. Should only be defined if ``t2w_available`` is True.
    lh_wm_surf
    rh_wm_surf
    lh_pial_surf
    rh_pial_surf
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t2w",
                "lh_wm_surf",
                "rh_wm_surf",
                "lh_pial_surf",
                "rh_pial_surf",
            ],
        ),
        name="inputnode",
    )

    # Load template scene file
    brainsprite_scene_template = pkgrf(
        "xcp_d",
        "data/executive_summary_scenes/brainsprite_template.scene.gz",
    )
    pngs_scene_template = pkgrf("xcp_d", "data/executive_summary_scenes/pngs_template.scene.gz")

    if t2w_available:
        image_types = ["T1", "T2"]
    else:
        image_types = ["T1"]

    for image_type in image_types:
        inputnode_anat_name = f"{image_type.lower()}w"
        # Create frame-wise PNGs
        get_number_of_frames = pe.Node(
            Function(
                function=get_n_frames,
                input_names=["anat_file"],
                output_names=["frame_numbers"],
            ),
            name=f"get_number_of_frames_{image_type}",
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, get_number_of_frames, [(inputnode_anat_name, "anat_file")]),
        ])
        # fmt:on

        # Modify template scene file with file paths
        modify_brainsprite_template_scene = pe.MapNode(
            Function(
                function=modify_brainsprite_scene_template,
                input_names=[
                    "slice_number",
                    "anat_file",
                    "rh_pial_surf",
                    "lh_pial_surf",
                    "rh_wm_surf",
                    "lh_wm_surf",
                    "scene_template",
                ],
                output_names=["out_file"],
            ),
            name=f"modify_brainsprite_template_scene_{image_type}",
            iterfield=["slice_number"],
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
        )
        modify_brainsprite_template_scene.inputs.scene_template = brainsprite_scene_template

        # fmt:off
        workflow.connect([
            (inputnode, modify_brainsprite_template_scene, [
                (inputnode_anat_name, "anat_file"),
                ("lh_wm_surf", "lh_wm_surf"),
                ("rh_wm_surf", "rh_wm_surf"),
                ("lh_pial_surf", "lh_pial_surf"),
                ("rh_pial_surf", "rh_pial_surf"),
            ]),
            (get_number_of_frames, modify_brainsprite_template_scene, [
                ("frame_numbers", "slice_number"),
            ]),
        ])
        # fmt:on

        create_framewise_pngs = pe.MapNode(
            ShowScene(
                scene_name_or_number=1,
                image_width=900,
                image_height=800,
            ),
            name=f"create_framewise_pngs_{image_type}",
            iterfield=["scene_file"],
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (modify_brainsprite_template_scene, create_framewise_pngs, [
                ("out_file", "scene_file"),
            ]),
        ])
        # fmt:on

        # Make mosaic
        make_mosaic_node = pe.Node(
            Function(
                function=make_mosaic,
                input_names=["png_files"],
                output_names=["mosaic_file"],
            ),
            name=f"make_mosaic_{image_type}",
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (create_framewise_pngs, make_mosaic_node, [("out_file", "png_files")]),
        ])
        # fmt:on

        ds_mosaic_file = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["desc"],
                desc="mosaic",
                datatype="figures",
                suffix=f"{image_type}w",
            ),
            name=f"ds_mosaic_file_{image_type}",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_mosaic_file, [(inputnode_anat_name, "source_file")]),
            (make_mosaic_node, ds_mosaic_file, [("mosaic_file", "in_file")]),
        ])
        # fmt:on

        # Start working on the selected PNG images for the button
        modify_pngs_template_scene = pe.Node(
            Function(
                function=modify_pngs_scene_template,
                input_names=[
                    "anat_file",
                    "rh_pial_surf",
                    "lh_pial_surf",
                    "rh_wm_surf",
                    "lh_wm_surf",
                    "scene_template",
                ],
                output_names=["out_file"],
            ),
            name=f"modify_pngs_template_scene_{image_type}",
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
        )
        modify_pngs_template_scene.inputs.scene_template = pngs_scene_template

        # fmt:off
        workflow.connect([
            (inputnode, modify_pngs_template_scene, [
                (inputnode_anat_name, "anat_file"),
                ("lh_wm_surf", "lh_wm_surf"),
                ("rh_wm_surf", "rh_wm_surf"),
                ("lh_pial_surf", "lh_pial_surf"),
                ("rh_pial_surf", "rh_pial_surf"),
            ])
        ])
        # fmt:on

        # Create specific PNGs for button
        get_png_scene_names = pe.Node(
            Function(
                function=get_png_image_names,
                output_names=["scene_index", "scene_descriptions"],
            ),
            name=f"get_png_scene_names_{image_type}",
        )

        create_scenewise_pngs = pe.MapNode(
            ShowScene(image_width=900, image_height=800),
            name=f"create_scenewise_pngs_{image_type}",
            iterfield=["scene_name_or_number"],
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (modify_pngs_template_scene, create_scenewise_pngs, [
                ("out_file", "scene_file"),
            ]),
            (get_png_scene_names, create_scenewise_pngs, [
                ("scene_index", "scene_name_or_number"),
            ]),
        ])
        # fmt:on

        ds_scenewise_pngs = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["desc"],
                datatype="figures",
                suffix=f"{image_type}w",
            ),
            name=f"ds_scenewise_pngs_{image_type}",
            run_without_submitting=False,
            iterfield=["desc", "in_file"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_scenewise_pngs, [(inputnode_anat_name, "source_file")]),
            (get_png_scene_names, ds_scenewise_pngs, [("scene_descriptions", "desc")]),
            (create_scenewise_pngs, ds_scenewise_pngs, [("out_file", "in_file")]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_execsummary_wf(
    bold_file,
    output_dir,
    layout,
    name="execsummary_wf",
):
    """Generate the figures for an executive summary.

    Parameters
    ----------
    bold_file
        BOLD data before post-processing.
    %(output_dir)s
    layout
    %(name)s

    Inputs
    ------
    bold_file
        BOLD data before post-processing.
        Set from the parameter.
    boldref_file
        The boldref file associated with the BOLD file.
        This should only be defined (and used) for NIFTI inputs.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "boldref_file",
            ]
        ),  # a nifti boldref
        name="inputnode",
    )
    inputnode.inputs.bold_file = bold_file

    # Get bb_registration_file prefix from fmriprep
    # TODO: Replace with interfaces.
    all_files = list(layout.get_files())
    current_bold_file = os.path.basename(bold_file)
    if "_space" in current_bold_file:
        bb_register_prefix = current_bold_file.split("_space")[0]
    else:
        bb_register_prefix = current_bold_file.split("_desc")[0]

    # check if there is a bb_registration_file or coregister file
    patterns = ("*bbregister_bold.svg", "*coreg_bold.svg", "*bbr_bold.svg")
    registration_file = [pat for pat in patterns if fnmatch.filter(all_files, pat)]
    # Get the T1w registration file
    bold_t1w_registration_file = fnmatch.filter(
        all_files, "*" + bb_register_prefix + registration_file[0]
    )[0]

    # Plot the reference bold image
    plot_boldref = pe.Node(PlotImage(), name="plot_boldref")

    # fmt:off
    workflow.connect([
        (inputnode, plot_boldref, [("boldref_file", "in_file")]),
    ])
    # fmt:on

    # Write out the figures.
    ds_boldref_figure = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc="boldref",
        ),
        name="ds_boldref_figure",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_boldref_figure, [("bold_file", "source_file")]),
        (plot_boldref, ds_boldref_figure, [("out_file", "in_file")]),
    ])
    # fmt:on

    ds_registration_figure = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            in_file=bold_t1w_registration_file,
            dismiss_entities=["den"],
            datatype="figures",
            desc="bbregister",
        ),
        name="ds_registration_figure",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_registration_figure, [("bold_file", "source_file")]),
    ])
    # fmt:on

    return workflow
