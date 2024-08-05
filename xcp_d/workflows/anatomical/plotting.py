# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for generating plots from anatomical data."""

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.data import load as load_data
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import ResampleToImage
from xcp_d.interfaces.workbench import ShowScene
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.execsummary import (
    get_n_frames,
    get_png_image_names,
    make_mosaic,
    modify_brainsprite_scene_template,
    modify_pngs_scene_template,
)
from xcp_d.workflows.plotting import init_plot_overlay_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_brainsprite_figures_wf(t1w_available, t2w_available, name="brainsprite_figures_wf"):
    """Create mosaic and PNG files for executive summary brainsprite.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.plotting import init_brainsprite_figures_wf

            with mock_config():
                wf = init_brainsprite_figures_wf(
                    t1w_available=True,
                    t2w_available=True,
                    name="brainsprite_figures_wf",
                )

    Parameters
    ----------
    t1w_available : bool
        True if a T1w image is available.
    t2w_available : bool
        True if a T2w image is available.
    %(name)s
        Default is "init_brainsprite_figures_wf".

    Inputs
    ------
    t1w
        Path to T1w image. Optional. Should only be defined if ``t1w_available`` is True.
    t2w
        Path to T2w image. Optional. Should only be defined if ``t2w_available`` is True.
    lh_wm_surf
    rh_wm_surf
    lh_pial_surf
    rh_pial_surf
    """
    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir
    omp_nthreads = config.nipype.omp_nthreads

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
    brainsprite_scene_template = str(
        load_data("executive_summary_scenes/brainsprite_template.scene.gz")
    )
    pngs_scene_template = str(load_data("executive_summary_scenes/pngs_template.scene.gz"))

    if t1w_available and t2w_available:
        image_types = ["T1", "T2"]
    elif t2w_available:
        image_types = ["T2"]
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
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
            omp_nthreads=omp_nthreads,
        )
        workflow.connect([
            (inputnode, get_number_of_frames, [(inputnode_anat_name, "anat_file")]),
        ])  # fmt:skip

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
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
            omp_nthreads=omp_nthreads,
        )
        modify_brainsprite_template_scene.inputs.scene_template = brainsprite_scene_template
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
        ])  # fmt:skip

        create_framewise_pngs = pe.MapNode(
            ShowScene(
                scene_name_or_number=1,
                image_width=900,
                image_height=800,
            ),
            name=f"create_framewise_pngs_{image_type}",
            iterfield=["scene_file"],
            mem_gb=1,
            omp_nthreads=omp_nthreads,
        )
        workflow.connect([
            (modify_brainsprite_template_scene, create_framewise_pngs, [
                ("out_file", "scene_file"),
            ]),
        ])  # fmt:skip

        # Make mosaic
        make_mosaic_node = pe.Node(
            Function(
                function=make_mosaic,
                input_names=["png_files"],
                output_names=["mosaic_file"],
            ),
            name=f"make_mosaic_{image_type}",
            mem_gb=1,
            omp_nthreads=omp_nthreads,
        )

        workflow.connect([(create_framewise_pngs, make_mosaic_node, [("out_file", "png_files")])])

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
        workflow.connect([
            (inputnode, ds_mosaic_file, [(inputnode_anat_name, "source_file")]),
            (make_mosaic_node, ds_mosaic_file, [("mosaic_file", "in_file")]),
        ])  # fmt:skip

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
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
            omp_nthreads=omp_nthreads,
        )
        modify_pngs_template_scene.inputs.scene_template = pngs_scene_template
        workflow.connect([
            (inputnode, modify_pngs_template_scene, [
                (inputnode_anat_name, "anat_file"),
                ("lh_wm_surf", "lh_wm_surf"),
                ("rh_wm_surf", "rh_wm_surf"),
                ("lh_pial_surf", "lh_pial_surf"),
                ("rh_pial_surf", "rh_pial_surf"),
            ])
        ])  # fmt:skip

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
            mem_gb=1,
            omp_nthreads=omp_nthreads,
        )
        workflow.connect([
            (modify_pngs_template_scene, create_scenewise_pngs, [("out_file", "scene_file")]),
            (get_png_scene_names, create_scenewise_pngs, [
                ("scene_index", "scene_name_or_number"),
            ]),
        ])  # fmt:skip

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
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_scenewise_pngs, [(inputnode_anat_name, "source_file")]),
            (get_png_scene_names, ds_scenewise_pngs, [("scene_descriptions", "desc")]),
            (create_scenewise_pngs, ds_scenewise_pngs, [("out_file", "in_file")]),
        ])  # fmt:skip

    return workflow


@fill_doc
def init_execsummary_anatomical_plots_wf(
    t1w_available,
    t2w_available,
    name="execsummary_anatomical_plots_wf",
):
    """Generate the anatomical figures for an executive summary.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.plotting import init_execsummary_anatomical_plots_wf

            with mock_config():
                wf = init_execsummary_anatomical_plots_wf(
                    t1w_available=True,
                    t2w_available=True,
                )

    Parameters
    ----------
    t1w_available : bool
        Generally True.
    t2w_available : bool
        Generally False.
    %(name)s

    Inputs
    ------
    t1w
        T1w image, after warping to standard space.
    t2w
        T2w image, after warping to standard space.
    template
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t2w",
                "template",
            ],
        ),
        name="inputnode",
    )

    # Start plotting the overlay figures
    # Atlas in T1w/T2w, T1w/T2w in Atlas
    anatomicals = (["t1w"] if t1w_available else []) + (["t2w"] if t2w_available else [])
    for anat in anatomicals:
        # Resample anatomical to match resolution of template data
        resample_anat = pe.Node(
            ResampleToImage(),
            name=f"resample_{anat}",
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, resample_anat, [
                (anat, "in_file"),
                ("template", "target_file"),
            ]),
        ])  # fmt:skip

        plot_anat_on_atlas_wf = init_plot_overlay_wf(
            desc="AnatOnAtlas",
            name=f"plot_{anat}_on_atlas_wf",
        )
        workflow.connect([
            (inputnode, plot_anat_on_atlas_wf, [
                ("template", "inputnode.underlay_file"),
                (anat, "inputnode.name_source"),
            ]),
            (resample_anat, plot_anat_on_atlas_wf, [("out_file", "inputnode.overlay_file")]),
        ])  # fmt:skip

        plot_atlas_on_anat_wf = init_plot_overlay_wf(
            desc="AtlasOnAnat",
            name=f"plot_atlas_on_{anat}_wf",
        )
        workflow.connect([
            (inputnode, plot_atlas_on_anat_wf, [
                ("template", "inputnode.overlay_file"),
                (anat, "inputnode.name_source"),
            ]),
            (resample_anat, plot_atlas_on_anat_wf, [("out_file", "inputnode.underlay_file")]),
        ])  # fmt:skip

    # TODO: Add subcortical overlay images as well.
    # 1. Binarize atlas.

    return workflow
