# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for generating the executive summary."""
import fnmatch
import os

from nipype import Function, logging
from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import BinaryMath, ResampleToImage
from xcp_d.interfaces.plotting import AnatomicalPlot, PNGAppend
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
    t1w_available,
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

            from xcp_d.workflows.execsummary import init_brainsprite_figures_wf

            wf = init_brainsprite_figures_wf(
                output_dir=".",
                t1w_available=True,
                t2w_available=True,
                mem_gb=0.1,
                omp_nthreads=1,
                name="brainsprite_figures_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    t1w_available : bool
        True if a T1w image is available.
    t2w_available : bool
        True if a T2w image is available.
    %(mem_gb)s
    %(omp_nthreads)s
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
def init_execsummary_functional_plots_wf(
    preproc_nifti,
    t1w_available,
    t2w_available,
    output_dir,
    layout,
    name="execsummary_functional_plots_wf",
):
    """Generate the functional figures for an executive summary.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.execsummary import init_execsummary_functional_plots_wf

            wf = init_execsummary_functional_plots_wf(
                preproc_nifti=None,
                t1w_available=True,
                t2w_available=True,
                output_dir=".",
                layout=None,
                name="execsummary_functional_plots_wf",
            )

    Parameters
    ----------
    preproc_nifti : :obj:`str` or None
        BOLD data before post-processing.
        A NIFTI file, not a CIFTI.
    t1w_available : :obj:`bool`
        Generally True.
    t2w_available : :obj:`bool`
        Generally False.
    %(output_dir)s
    %(layout)s
    %(name)s

    Inputs
    ------
    preproc_nifti
        BOLD data before post-processing.
        A NIFTI file, not a CIFTI.
        Set from the parameter.
    %(boldref)s
    t1w
    t2w
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "preproc_nifti",
                "boldref",
                "t1w",
                "t2w",  # optional
            ]
        ),  # a nifti boldref
        name="inputnode",
    )
    if preproc_nifti:
        inputnode.inputs.preproc_nifti = preproc_nifti

        # Only grab the bb_registration_file if the preprocessed BOLD file is a parameter.
        # Get bb_registration_file prefix from fmriprep
        # TODO: Replace with interfaces.
        all_files = list(layout.get_files())
        current_bold_file = os.path.basename(preproc_nifti)
        if "_space" in current_bold_file:
            bb_register_prefix = current_bold_file.split("_space")[0]
        else:
            bb_register_prefix = current_bold_file.split("_desc")[0]

        # check if there is a bb_registration_file or coregister file
        patterns = ("*bbregister_bold.svg", "*coreg_bold.svg", "*bbr_bold.svg")
        registration_file = [pat for pat in patterns if fnmatch.filter(all_files, pat)]
        # Get the T1w registration file
        bold_t1w_registration_file = fnmatch.filter(
            all_files, f"*{bb_register_prefix}{registration_file[0]}"
        )[0]

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
        workflow.connect([(inputnode, ds_registration_figure, [("preproc_nifti", "source_file")])])
        # fmt:on
    else:
        LOGGER.warning(
            "Preprocessed NIFTI file not provided as a parameter, "
            "so the BBReg figure will not be extracted."
        )

    # Plot the mean bold image
    calculate_mean_bold = pe.Node(
        BinaryMath(expression="np.mean(img, axis=3)"),
        name="calculate_mean_bold",
    )
    plot_meanbold = pe.Node(AnatomicalPlot(), name="plot_meanbold")

    # fmt:off
    workflow.connect([
        (inputnode, calculate_mean_bold, [("preproc_nifti", "in_file")]),
        (calculate_mean_bold, plot_meanbold, [("out_file", "in_file")]),
    ])
    # fmt:on

    # Write out the figures.
    ds_meanbold_figure = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc="mean",
        ),
        name="ds_meanbold_figure",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_meanbold_figure, [("preproc_nifti", "source_file")]),
        (plot_meanbold, ds_meanbold_figure, [("out_file", "in_file")]),
    ])
    # fmt:on

    # Plot the reference bold image
    plot_boldref = pe.Node(AnatomicalPlot(), name="plot_boldref")

    # fmt:off
    workflow.connect([
        (inputnode, plot_boldref, [("boldref", "in_file")]),
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
        (inputnode, ds_boldref_figure, [("preproc_nifti", "source_file")]),
        (plot_boldref, ds_boldref_figure, [("out_file", "in_file")]),
    ])
    # fmt:on

    # Start plotting the overlay figures
    # T1 in Task, Task in T1, Task in T2, T2 in Task
    if t1w_available:
        # Resample T1w to match resolution of task data
        resample_t1w = pe.Node(
            ResampleToImage(),
            name="resample_t1w",
        )

        # fmt:off
        workflow.connect([
            (inputnode, resample_t1w, [
                ("t1w", "in_file"),
            ]),
            (calculate_mean_bold, resample_t1w, [
                ("out_file", "target_file"),
            ]),
        ])
        # fmt:on

        plot_t1w_on_task_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="T1wOnTask",
            name="plot_t1w_on_task_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_t1w_on_task_wf, [
                ("preproc_nifti", "inputnode.name_source"),
            ]),
            (calculate_mean_bold, plot_t1w_on_task_wf, [
                ("out_file", "inputnode.underlay_file"),
            ]),
            (resample_t1w, plot_t1w_on_task_wf, [
                ("out_file", "inputnode.overlay_file"),
            ]),
        ])
        # fmt:on

        plot_task_on_t1w_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="TaskOnT1w",
            name="plot_task_on_t1w_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_task_on_t1w_wf, [
                ("preproc_nifti", "inputnode.name_source"),
            ]),
            (calculate_mean_bold, plot_task_on_t1w_wf, [
                ("out_file", "inputnode.overlay_file"),
            ]),
            (resample_t1w, plot_task_on_t1w_wf, [
                ("out_file", "inputnode.underlay_file"),
            ]),
        ])
        # fmt:on

    if t2w_available:
        # Resample T2w to match resolution of task data
        resample_t2w = pe.Node(
            ResampleToImage(),
            name="resample_t2w",
        )

        # fmt:off
        workflow.connect([
            (inputnode, resample_t2w, [
                ("t2w", "in_file"),
            ]),
            (calculate_mean_bold, resample_t2w, [
                ("out_file", "target_file"),
            ]),
        ])
        # fmt:on

        plot_t2w_on_task_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="T2wOnTask",
            name="plot_t2w_on_task_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_t2w_on_task_wf, [
                ("preproc_nifti", "inputnode.underlay_file"),
                ("preproc_nifti", "inputnode.name_source"),
            ]),
            (resample_t2w, plot_t2w_on_task_wf, [
                ("out_file", "inputnode.overlay_file"),
            ]),
        ])
        # fmt:on

        plot_task_on_t2w_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="TaskOnT2w",
            name="plot_task_on_t2w_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_task_on_t2w_wf, [
                ("preproc_nifti", "inputnode.overlay_file"),
                ("preproc_nifti", "inputnode.name_source"),
            ]),
            (resample_t2w, plot_task_on_t2w_wf, [
                ("out_file", "inputnode.underlay_file"),
            ]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_execsummary_anatomical_plots_wf(
    t1w_available,
    t2w_available,
    output_dir,
    name="execsummary_anatomical_plots_wf",
):
    """Generate the anatomical figures for an executive summary.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.execsummary import init_execsummary_anatomical_plots_wf

            wf = init_execsummary_anatomical_plots_wf(
                t1w_available=True,
                t2w_available=True,
                output_dir=".",
                name="execsummary_anatomical_plots_wf",
            )

    Parameters
    ----------
    t1w_available : bool
        Generally True.
    t2w_available : bool
        Generally False.
    %(output_dir)s
    %(name)s

    Inputs
    ------
    t1w
    t2w
    template
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t2w",
                "template",
            ]
        ),
        name="inputnode",
    )

    # Start plotting the overlay figures
    # Atlas in T1w, T1w in Atlas
    if t1w_available:
        # Resample T1w to match resolution of template data
        resample_t1w = pe.Node(
            ResampleToImage(),
            name="resample_t1w",
        )

        # fmt:off
        workflow.connect([
            (inputnode, resample_t1w, [
                ("t1w", "in_file"),
                ("template", "target_file"),
            ])
        ])
        # fmt:on

        plot_t1w_on_atlas_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="AnatOnAtlas",
            name="plot_t1w_on_atlas_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_t1w_on_atlas_wf, [
                ("template", "inputnode.underlay_file"),
                ("t1w", "inputnode.name_source"),
            ]),
            (resample_t1w, plot_t1w_on_atlas_wf, [
                ("out_file", "inputnode.overlay_file"),
            ]),
        ])
        # fmt:on

        plot_atlas_on_t1w_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="AtlasOnAnat",
            name="plot_atlas_on_t1w_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_atlas_on_t1w_wf, [
                ("template", "inputnode.overlay_file"),
                ("t1w", "inputnode.name_source"),
            ]),
            (resample_t1w, plot_atlas_on_t1w_wf, [
                ("out_file", "inputnode.underlay_file"),
            ]),
        ])
        # fmt:on

    # Atlas in T2w, T2w in Atlas
    if t2w_available:
        # Resample T2w to match resolution of template data
        resample_t2w = pe.Node(
            ResampleToImage(),
            name="resample_t2w",
        )

        # fmt:off
        workflow.connect([
            (inputnode, resample_t2w, [
                ("t2w", "in_file"),
                ("template", "target_file"),
            ])
        ])
        # fmt:on

        plot_t2w_on_atlas_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="AnatOnAtlas",
            name="plot_t2w_on_atlas_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_t2w_on_atlas_wf, [
                ("template", "inputnode.underlay_file"),
                ("t2w", "inputnode.name_source"),
            ]),
            (resample_t2w, plot_t2w_on_atlas_wf, [
                ("out_file", "inputnode.overlay_file"),
            ]),
        ])
        # fmt:on

        plot_atlas_on_t2w_wf = init_plot_overlay_wf(
            output_dir=output_dir,
            desc="AtlasOnAnat",
            name="plot_atlas_on_t2w_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, plot_atlas_on_t2w_wf, [
                ("template", "inputnode.overlay_file"),
                ("t2w", "inputnode.name_source"),
            ]),
            (resample_t2w, plot_atlas_on_t2w_wf, [
                ("out_file", "inputnode.underlay_file"),
            ]),
        ])
        # fmt:on

    # TODO: Add subcortical overlay images as well.
    # 1. Binarize atlas.

    return workflow


@fill_doc
def init_plot_custom_slices_wf(
    output_dir,
    desc,
    name="plot_custom_slices_wf",
):
    """Plot a custom selection of slices with Slicer.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.execsummary import init_plot_custom_slices_wf

            wf = init_plot_custom_slices_wf(
                output_dir=".",
                desc="AtlasOnSubcorticals",
                name="plot_custom_slices_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    desc : :obj:`str`
        String to be used as ``desc`` entity in output filename.
    %(name)s
        Default is "plot_custom_slices_wf".
    """
    # NOTE: These slices are almost certainly specific to a given MNI template and resolution.
    SINGLE_SLICES = ["x", "x", "x", "y", "y", "y", "z", "z", "z"]
    SLICE_NUMBERS = [36, 45, 52, 43, 54, 65, 23, 33, 39]

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "underlay_file",
                "overlay_file",
                "name_source",
            ],
        ),
        name="inputnode",
    )

    # slices/slicer does not do well trying to make the red outline when it
    # cannot find the edges, so cannot use the ROI files with some low intensities.
    binarize_edges = pe.Node(
        BinaryMath(expression="img.astype(bool).astype(int)"),
        name="binarize_edges",
    )

    workflow.connect([(inputnode, binarize_edges, [("overlay_file", "in_file")])])

    make_image = pe.MapNode(
        fsl.Slicer(args="-u -L"),
        name="make_image",
        iterfield=["single_slice", "slice_number"],
    )
    make_image.inputs.single_slice = SINGLE_SLICES
    make_image.inputs.slice_number = SLICE_NUMBERS

    # fmt:off
    workflow.connect([
        (inputnode, make_image, [("underlay_file", "in_file")]),
        (binarize_edges, make_image, [("out_file", "image_edges")]),
    ])
    # fmt:on

    combine_images = pe.Node(
        PNGAppend(),
        name="combine_images",
    )

    # fmt:off
    workflow.connect([
        (make_image, combine_images, [("out_file", "in_files")]),
    ])
    # fmt:on

    ds_overlay_figure = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc=desc,
            extension=".png",
        ),
        name="ds_overlay_figure",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_overlay_figure, [("name_source", "source_file")]),
        (combine_images, ds_overlay_figure, [("out_file", "in_file")]),
    ])
    # fmt:on

    return workflow


def init_plot_overlay_wf(
    output_dir,
    desc,
    name="plot_overlay_wf",
):
    """Use the default slices from slicesdir to make a plot."""
    from xcp_d.interfaces.plotting import SlicesDir

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "underlay_file",
                "overlay_file",
                "name_source",
            ],
        ),
        name="inputnode",
    )

    plot_overlay_figure = pe.Node(
        SlicesDir(out_extension=".png"),
        name="plot_overlay_figure",
    )

    # fmt:off
    workflow.connect([
        (inputnode, plot_overlay_figure, [
            ("underlay_file", "in_files"),
            ("overlay_file", "outline_image"),
        ]),
    ])
    # fmt:on

    ds_overlay_figure = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc=desc,
            extension=".png",
        ),
        name="ds_overlay_figure",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_overlay_figure, [("name_source", "source_file")]),
        (plot_overlay_figure, ds_overlay_figure, [("out_files", "in_file")]),
    ])
    # fmt:on

    return workflow
