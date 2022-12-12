# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for generating the executive summary."""
import fnmatch
import os

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.connectivity import ApplyTransformsx
from xcp_d.interfaces.surfplotting import GenerateBrainspriteHTML, PlotImage, PlotSVGData
from xcp_d.interfaces.workbench import ShowScene
from xcp_d.utils.bids import find_nifti_bold_files
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.execsummary import (
    get_n_frames,
    get_png_image_names,
    make_mosaic,
    modify_brainsprite_scene_template,
    modify_pngs_scene_template,
)
from xcp_d.utils.utils import _t12native, get_std2bold_xforms

LOGGER = logging.getLogger("nipype.workflow")


def init_brainsprite_mini_wf(
    output_dir,
    image_type,
    brainsprite_scene_template,
    pngs_scene_template,
    name="init_brainsprite_mini_wf",
):
    """Create an executive summary-style brainsprite file."""
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "anat_file",
                "image_type",  # T1w or T2w
                "lh_wm_surf",
                "rh_wm_surf",
                "lh_pial_surf",
                "rh_pial_surf",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.image_type = image_type

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "scenewise_pngs",
                "mosaic",
            ]
        ),
        name="outputnode",
    )

    # Modify template scene file with file paths
    modify_brainsprite_template_scene = pe.Node(
        Function(
            function=modify_brainsprite_scene_template,
            input_names=[
                "anat_file",
                "rh_pial_file",
                "lh_pial_file",
                "rh_white_file",
                "lh_white_file",
                "scene_template",
            ],
            output_names=["out_file"],
        ),
        name="modify_brainsprite_template_scene",
    )
    modify_brainsprite_template_scene.inputs.scene_template = brainsprite_scene_template

    # fmt:off
    workflow.connect([
        (inputnode, modify_brainsprite_template_scene, [
            ("anat_file", "anat_file"),
            ("lh_wm_surf", "lh_white_file"),
            ("rh_wm_surf", "rh_white_file"),
            ("lh_pial_surf", "lh_pial_file"),
            ("rh_pial_surf", "rh_pial_file"),
        ]),
    ])
    # fmt:on

    # Create slice-wise PNGs
    get_number_of_frames = pe.Node(
        Function(
            function=get_n_frames,
            input_names=["scene_file"],
            output_names=["frame_numbers"],
        ),
        name="get_number_of_frames",
    )

    # fmt:off
    workflow.connect([
        (modify_brainsprite_template_scene, get_number_of_frames, [
            ("out_file", "scene_file"),
        ]),
    ])
    # fmt:on

    create_framewise_pngs = pe.MapNode(
        ShowScene(
            image_width=900,
            image_height=800,
        ),
        name="create_framewise_pngs",
        iterfield=["scene_name_or_number"],
    )

    # fmt:off
    workflow.connect([
        (modify_brainsprite_template_scene, create_framewise_pngs, [
            ("out_file", "scene_file"),
        ]),
        (get_number_of_frames, create_framewise_pngs, [
            ("frame_numbers", "scene_name_or_number"),
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
        name="make_mosaic_node",
    )

    # fmt:off
    workflow.connect([
        (create_framewise_pngs, make_mosaic_node, [
            ("out_file", "png_files"),
        ]),
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
        name="ds_mosaic_file",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_mosaic_file, [
            ("anat_file", "source_file"),
        ]),
        (make_mosaic_node, ds_mosaic_file, [
            ("mosaic_file", "in_file"),
        ]),
        (ds_mosaic_file, outputnode, [
            ("out_file", "mosaic"),
        ]),
    ])
    # fmt:on

    # Start working on the selected PNG images for the button
    modify_pngs_template_scene = pe.Node(
        Function(
            function=modify_pngs_scene_template,
            input_names=[
                "anat_file",
                "image_type",
                "rh_pial_file",
                "lh_pial_file",
                "rh_white_file",
                "lh_white_file",
                "scene_template",
            ],
            output_names=["out_file"],
        ),
        name="modify_pngs_template_scene",
    )
    modify_pngs_template_scene.inputs.scene_template = pngs_scene_template

    # fmt:off
    workflow.connect([
        (inputnode, modify_pngs_template_scene, [
            ("anat_file", "anat_file"),
            ("image_type", "image_type"),
            ("lh_wm_surf", "lh_white_file"),
            ("rh_wm_surf", "rh_white_file"),
            ("lh_pial_surf", "lh_pial_file"),
            ("rh_pial_surf", "rh_pial_file"),
        ])
    ])
    # fmt:on

    # Create specific PNGs for button
    get_png_scene_names = pe.Node(
        Function(
            function=get_png_image_names,
            input_names=["image_type"],
            output_names=["scene_index", "scene_descriptions"],
        ),
        name="get_png_scene_names",
    )

    # fmt:off
    workflow.connect([
        (inputnode, get_png_scene_names, [
            ("image_type", "image_type"),
        ]),
    ])
    # fmt:on

    create_scenewise_pngs = pe.MapNode(
        ShowScene(image_width=900, image_height=800),
        name="create_scenewise_pngs",
        iterfield=["scene_name_or_number"],
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

    # What about relative paths?
    # Do I need the HTML file to contain relative paths from itself?
    # Or from the report HTML file's path?
    ds_scenewise_pngs = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["desc"],
            datatype="figures",
            suffix=f"{image_type}w",
        ),
        name="ds_scenewise_pngs",
        run_without_submitting=False,
        iterfield=["desc", "in_file"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_scenewise_pngs, [
            ("anat_file", "source_file"),
        ]),
        (get_png_scene_names, ds_scenewise_pngs, [
            ("scene_descriptions", "desc"),
        ]),
        (create_scenewise_pngs, ds_scenewise_pngs, [
            ("out_file", "in_file"),
        ]),
        (ds_scenewise_pngs, outputnode, [
            ("out_file", "scenewise_pngs"),
        ]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_brainsprite_wf(
    output_dir,
    t2w_available,
    mem_gb,
    omp_nthreads,
    name="init_brainsprite_wf",
):
    """Create a brainsprite figure from stuff.

    Parameters
    ----------
    %(layout)s
    %(fmri_dir)s
    %(subject_id)s
    %(output_dir)s
    dcan_qc : bool
        Whether to run DCAN QC or not.
    %(input_type)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "init_brainsprite_wf".

    Inputs
    ------
    t1w
    t1seg
    """
    from pkg_resources import resource_filename as pkgrf

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t2w",
                "t1seg",
                "lh_wm_surf",
                "rh_wm_surf",
                "lh_pial_surf",
                "rh_pial_surf",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "T1w_mosaic",
                "T1w_scenewise_pngs",
                "T2w_mosaic",
                "T2w_scenewise_pngs",
            ],
        ),
        name="outputnode",
    )

    # Load template scene file
    brainsprite_scene_template = pkgrf("xcp_d", "data/parasagittal_Tx_169_template.scene.gz")
    pngs_scene_template = pkgrf("xcp_d", "data/image_template_temp.scene.gz")

    if t2w_available:
        image_types = ["T1", "T2"]
    else:
        image_types = ["T1"]

    for image_type in image_types:
        brainsprite_mini_wf = init_brainsprite_mini_wf(
            output_dir=output_dir,
            image_type=image_type,
            brainsprite_scene_template=brainsprite_scene_template,
            pngs_scene_template=pngs_scene_template,
            name=f"brainsprite_mini_{image_type}_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, brainsprite_mini_wf, [
                (f"{image_type.lower()}w", "inputnode.anat_file"),
                ("t1seg", "inputnode.t1seg"),
                ("lh_wm_surf", "inputnode.lh_wm_surf"),
                ("rh_wm_surf", "inputnode.rh_wm_surf"),
                ("lh_pial_surf", "inputnode.lh_pial_surf"),
                ("rh_pial_surf", "inputnode.rh_pial_surf"),
            ]),
            (brainsprite_mini_wf, outputnode, [
                ("outputnode.mosaic", f"{image_type}w_mosaic"),
                ("outputnode.scenewise_pngs", f"{image_type}w_scenewise_pngs"),
            ]),
        ])
        # fmt:on

        # Create brainsprite HTMLs, but only append BrainSprite JS to the latter of them.
        generate_brainsprite_html = pe.Node(
            GenerateBrainspriteHTML(
                add_javascript=image_type == image_types[-1],
            ),
            name=f"generate_brainsprite_html_{image_type}",
        )

        # fmt:off
        workflow.connect([
            (brainsprite_mini_wf, generate_brainsprite_html, [
                ("outputnode.mosaic", "mosaic"),
                ("outputnode.scenewise_pngs", "scenewise_pngs"),
            ]),
        ])
        # fmt:on

        ds_brainsprite_html = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                datatype="figures",
                desc="brainsprite",
                suffix=f"{image_type}w",
            ),
            name=f"ds_brainsprite_html_{image_type}",
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_brainsprite_html, [
                (f"{image_type.lower()}w", "source_file"),
            ]),
            (generate_brainsprite_html, ds_brainsprite_html, [
                ("outputnode.html_file", "in_file"),
            ]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_execsummary_wf(
    omp_nthreads,
    bold_file,
    output_dir,
    TR,
    mem_gb,
    layout,
    name="execsummary_wf",
):
    """Generate an executive summary.

    Parameters
    ----------
    %(omp_nthreads)s
    bold_file
    %(output_dir)s
    TR
    %(mem_gb)s
    layout
    %(name)s

    Inputs
    ------
    t1w
    t1seg
    regressed_data
    residual_data
    filtered_motion
    tmask
    rawdata
    mask
    %(template_to_t1w)s
    %(dummy_scans)s
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t1seg",
                "regressed_data",
                "residual_data",
                "filtered_motion",
                "tmask",
                "rawdata",
                "mask",
                "template_to_t1w",
                "dummy_scans",
            ]
        ),
        name="inputnode",
    )
    inputnode.inputs.bold_file = bold_file
    # Get bb_registration_file prefix from fmriprep
    all_files = list(layout.get_files())
    current_bold_file = os.path.basename(bold_file)
    if "_space" in current_bold_file:
        bb_register_prefix = current_bold_file.split("_space")[0]
    else:
        bb_register_prefix = current_bold_file.split("_desc")[0]

    # check if there is a bb_registration_file or coregister file
    patterns = ("*bbregister_bold.svg", "*coreg_bold.svg", "*bbr_bold.svg")
    registration_file = [pat for pat in patterns if fnmatch.filter(all_files, pat)]
    #  Get the T1w registration file
    bold_t1w_registration_file = fnmatch.filter(
        all_files, "*" + bb_register_prefix + registration_file[0]
    )[0]

    find_nifti_files = pe.Node(
        Function(
            function=find_nifti_bold_files,
            input_names=["bold_file", "template_to_t1w"],
            output_names=["nifti_bold_file", "nifti_boldref_file"],
        ),
        name="find_nifti_files",
    )

    # fmt:off
    workflow.connect([
        (inputnode, find_nifti_files, [
            ("bold_file", "bold_file"),
            ("template_to_t1w", "template_to_t1w"),
        ]),
    ])
    # fmt:on

    # Plot the reference bold image
    plotrefbold_wf = pe.Node(PlotImage(), name="plotrefbold_wf")

    # fmt:off
    workflow.connect([
        (find_nifti_files, plotrefbold_wf, [
            ("nifti_boldref_file", "in_file"),
        ]),
    ])
    # fmt:on

    find_t1_to_native = pe.Node(
        Function(
            function=_t12native,
            input_names=["fname"],
            output_names=["t1w_to_native_xform"],
        ),
        name="find_t1_to_native",
    )

    # fmt:off
    workflow.connect([
        (find_nifti_files, find_t1_to_native, [
            ("nifti_bold_file", "fname"),
        ]),
    ])
    # fmt:on

    # Get the transform file to native space
    get_std2native_transform = pe.Node(
        Function(
            input_names=["bold_file", "template_to_t1w", "t1w_to_native"],
            output_names=["transform_list"],
            function=get_std2bold_xforms,
        ),
        name="get_std2native_transform",
    )

    # fmt:off
    workflow.connect([
        (find_nifti_files, get_std2native_transform, [
            ("nifti_bold_file", "bold_file"),
        ]),
        (find_t1_to_native, get_std2native_transform, [
            ("t1w_to_native_xform", "t1w_to_native"),
        ]),
    ])
    # fmt:on

    # Transform the file to native space
    resample_parc = pe.Node(
        ApplyTransformsx(
            dimension=3,
            input_image=str(
                get_template(
                    "MNI152NLin2009cAsym",
                    resolution=1,
                    desc="carpet",
                    suffix="dseg",
                    extension=[".nii", ".nii.gz"],
                )
            ),
            interpolation="MultiLabel",
        ),
        name="resample_parc",
        n_procs=omp_nthreads,
        mem_gb=mem_gb * 3 * omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (find_nifti_files, resample_parc, [
            ("nifti_boldref_file", "reference_image"),
        ]),
    ])
    # fmt:on

    # Plot the SVG files
    plot_svgx_wf = pe.Node(
        PlotSVGData(TR=TR),
        name="plot_svgx_wf",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # Write out the necessary files:
    # Reference file
    ds_plot_bold_reference_file_wf = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir, dismiss_entities=["den"], datatype="figures", desc="boldref"
        ),
        name="ds_plot_bold_reference_file_wf",
        run_without_submitting=True,
    )

    # Plot SVG before
    ds_plot_svg_before_wf = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc="precarpetplot",
        ),
        name="plot_svgxbe",
        run_without_submitting=True,
    )
    # Plot SVG after
    ds_plot_svg_after_wf = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc="postcarpetplot",
        ),
        name="plot_svgx_after",
        run_without_submitting=True,
    )
    # Bold T1 registration file
    ds_registration_wf = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            in_file=bold_t1w_registration_file,
            dismiss_entities=["den"],
            datatype="figures",
            desc="bbregister",
        ),
        name="bb_registration_file",
        run_without_submitting=True,
    )

    # Connect all the workflows
    # fmt:off
    workflow.connect([
        (plotrefbold_wf, ds_plot_bold_reference_file_wf, [('out_file', 'in_file')]),
        (inputnode, plot_svgx_wf, [
            ('filtered_motion', 'filtered_motion'),
            ('regressed_data', 'regressed_data'),
            ('residual_data', 'residual_data'),
            ('mask', 'mask'),
            ('bold_file', 'rawdata'),
            ('tmask', 'tmask'),
            ('dummy_scans', 'dummy_scans'),
        ]),
        (inputnode, get_std2native_transform, [('template_to_t1w', 'template_to_t1w')]),
        (get_std2native_transform, resample_parc, [('transform_list', 'transforms')]),
        (resample_parc, plot_svgx_wf, [('output_image', 'seg_data')]),
        (plot_svgx_wf, ds_plot_svg_before_wf, [('before_process', 'in_file')]),
        (plot_svgx_wf, ds_plot_svg_after_wf, [('after_process', 'in_file')]),
        (inputnode, ds_plot_svg_before_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_plot_svg_after_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_plot_bold_reference_file_wf, [('bold_file', 'source_file')]),
        (inputnode, ds_registration_wf, [('bold_file', 'source_file')]),
    ])
    # fmt:on

    return workflow
