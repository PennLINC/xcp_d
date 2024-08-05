"""Workflows for generating plots from imaging data."""

from nipype.interfaces import fsl
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.execsummary import FormatForBrainSwipes
from xcp_d.interfaces.nilearn import BinaryMath
from xcp_d.interfaces.plotting import PNGAppend
from xcp_d.utils.doc import fill_doc


def init_plot_overlay_wf(desc, name="plot_overlay_wf"):
    """Use the default slices from slicesdir to make a plot."""
    from xcp_d.interfaces.plotting import SlicesDir

    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir

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
        mem_gb=1,
    )

    workflow.connect([
        (inputnode, plot_overlay_figure, [
            ("underlay_file", "in_files"),
            ("overlay_file", "outline_image"),
        ]),
    ])  # fmt:skip

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
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (inputnode, ds_overlay_figure, [("name_source", "source_file")]),
        (plot_overlay_figure, ds_overlay_figure, [("out_files", "in_file")]),
    ])  # fmt:skip

    reformat_for_brain_swipes = pe.Node(FormatForBrainSwipes(), name="reformat_for_brain_swipes")
    workflow.connect([
        (plot_overlay_figure, reformat_for_brain_swipes, [("slicewise_files", "in_files")]),
    ])  # fmt:skip

    ds_reformatted_figure = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc=f"{desc}BrainSwipes",
            extension=".png",
        ),
        name="ds_reformatted_figure",
        run_without_submitting=True,
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([
        (inputnode, ds_reformatted_figure, [("name_source", "source_file")]),
        (reformat_for_brain_swipes, ds_reformatted_figure, [("out_file", "in_file")]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_plot_custom_slices_wf(
    output_dir,
    desc,
    name="plot_custom_slices_wf",
):
    """Plot a custom selection of slices with Slicer.

    This workflow is used to produce subcortical registration plots specifically for
    infant data.

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

    Inputs
    ------
    underlay_file
    overlay_file
    name_source
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
        mem_gb=1,
    )

    workflow.connect([(inputnode, binarize_edges, [("overlay_file", "in_file")])])

    make_image = pe.MapNode(
        fsl.Slicer(show_orientation=True, label_slices=True),
        name="make_image",
        iterfield=["single_slice", "slice_number"],
        mem_gb=1,
    )
    make_image.inputs.single_slice = SINGLE_SLICES
    make_image.inputs.slice_number = SLICE_NUMBERS
    workflow.connect([
        (inputnode, make_image, [("underlay_file", "in_file")]),
        (binarize_edges, make_image, [("out_file", "image_edges")]),
    ])  # fmt:skip

    combine_images = pe.Node(
        PNGAppend(out_file="out.png"),
        name="combine_images",
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )

    workflow.connect([(make_image, combine_images, [("out_file", "in_files")])])

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
        mem_gb=config.DEFAULT_MEMORY_MIN_GB,
    )
    workflow.connect([
        (inputnode, ds_overlay_figure, [("name_source", "source_file")]),
        (combine_images, ds_overlay_figure, [("out_file", "in_file")]),
    ])  # fmt:skip

    return workflow
