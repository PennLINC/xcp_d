# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for generating the executive summary."""
import fnmatch
import os

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.surfplotting import BrainPlotx, PlotImage, RibbontoStatmap
from xcp_d.utils.bids import get_freesurfer_dir
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plot import plot_ribbon_svg

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_brainsprite_wf(
    layout,
    fmri_dir,
    subject_id,
    output_dir,
    dcan_qc,
    input_type,
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
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w", "t1seg"]),
        name="inputnode",
    )
    ribbon2statmap = pe.Node(
        RibbontoStatmap(),
        name="ribbon2statmap",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    if dcan_qc:
        # Create a brainsprite if dcan_qc is True
        plot_ribbon = pe.Node(
            BrainPlotx(),
            name="brainsprite",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
    else:
        # Otherwise, make a static mosaic plot
        plot_ribbon = pe.Node(
            Function(
                input_names=["template", "in_file"],
                output_names=["plot_file"],
                function=plot_ribbon_svg,
            ),
            name="ribbon_mosaic",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

    ds_brainspriteplot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
            dismiss_entities=["desc"],
            desc="brainplot",
            datatype="figures",
        ),
        name="ds_brainspriteplot",
    )

    use_t1seg_as_ribbon = False
    if input_type in ("dcan", "hcp"):
        # The dcan2fmriprep/hcp2fmriprep functions copy the ribbon file to the derivatives dset.
        ribbon = layout.get(
            return_type="file",
            subject=subject_id,
            desc="ribbon",
            extension="nii.gz",
        )
        if len(ribbon) != 1:
            LOGGER.warning(f"{len(ribbon)} matches found for the ribbon file: {ribbon}")
            use_t1seg_as_ribbon = True
        else:
            ribbon = ribbon[0]

    else:
        fmri_dir = os.path.abspath(fmri_dir)

        # NOTE: I try to avoid try/except statements, but this will be replaced very soon.
        try:
            freesurfer_dir = get_freesurfer_dir(fmri_dir)

            ribbon = os.path.join(freesurfer_dir, f"sub-{subject_id}", "mri", "ribbon.mgz")
            LOGGER.info(f"Using {ribbon} for ribbon.")
            if not os.path.isfile(ribbon):
                LOGGER.warning(f"File DNE: {ribbon}")
                use_t1seg_as_ribbon = True
        except NotADirectoryError:
            use_t1seg_as_ribbon = True

    if use_t1seg_as_ribbon:
        LOGGER.info("Using T1w segmentation for ribbon.")
        # fmt:off
        workflow.connect([(inputnode, ribbon2statmap, [("t1seg", "ribbon")])])
        # fmt:on
    else:
        ribbon2statmap.inputs.ribbon = ribbon

    # fmt:off
    workflow.connect(
        [
            (inputnode, plot_ribbon, [("t1w", "template")]),
            (ribbon2statmap, plot_ribbon, [("out_file", "in_file")]),
            (plot_ribbon, ds_brainspriteplot, [("plot_file", "in_file")]),
            (inputnode, ds_brainspriteplot, [("t1w", "source_file")]),
        ]
    )
    # fmt:on

    return workflow


@fill_doc
def init_execsummary_wf(
    bold_file,
    output_dir,
    layout,
    mem_gb,
    omp_nthreads,
    name="execsummary_wf",
):
    """Generate the figures for an executive summary.

    Parameters
    ----------
    bold_file
        BOLD data before post-processing.
    %(output_dir)s
    layout
    %(mem_gb)s
    %(omp_nthreads)s
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
                "boldref_file",  # a nifti boldref
            ]
        ),
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
