# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for generating the executive summary."""
import fnmatch
import os
from pathlib import Path

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.connectivity import ApplyTransformsx
from xcp_d.interfaces.surfplotting import (
    BrainPlotx,
    PlotImage,
    PlotSVGData,
    RibbontoStatmap,
)
from xcp_d.utils.bids import find_nifti_boldref_file
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plot import plot_ribbon_svg
from xcp_d.utils.utils import _t12native, get_std2bold_xforms

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
        # verify freesurfer directory
        fmri_path = Path(fmri_dir).absolute()

        # for fmriprep and nibabies versions before XXXX,
        # the freesurfer dir was placed at the same level as the main derivatives
        freesurfer_paths = [fp for fp in fmri_path.parent.glob("*freesurfer*") if fp.is_dir()]
        if len(freesurfer_paths) == 0:
            # for later versions, the freesurfer dir is placed in sourcedata
            # within the main derivatives folder
            freesurfer_paths = [
                fp for fp in fmri_path.glob("sourcedata/*freesurfer*") if fp.is_dir()
            ]

        if len(freesurfer_paths) > 0:
            freesurfer_path = freesurfer_paths[0]
            LOGGER.info(f"Freesurfer directory found at {freesurfer_path}.")
            ribbon = freesurfer_path / f"sub-{subject_id}" / "mri" / "ribbon.mgz"
            LOGGER.info(f"Using {ribbon} for ribbon.")

            if not ribbon.is_file():
                LOGGER.warning(f"File DNE: {ribbon}")
                use_t1seg_as_ribbon = True

        else:
            LOGGER.info("No Freesurfer derivatives found.")
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
    TR,
    layout,
    cifti,
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
    TR
    layout
    %(cifti)s
        The dseg file will not be loaded or transformed to BOLD space for CIFTI data.
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
    regressed_data
        BOLD data after regression, but before filtering.
    residual_data
        BOLD data after regression and filtering.
    filtered_motion
    tmask
    mask
    %(template_to_t1w)s
    %(dummy_scans)s
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "regressed_data",
                "residual_data",
                "filtered_motion",
                "tmask",
                "template_to_t1w",
                "dummy_scans",
                # nifti-only inputs
                "boldref_file",
                "mask",
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
    # Get the T1w registration file
    bold_t1w_registration_file = fnmatch.filter(
        all_files, "*" + bb_register_prefix + registration_file[0]
    )[0]

    # Plot the reference bold image
    plot_boldref = pe.Node(PlotImage(), name="plot_boldref")

    if not cifti:
        # NIFTI files require a tissue-type segmentation in the same space as the BOLD data.

        # Get the transform file to native space
        # Given that xcp-d doesn't process native-space data, this transform will never be used.
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
            (inputnode, find_t1_to_native, [("bold_file", "fname")]),
        ])
        # fmt:on

        # Get the set of transforms from MNI152NLin6Asym (the dseg) to the BOLD space.
        # Given that xcp-d doesn't process native-space data, this transform will never be used.
        get_mni_to_bold_xforms = pe.Node(
            Function(
                input_names=["bold_file", "template_to_t1w", "t1w_to_native"],
                output_names=["transform_list"],
                function=get_std2bold_xforms,
            ),
            name="get_std2native_transform",
        )

        # fmt:off
        workflow.connect([
            (inputnode, get_mni_to_bold_xforms, [
                ("template_to_t1w", "template_to_t1w"),
                ("bold_file", "bold_file"),
            ]),
            (find_t1_to_native, get_mni_to_bold_xforms, [
                ("t1w_to_native_xform", "t1w_to_native"),
            ]),
        ])
        # fmt:on

        # Transform a dseg file to the same space as the BOLD data
        warp_dseg_to_bold = pe.Node(
            ApplyTransformsx(
                dimension=3,
                input_image=str(
                    get_template(
                        "MNI152NLin6Asym",
                        resolution=1,
                        desc="carpet",
                        suffix="dseg",
                        extension=[".nii", ".nii.gz"],
                    )
                ),
                interpolation="MultiLabel",
            ),
            name="warp_dseg_to_bold",
            n_procs=omp_nthreads,
            mem_gb=mem_gb * 3 * omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, warp_dseg_to_bold, [("bold_file", "reference_image")]),
            (get_mni_to_bold_xforms, warp_dseg_to_bold, [("transform_list", "transforms")]),
            (inputnode, plot_boldref, [("boldref_file", "in_file")]),
        ])
        # fmt:on
    else:
        find_nifti_boldref = pe.Node(
            Function(
                function=find_nifti_boldref_file,
                input_names=["bold_file", "template_to_t1w"],
                output_names=["nifti_boldref_file"],
            ),
            name="find_nifti_boldref",
        )

        # fmt:off
        workflow.connect([
            (inputnode, find_nifti_boldref, [
                ("bold_file", "bold_file"),
                ("template_to_t1w", "template_to_t1w"),
            ]),
            (find_nifti_boldref, plot_boldref, [
                ("nifti_boldref_file", "in_file"),
            ]),
        ])
        # fmt:on

    # Generate preprocessing and postprocessing carpet plots.
    plot_carpets = pe.Node(
        PlotSVGData(TR=TR),
        name="plot_carpets",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, plot_carpets, [
            ("filtered_motion", "filtered_motion"),
            ("regressed_data", "regressed_data"),
            ("residual_data", "residual_data"),
            ("bold_file", "rawdata"),
            ("tmask", "tmask"),
            ("dummy_scans", "dummy_scans"),
        ]),
    ])
    # fmt:on

    if not cifti:
        # fmt:off
        workflow.connect([
            (inputnode, plot_carpets, [("mask", "mask")]),
            (warp_dseg_to_bold, plot_carpets, [("output_image", "seg_data")]),
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

    ds_preproc_carpet = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc="precarpetplot",
        ),
        name="ds_preproc_carpet",
        run_without_submitting=True,
    )

    ds_postproc_carpet = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["den"],
            datatype="figures",
            desc="postcarpetplot",
        ),
        name="ds_postproc_carpet",
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_preproc_carpet, [("bold_file", "source_file")]),
        (inputnode, ds_postproc_carpet, [("bold_file", "source_file")]),
        (plot_carpets, ds_preproc_carpet, [("before_process", "in_file")]),
        (plot_carpets, ds_postproc_carpet, [("after_process", "in_file")]),
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
