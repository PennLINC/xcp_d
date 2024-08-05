"""Workflows for processing volumetric anatomical data."""

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d import config
from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import list_to_str
from xcp_d.workflows.anatomical.plotting import init_execsummary_anatomical_plots_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_postprocess_anat_wf(
    t1w_available,
    t2w_available,
    target_space,
    name="postprocess_anat_wf",
):
    """Copy T1w, segmentation, and, optionally, T2w to the derivative directory.

    If necessary, this workflow will also warp the images to standard space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.volume import init_postprocess_anat_wf

            with mock_config():
                wf = init_postprocess_anat_wf(
                    t1w_available=True,
                    t2w_available=True,
                    target_space="MNI152NLin6Asym",
                    name="postprocess_anat_wf",
                )

    Parameters
    ----------
    t1w_available : bool
        True if a preprocessed T1w is available, False if not.
    t2w_available : bool
        True if a preprocessed T2w is available, False if not.
    target_space : :obj:`str`
        Target NIFTI template for T1w.
    %(name)s
        Default is "postprocess_anat_wf".

    Inputs
    ------
    t1w : :obj:`str`
        Path to the preprocessed T1w file.
        This file may be in standard space or native T1w space.
    t2w : :obj:`str` or None
        Path to the preprocessed T2w file.
        This file may be in standard space or native T1w space.
    anat_dseg : :obj:`str`
        Path to the T1w segmentation file.
    %(anat_to_template_xfm)s
        We need to use MNI152NLin6Asym for the template.
    template : :obj:`str`
        The target template.

    Outputs
    -------
    t1w : :obj:`str`
        Path to the preprocessed T1w file in standard space.
    t2w : :obj:`str` or None
        Path to the preprocessed T2w file in standard space.
    """
    workflow = Workflow(name=name)
    output_dir = config.execution.xcp_d_dir
    input_type = config.workflow.input_type
    omp_nthreads = config.nipype.omp_nthreads

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t2w",
                "anat_dseg",
                "anat_to_template_xfm",
                "template",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w", "t2w"]),
        name="outputnode",
    )

    # Split cohort out of the space for MNIInfant templates.
    cohort = None
    if "+" in target_space:
        target_space, cohort = target_space.split("+")

    template_file = str(
        get_template(template=target_space, cohort=cohort, resolution=1, desc=None, suffix="T1w")
    )
    inputnode.inputs.template = template_file

    ds_anat_dseg_std = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            space=target_space,
            cohort=cohort,
            extension=".nii.gz",
        ),
        name="ds_anat_dseg_std",
        run_without_submitting=False,
    )

    workflow.connect([(inputnode, ds_anat_dseg_std, [("anat_dseg", "source_file")])])

    if t1w_available:
        ds_t1w_std = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=target_space,
                cohort=cohort,
                extension=".nii.gz",
            ),
            name="ds_t1w_std",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_t1w_std, [("t1w", "source_file")]),
            (ds_t1w_std, outputnode, [("out_file", "t1w")]),
        ])  # fmt:skip

    if t2w_available:
        ds_t2w_std = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=target_space,
                cohort=cohort,
                extension=".nii.gz",
            ),
            name="ds_t2w_std",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_t2w_std, [("t2w", "source_file")]),
            (ds_t2w_std, outputnode, [("out_file", "t2w")]),
        ])  # fmt:skip

    if input_type in ("dcan", "hcp", "ukb"):
        # Assume that the T1w, T1w segmentation, and T2w files are in standard space,
        # but don't have the "space" entity, for the "dcan" and "hcp" derivatives.
        # This is a bug, and the converted filenames are inaccurate, so we have this
        # workaround in place.
        workflow.connect([(inputnode, ds_anat_dseg_std, [("anat_dseg", "in_file")])])

        if t1w_available:
            workflow.connect([(inputnode, ds_t1w_std, [("t1w", "in_file")])])

        if t2w_available:
            workflow.connect([(inputnode, ds_t2w_std, [("t2w", "in_file")])])

    else:
        out = (
            ["T1w"] if t1w_available else [] + ["T2w"] if t2w_available else [] + ["segmentation"]
        )
        workflow.__desc__ = f"""\
Native-space {list_to_str(out)} images were transformed to {target_space} space at 1 mm3
resolution.
"""
        warp_anat_dseg_to_template = pe.Node(
            ApplyTransforms(
                num_threads=2,
                interpolation="GenericLabel",
                input_image_type=3,
                dimension=3,
            ),
            name="warp_anat_dseg_to_template",
            mem_gb=2,
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, warp_anat_dseg_to_template, [
                ("anat_dseg", "input_image"),
                ("anat_to_template_xfm", "transforms"),
                ("template", "reference_image"),
            ]),
            (warp_anat_dseg_to_template, ds_anat_dseg_std, [("output_image", "in_file")]),
        ])  # fmt:skip

        if t1w_available:
            # Warp the native T1w-space T1w, T1w segmentation, and T2w files to standard space.
            warp_t1w_to_template = pe.Node(
                ApplyTransforms(
                    num_threads=2,
                    interpolation="LanczosWindowedSinc",
                    input_image_type=3,
                    dimension=3,
                ),
                name="warp_t1w_to_template",
                mem_gb=2,
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (inputnode, warp_t1w_to_template, [
                    ("t1w", "input_image"),
                    ("anat_to_template_xfm", "transforms"),
                    ("template", "reference_image"),
                ]),
                (warp_t1w_to_template, ds_t1w_std, [("output_image", "in_file")]),
            ])  # fmt:skip

        if t2w_available:
            warp_t2w_to_template = pe.Node(
                ApplyTransforms(
                    num_threads=2,
                    interpolation="LanczosWindowedSinc",
                    input_image_type=3,
                    dimension=3,
                ),
                name="warp_t2w_to_template",
                mem_gb=2,
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (inputnode, warp_t2w_to_template, [
                    ("t2w", "input_image"),
                    ("anat_to_template_xfm", "transforms"),
                    ("template", "reference_image"),
                ]),
                (warp_t2w_to_template, ds_t2w_std, [("output_image", "in_file")]),
            ])  # fmt:skip

    if config.workflow.abcc_qc:
        execsummary_anatomical_plots_wf = init_execsummary_anatomical_plots_wf(
            t1w_available=t1w_available,
            t2w_available=t2w_available,
        )
        workflow.connect([
            (inputnode, execsummary_anatomical_plots_wf, [("template", "inputnode.template")]),
        ])  # fmt:skip

        if t1w_available:
            workflow.connect([
                (ds_t1w_std, execsummary_anatomical_plots_wf, [("out_file", "inputnode.t1w")]),
            ])  # fmt:skip

        if t2w_available:
            workflow.connect([
                (ds_t2w_std, execsummary_anatomical_plots_wf, [("out_file", "inputnode.t2w")]),
            ])  # fmt:skip

    return workflow
