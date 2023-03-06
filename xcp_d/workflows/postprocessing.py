# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing BOLD data."""
from nipype.interfaces import utility as niu
from nipype.interfaces.workbench.cifti import CiftiSmooth
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from num2words import num2words
from pkg_resources import resource_filename as pkgrf

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import Smooth
from xcp_d.interfaces.prepostcleaning import FlagMotionOutliers, RemoveDummyVolumes
from xcp_d.interfaces.restingstate import DespikePatch
from xcp_d.interfaces.workbench import CiftiConvert, FixCiftiIntent
from xcp_d.utils.confounds import consolidate_confounds
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.plotting import plot_design_matrix
from xcp_d.utils.utils import estimate_brain_radius, fwhm2sigma


def init_prepare_confounds_wf(
    output_dir,
    TR,
    params,
    dummy_scans,
    motion_filter_type,
    band_stop_min,
    band_stop_max,
    motion_filter_order,
    head_radius,
    fd_thresh,
    mem_gb,
    omp_nthreads,
    name="prepare_confounds_wf",
):
    """Prepare confounds.

    This workflow loads and consolidates confounds, removes dummy volumes,
    filters motion parameters, calculates framewise displacement, and flags outlier volumes.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.postprocessing import init_prepare_confounds_wf

            wf = init_prepare_confounds_wf(
                output_dir=".",
                TR=0.8,
                params="27P",
                dummy_scans="auto",
                motion_filter_type="notch",
                band_stop_min=12,
                band_stop_max=20,
                motion_filter_order=4,
                head_radius="auto",
                fd_thresh=0.2,
                mem_gb=0.1,
                omp_nthreads=1,
                name="prepare_confounds_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    %(TR)s
    %(params)s
    %(dummy_scans)s
    %(motion_filter_type)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(motion_filter_order)s
    %(head_radius)s
    %(fd_thresh)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "prepare_confounds_wf".

    Inputs
    ------
    %(name_source)s
    preprocessed_bold : :obj:`str`
    %(fmriprep_confounds_file)s
    %(custom_confounds_file)s
    %(t1w_mask)s
    %(dummy_scans)s
        Set from the parameter.

    Outputs
    -------
    preprocessed_bold : :obj:`str`
    %(fmriprep_confounds_file)s
    %(confounds_file)s
        The selected confounds, potentially including custom confounds, after dummy scan removal.
    %(dummy_scans)s
        If originally set to "auto", this output will have the actual number of dummy volumes.
    %(head_radius)s
        If originally set to "auto", this output will have the estimated head radius.
    %(filtered_motion)s
    filtered_motion_metadata : :obj:`dict`
    %(temporal_mask)s
    temporal_mask_metadata : :obj:`dict`
    """
    workflow = Workflow(name=name)

    dummy_scans_str = ""
    if dummy_scans == "auto":
        dummy_scans_str = (
            "Non-steady-state volumes were extracted from the preprocessed confounds "
            "and were discarded from both the BOLD data and nuisance regressors. "
        )
    elif dummy_scans > 0:
        dummy_scans_str = (
            f"The first {num2words(dummy_scans)} of both the BOLD data and nuisance "
            "regressors were discarded. "
        )
    workflow.__desc__ = f"{dummy_scans_str}. Then, outlier detection was performed."

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "preprocessed_bold",
                "fmriprep_confounds_file",
                "custom_confounds_file",
                "t1w_mask",
                "dummy_scans",
            ],
        ),
        name="inputnode",
    )
    inputnode.inputs.dummy_scans = dummy_scans

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "preprocessed_bold",
                "fmriprep_confounds_file",  # used to calculate motion in concatenation workflow
                "confounds_file",
                "dummy_scans",
                "head_radius",
                "filtered_motion",
                "filtered_motion_metadata",
                "temporal_mask",
                "temporal_mask_metadata",
            ],
        ),
        name="outputnode",
    )

    consolidate_confounds_node = pe.Node(
        niu.Function(
            input_names=[
                "img_file",
                "custom_confounds_file",
                "params",
            ],
            output_names=["out_file"],
            function=consolidate_confounds,
        ),
        name="consolidate_confounds",
    )
    consolidate_confounds_node.inputs.params = params

    # Load and filter confounds
    # fmt:off
    workflow.connect([
        (inputnode, consolidate_confounds_node, [
            ("name_source", "img_file"),
            ("custom_confounds_file", "custom_confounds_file"),
        ]),
    ])
    # fmt:on

    determine_head_radius = pe.Node(
        niu.Function(
            function=estimate_brain_radius,
            input_names=["mask_file", "head_radius"],
            output_names=["head_radius"],
        ),
        name="determine_head_radius",
    )
    determine_head_radius.inputs.head_radius = head_radius

    # fmt:off
    workflow.connect([
        (inputnode, determine_head_radius, [("t1w_mask", "mask_file")]),
        (determine_head_radius, outputnode, [("head_radius", "head_radius")]),
    ])
    # fmt:on

    flag_motion_outliers = pe.Node(
        FlagMotionOutliers(
            TR=TR,
            band_stop_min=band_stop_min,
            band_stop_max=band_stop_max,
            motion_filter_type=motion_filter_type,
            motion_filter_order=motion_filter_order,
            fd_thresh=fd_thresh,
        ),
        name="flag_motion_outliers",
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (determine_head_radius, flag_motion_outliers, [("head_radius", "head_radius")]),
        (flag_motion_outliers, outputnode, [
            ("filtered_motion", "filtered_motion"),
            ("filtered_motion_metadata", "filtered_motion_metadata"),
            ("temporal_mask", "temporal_mask"),
            ("temporal_mask_metadata", "temporal_mask_metadata"),
        ])
    ])
    # fmt:on

    plot_design_matrix_node = pe.Node(
        niu.Function(
            input_names=["design_matrix", "temporal_mask"],
            output_names=["design_matrix_figure"],
            function=plot_design_matrix,
        ),
        name="plot_design_matrix",
    )

    # fmt:off
    workflow.connect([
        (flag_motion_outliers, plot_design_matrix_node, [("temporal_mask", "temporal_mask")]),
    ])
    # fmt:on

    ds_design_matrix_plot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["space", "res", "den", "desc"],
            datatype="figures",
            suffix="design",
            extension=".svg",
        ),
        name="ds_design_matrix_plot",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_design_matrix_plot, [("name_source", "source_file")]),
        (plot_design_matrix_node, ds_design_matrix_plot, [("design_matrix_figure", "in_file")]),
    ])
    # fmt:on

    if dummy_scans:
        remove_dummy_scans = pe.Node(
            RemoveDummyVolumes(),
            name="remove_dummy_scans",
            mem_gb=2 * mem_gb,  # assume it takes a lot of memory
        )

        # fmt:off
        workflow.connect([
            (inputnode, remove_dummy_scans, [
                ("preprocessed_bold", "bold_file"),
                ("dummy_scans", "dummy_scans"),
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ]),
            (consolidate_confounds_node, remove_dummy_scans, [("out_file", "confounds_file")]),
            (remove_dummy_scans, flag_motion_outliers, [
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file_dropped_TR", "fmriprep_confounds_file"),
            ]),
            (remove_dummy_scans, plot_design_matrix_node, [
                ("confounds_file_dropped_TR", "design_matrix"),
            ]),
            (remove_dummy_scans, outputnode, [
                ("bold_file_dropped_TR", "preprocessed_bold"),
                ("fmriprep_confounds_file_dropped_TR", "fmriprep_confounds_file"),
                ("confounds_file_dropped_TR", "confounds_file"),
                ("dummy_scans", "dummy_scans"),
            ]),
        ])
        # fmt:on

    else:
        # fmt:off
        workflow.connect([
            (inputnode, flag_motion_outliers, [
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ]),
            (inputnode, outputnode, [
                ("bold_file", "preprocessed_bold"),
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
                ("dummy_scans", "dummy_scans"),
            ]),
            (consolidate_confounds_node, outputnode, [("confounds_file", "confounds_file")]),
            (consolidate_confounds_node, plot_design_matrix_node, [("out_file", "design_matrix")]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_despike_wf(
    TR,
    cifti,
    mem_gb,
    omp_nthreads,
    name="despike_wf",
):
    """Despike BOLD data with 3dDespike.

    Despiking truncates large spikes in the BOLD times series.
    Despiking reduces/limits the amplitude or magnitude of large spikes,
    but preserves those data points with an imputed reduced amplitude.
    Despiking is done before regression and filtering to minimize the impact of spikes.
    Despiking is applied to whole volumes and data, and different from temporal censoring.
    It can be added to the command line arguments with ``--despike``.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.postprocessing import init_despike_wf

            wf = init_despike_wf(
                TR=0.8,
                cifti=True,
                mem_gb=0.1,
                omp_nthreads=1,
                name="despike_wf",
            )

    Parameters
    ----------
    %(TR)s
    %(cifti)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "despike_wf".

    Inputs
    ------
    bold_file : str
        A NIFTI or CIFTI BOLD file to despike.

    Outputs
    -------
    bold_file : str
        The despiked NIFTI or CIFTI BOLD file.
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="outputnode")

    despike3d = pe.Node(
        DespikePatch(outputtype="NIFTI_GZ", args="-nomask -NEW"),
        name="despike3d",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    if cifti:
        workflow.__desc__ = (
            "The BOLD data were converted to NIfTI format, despiked with 3dDespike, "
            "and converted back to CIFTI format."
        )

        # first, convert the cifti to a nifti
        convert_to_nifti = pe.Node(
            CiftiConvert(target="to"),
            name="convert_to_nifti",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, convert_to_nifti, [("bold_file", "in_file")]),
            (convert_to_nifti, despike3d, [("out_file", "in_file")]),
        ])
        # fmt:on

        # finally, convert the despiked nifti back to cifti
        convert_to_cifti = pe.Node(
            CiftiConvert(target="from", TR=TR),
            name="convert_to_cifti",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, convert_to_cifti, [("bold_file", "cifti_template")]),
            (despike3d, convert_to_cifti, [("out_file", "in_file")]),
            (convert_to_cifti, outputnode, [("out_file", "bold_file")]),
        ])
        # fmt:on

    else:
        workflow.__desc__ = "The BOLD data were despiked with 3dDespike."

        # fmt:off
        workflow.connect([
            (inputnode, despike3d, [("bold_file", "in_file")]),
            (despike3d, outputnode, [("out_file", "bold_file")]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_resd_smoothing_wf(
    smoothing,
    cifti,
    mem_gb,
    omp_nthreads,
    name="resd_smoothing_wf",
):
    """Smooth BOLD residuals.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.postprocessing import init_resd_smoothing_wf

            wf = init_resd_smoothing_wf(
                smoothing=6,
                cifti=False,
                mem_gb=0.1,
                omp_nthreads=1,
                name="resd_smoothing_wf",
            )

    Parameters
    ----------
    smoothing
    %(cifti)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "resd_smoothing_wf".

    Inputs
    ------
    bold_file

    Outputs
    -------
    smoothed_bold
    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["bold_file"]), name="inputnode")
    outputnode = pe.Node(niu.IdentityInterface(fields=["smoothed_bold"]), name="outputnode")

    # Turn specified FWHM (Full-Width at Half Maximum) to standard deviation.
    sigma_lx = fwhm2sigma(smoothing)
    if cifti:
        workflow.__desc__ = f""" \
The processed BOLD  was smoothed using Connectome Workbench with a gaussian kernel
size of {str(smoothing)} mm  (FWHM).
"""

        # Call connectome workbench to smooth for each hemisphere
        smooth_data = pe.Node(
            CiftiSmooth(
                sigma_surf=sigma_lx,  # the size of the surface kernel
                sigma_vol=sigma_lx,  # the volume of the surface kernel
                direction="COLUMN",  # which direction to smooth along@
                right_surf=pkgrf(  # pull out atlases for each hemisphere
                    "xcp_d",
                    (
                        "data/ciftiatlas/"
                        "Q1-Q6_RelatedParcellation210.R.midthickness_32k_fs_LR.surf.gii"
                    ),
                ),
                left_surf=pkgrf(
                    "xcp_d",
                    (
                        "data/ciftiatlas/"
                        "Q1-Q6_RelatedParcellation210.L.midthickness_32k_fs_LR.surf.gii"
                    ),
                ),
            ),
            name="cifti_smoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # Always check the intent code in CiftiSmooth's output file
        fix_cifti_intent = pe.Node(
            FixCiftiIntent(),
            name="fix_cifti_intent",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (smooth_data, fix_cifti_intent, [("out_file", "in_file")]),
            (fix_cifti_intent, outputnode, [("out_file", "smoothed_bold")]),
        ])
        # fmt:on

    else:
        workflow.__desc__ = f""" \
The processed BOLD was smoothed using Nilearn with a gaussian kernel size of {str(smoothing)} mm
(FWHM).
"""
        # Use nilearn to smooth the image
        smooth_data = pe.Node(
            Smooth(fwhm=smoothing),  # FWHM = kernel size
            name="nifti_smoothing",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (smooth_data, outputnode, [("out_file", "smoothed_bold")]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (inputnode, smooth_data, [("bold_file", "in_file")]),
    ])
    # fmt:on

    return workflow
