# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing the BOLD data."""
import os

import nibabel as nb
import numpy as np
from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from num2words import num2words

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import DenoiseNifti
from xcp_d.interfaces.prepostcleaning import (
    Censor,
    ConvertTo32,
    FlagMotionOutliers,
    RemoveDummyVolumes,
)
from xcp_d.utils.confounds import (
    consolidate_confounds,
    describe_censoring,
    describe_regression,
    get_custom_confounds,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.filemanip import check_binary_mask
from xcp_d.utils.plotting import plot_design_matrix
from xcp_d.utils.utils import estimate_brain_radius
from xcp_d.workflows.connectivity import init_functional_connectivity_nifti_wf
from xcp_d.workflows.execsummary import init_execsummary_functional_plots_wf
from xcp_d.workflows.outputs import init_postproc_derivatives_wf
from xcp_d.workflows.plotting import init_qc_report_wf
from xcp_d.workflows.postprocessing import init_despike_wf, init_resd_smoothing_wf
from xcp_d.workflows.restingstate import init_alff_wf, init_reho_nifti_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_postprocess_nifti_wf(
    bold_file,
    bandpass_filter,
    high_pass,
    low_pass,
    bpf_order,
    motion_filter_type,
    motion_filter_order,
    band_stop_min,
    band_stop_max,
    smoothing,
    head_radius,
    params,
    output_dir,
    custom_confounds_folder,
    dummytime,
    dummy_scans,
    fd_thresh,
    despike,
    dcan_qc,
    run_data,
    n_runs,
    min_coverage,
    omp_nthreads,
    layout=None,
    name="bold_postprocess_wf",
):
    """Organize the bold processing workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            import os

            from xcp_d.utils.bids import collect_data, collect_run_data
            from xcp_d.workflows.bold import init_postprocess_nifti_wf
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()

            layout, subj_data = collect_data(
                bids_dir=fmri_dir,
                input_type="fmriprep",
                participant_label="01",
                task="imagery",
                bids_validate=False,
                cifti=False,
            )

            bold_file = subj_data["bold"][0]

            run_data = collect_run_data(
                layout=layout,
                input_type="fmriprep",
                bold_file=bold_file,
                cifti=False,
            )

            custom_confounds_folder = os.path.join(fmri_dir, "sub-01/func")

            wf = init_postprocess_nifti_wf(
                bold_file=bold_file,
                bandpass_filter=True,
                high_pass=0.01,
                low_pass=0.08,
                bpf_order=2,
                motion_filter_type="notch",
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                smoothing=6,
                head_radius=50.,
                params="27P",
                output_dir=".",
                custom_confounds_folder=custom_confounds_folder,
                dummytime=0,
                dummy_scans=2,
                fd_thresh=0.2,
                despike=True,
                dcan_qc=True,
                run_data=run_data,
                n_runs=1,
                min_coverage=0.5,
                omp_nthreads=1,
                layout=layout,
                name="nifti_postprocess_wf",
            )
            wf.inputs.inputnode.t1w = subj_data["t1w"]
            wf.inputs.inputnode.template_to_t1w_xfm = subj_data["template_to_t1w_xfm"]

    Parameters
    ----------
    bold_file: str
        bold file for post processing
    %(bandpass_filter)s
    %(high_pass)s
    %(low_pass)s
    %(bpf_order)s
    %(motion_filter_type)s
    %(motion_filter_order)s
    %(band_stop_min)s
    %(band_stop_max)s
    %(smoothing)s
    %(head_radius)s
    %(params)s
    %(output_dir)s
    %(custom_confounds_folder)s
    %(dummytime)s
    %(dummy_scans)s
    %(fd_thresh)s
    %(despike)s
    %(dcan_qc)s
    run_data : dict
    n_runs
        Number of runs being postprocessed by XCP-D.
        This is just used for the boilerplate, as this workflow only posprocesses one run.
    %(min_coverage)s
    %(omp_nthreads)s
    %(layout)s
    %(name)s
        Default is "nifti_postprocess_wf".

    Inputs
    ------
    bold_file
        BOLD series NIfTI file
    %(boldref)s
        Loaded in this workflow.
    bold_mask
        bold_mask from fmriprep
        Loaded in this workflow.
    %(custom_confounds_file)s
    %(template_to_t1w_xfm)s
        Fed from the subject workflow.
    t1w
        Preprocessed T1w image, warped to standard space.
        Fed from the subject workflow.
    t2w
        Preprocessed T2w image, warped to standard space.
        Fed from the subject workflow.
    t1seg
    t1w_mask
        T1w brain mask, used to estimate head/brain radius.
        Fed from the subject workflow.
    %(fmriprep_confounds_file)s
        Loaded in this workflow.
    %(t1w_to_native_xfm)s
    %(dummy_scans)s

    Outputs
    -------
    %(name_source)s
    preprocessed_bold : str
        The preprocessed BOLD file, after dummy scan removal.
    %(filtered_motion)s
    %(temporal_mask)s
    %(fmriprep_confounds_file)s
        After dummy scan removal.
    %(uncensored_denoised_bold)s
    %(interpolated_filtered_bold)s
    %(smoothed_denoised_bold)s
    %(boldref)s
    bold_mask
    %(t1w_to_native_xfm)s
    %(atlas_names)s
    %(timeseries)s
    %(timeseries_ciftis)s
        This will not be defined.

    References
    ----------
    .. footbibliography::
    """
    workflow = Workflow(name=name)

    TR = run_data["bold_metadata"]["RepetitionTime"]

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "boldref",
                "bold_mask",
                "custom_confounds_file",
                "template_to_t1w_xfm",
                "t1w",
                "t2w",
                "t1seg",
                "t1w_mask",
                "fmriprep_confounds_file",
                "t1w_to_native_xfm",
                "dummy_scans",
            ],
        ),
        name="inputnode",
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.boldref = run_data["boldref"]
    inputnode.inputs.fmriprep_confounds_file = run_data["confounds"]
    inputnode.inputs.t1w_to_native_xfm = run_data["t1w_to_native_xfm"]
    inputnode.inputs.dummy_scans = dummy_scans

    # TODO: This is a workaround for a bug in nibabies.
    # Once https://github.com/nipreps/nibabies/issues/245 is resolved
    # and a new release is made, remove this.
    mask_file = check_binary_mask(run_data["boldmask"])
    inputnode.inputs.bold_mask = mask_file

    # Load custom confounds
    # We need to run this function directly to access information in the confounds that is
    # used for the boilerplate.
    custom_confounds_file = get_custom_confounds(
        custom_confounds_folder,
        run_data["confounds"],
    )
    inputnode.inputs.custom_confounds_file = custom_confounds_file

    regression_description = describe_regression(params, custom_confounds_file)
    censoring_description = describe_censoring(
        motion_filter_type=motion_filter_type,
        motion_filter_order=motion_filter_order,
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        head_radius=head_radius,
        fd_thresh=fd_thresh,
    )

    if dummy_scans == 0 and dummytime != 0:
        dummy_scans = int(np.ceil(dummytime / TR))

    dummy_scans_str = ""
    if dummy_scans == "auto":
        dummy_scans_str = (
            "non-steady-state volumes were extracted from the preprocessed confounds "
            "and were discarded from both the BOLD data and nuisance regressors, then"
        )
    elif dummy_scans > 0:
        dummy_scans_str = (
            f"the first {num2words(dummy_scans)} of both the BOLD data and nuisance "
            "regressors were discarded, then "
        )

    bandpass_str = ""
    if bandpass_filter:
        bandpass_str = (
            "Any volumes censored earlier in the workflow were then interpolated in the residual "
            "time series produced by the regression. "
            "The interpolated timeseries were then band-pass filtered using a(n) "
            f"{num2words(bpf_order, ordinal=True)}-order Butterworth filter, "
            f"in order to retain signals within the {high_pass}-{low_pass} Hz frequency band. "
            "The filtered, interpolated time series were then re-censored to remove high-motion "
            "outlier volumes."
        )

    workflow.__desc__ = f"""\
For each of the {num2words(n_runs)} BOLD runs found per subject (across all tasks and sessions),
the following post-processing was performed.
First, {dummy_scans_str}outlier detection was performed.
{censoring_description}
{regression_description}
{bandpass_str}
"""

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "preprocessed_bold",
                "fmriprep_confounds_file",
                "filtered_motion",
                "temporal_mask",
                "uncensored_denoised_bold",
                "interpolated_filtered_bold",
                "censored_denoised_bold",
                "smoothed_denoised_bold",
                "boldref",
                "bold_mask",
                "t1w_to_native_xfm",
                "atlas_names",
                "timeseries",
                "timeseries_ciftis",  # will not be defined
            ],
        ),
        name="outputnode",
    )

    mem_gbx = _create_mem_gb(bold_file)

    downcast_data = pe.Node(
        ConvertTo32(),
        name="downcast_data",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, outputnode, [
            ("bold_file", "name_source"),
            ("t1w_to_native_xfm", "t1w_to_native_xfm"),
        ]),
        (inputnode, downcast_data, [
            ("bold_file", "bold_file"),
            ("boldref", "boldref"),
            ("bold_mask", "bold_mask"),
            ("t1w_mask", "t1w_mask"),
        ]),
        (downcast_data, outputnode, [
            ("bold_mask", "bold_mask"),
            ("boldref", "boldref"),
        ]),
    ])
    # fmt:on

    determine_head_radius = pe.Node(
        Function(
            function=estimate_brain_radius,
            input_names=["mask_file", "head_radius"],
            output_names=["head_radius"],
        ),
        name="determine_head_radius",
    )
    determine_head_radius.inputs.head_radius = head_radius

    # fmt:off
    workflow.connect([(downcast_data, determine_head_radius, [("t1w_mask", "mask_file")])])
    # fmt:on

    connectivity_wf = init_functional_connectivity_nifti_wf(
        output_dir=output_dir,
        min_coverage=min_coverage,
        mem_gb=mem_gbx["timeseries"],
        name="connectivity_wf",
        omp_nthreads=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (connectivity_wf, outputnode, [
            ("outputnode.atlas_names", "atlas_names"),
            ("outputnode.timeseries", "timeseries"),
        ]),
    ])
    # fmt:on

    if bandpass_filter:
        alff_wf = init_alff_wf(
            bold_file=bold_file,
            output_dir=output_dir,
            TR=TR,
            low_pass=low_pass,
            high_pass=high_pass,
            smoothing=smoothing,
            cifti=False,
            mem_gb=mem_gbx["timeseries"],
            omp_nthreads=omp_nthreads,
            name="alff_wf",
        )

    reho_wf = init_reho_nifti_wf(
        bold_file=bold_file,
        output_dir=output_dir,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        name="reho_wf",
    )

    postproc_derivatives_wf = init_postproc_derivatives_wf(
        smoothing=smoothing,
        bold_file=bold_file,
        bandpass_filter=bandpass_filter,
        params=params,
        cifti=False,
        dcan_qc=dcan_qc,
        output_dir=output_dir,
        low_pass=low_pass,
        high_pass=high_pass,
        motion_filter_type=motion_filter_type,
        TR=TR,
        name="postproc_derivatives_wf",
    )

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
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
    )

    resd_smoothing_wf = init_resd_smoothing_wf(
        mem_gb=mem_gbx["timeseries"],
        smoothing=smoothing,
        cifti=False,
        name="resd_smoothing_wf",
        omp_nthreads=omp_nthreads,
    )

    denoise_bold = pe.Node(
        DenoiseNifti(
            TR=TR,
            low_pass=low_pass,
            high_pass=high_pass,
            filter_order=bpf_order,
            bandpass_filter=bandpass_filter,
        ),
        name="denoise_bold",
        mem_gb=mem_gbx["timeseries"],
        n_procs=omp_nthreads,
    )

    censor_interpolated_data = pe.Node(
        Censor(),
        name="censor_interpolated_data",
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (denoise_bold, censor_interpolated_data, [("interpolated_filtered_bold", "in_file")]),
        (flag_motion_outliers, censor_interpolated_data, [("temporal_mask", "temporal_mask")]),
        (censor_interpolated_data, outputnode, [
            ("censored_denoised_bold", "censored_denoised_bold"),
        ]),
    ])
    # fmt:on

    consolidate_confounds_node = pe.Node(
        Function(
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
            ("bold_file", "img_file"),
            ("custom_confounds_file", "custom_confounds_file"),
        ]),
    ])
    # fmt:on

    plot_design_matrix_node = pe.Node(
        Function(
            input_names=["design_matrix", "temporal_mask"],
            output_names=["design_matrix_figure"],
            function=plot_design_matrix,
        ),
        name="plot_design_matrix",
    )

    qc_report_wf = init_qc_report_wf(
        output_dir=output_dir,
        TR=TR,
        motion_filter_type=motion_filter_type,
        fd_thresh=fd_thresh,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        dcan_qc=dcan_qc,
        cifti=False,
        name="qc_report_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [
            ("bold_file", "inputnode.name_source"),
            ("boldref", "inputnode.boldref"),
            ("bold_mask", "inputnode.bold_mask"),
            ("t1w_mask", "inputnode.t1w_mask"),
            ("template_to_t1w_xfm", "inputnode.template_to_t1w_xfm"),
            ("t1w_to_native_xfm", "inputnode.t1w_to_native_xfm"),
        ]),
        (determine_head_radius, qc_report_wf, [("head_radius", "inputnode.head_radius")]),
        (denoise_bold, qc_report_wf, [
            ("uncensored_denoised_bold", "inputnode.uncensored_denoised_bold"),
        ]),
        (denoise_bold, outputnode, [
            ("uncensored_denoised_bold", "uncensored_denoised_bold"),
            ("interpolated_filtered_bold", "interpolated_filtered_bold"),
        ]),
    ])
    # fmt:on

    # Remove TR first:
    if dummy_scans:
        remove_dummy_scans = pe.Node(
            RemoveDummyVolumes(),
            name="remove_dummy_scans",
            mem_gb=2 * mem_gbx["timeseries"],  # assume it takes a lot of memory
        )

        # fmt:off
        workflow.connect([
            (inputnode, remove_dummy_scans, [
                ("dummy_scans", "dummy_scans"),
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ]),
            (downcast_data, remove_dummy_scans, [("bold_file", "bold_file")]),
            (consolidate_confounds_node, remove_dummy_scans, [("out_file", "confounds_file")]),
            (remove_dummy_scans, outputnode, [
                ("bold_file_dropped_TR", "preprocessed_bold"),
                ("fmriprep_confounds_file_dropped_TR", "fmriprep_confounds_file"),
            ]),
            (remove_dummy_scans, flag_motion_outliers, [
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file_dropped_TR", "fmriprep_confounds_file"),
            ]),
            (remove_dummy_scans, denoise_bold, [("confounds_file_dropped_TR", "confounds_file")]),
            (remove_dummy_scans, qc_report_wf, [
                ("bold_file_dropped_TR", "inputnode.preprocessed_bold"),
                ("dummy_scans", "inputnode.dummy_scans"),
                ("fmriprep_confounds_file_dropped_TR", "inputnode.fmriprep_confounds_file"),
            ]),
            (remove_dummy_scans, plot_design_matrix_node, [
                ("confounds_file_dropped_TR", "design_matrix"),
            ]),
        ])
        # fmt:on

    else:
        # fmt:off
        workflow.connect([
            (inputnode, qc_report_wf, [
                ("bold_file", "inputnode.preprocessed_bold"),
                ("dummy_scans", "inputnode.dummy_scans"),
                ("fmriprep_confounds_file", "inputnode.fmriprep_confounds_file"),
            ]),
            (inputnode, flag_motion_outliers, [
                # fMRIPrep confounds file is needed for filtered motion.
                # The selected confounds are not guaranteed to include motion params.
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ]),
            (inputnode, outputnode, [
                ("bold_file", "preprocessed_bold"),
                ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ]),
            (consolidate_confounds_node, denoise_bold, [("out_file", "confounds_file")]),
            (consolidate_confounds_node, plot_design_matrix_node, [("out_file", "design_matrix")]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (determine_head_radius, flag_motion_outliers, [("head_radius", "head_radius")]),
        (flag_motion_outliers, plot_design_matrix_node, [("temporal_mask", "temporal_mask")]),
        (flag_motion_outliers, outputnode, [
            ("filtered_motion", "filtered_motion"),
            ("temporal_mask", "temporal_mask"),
        ]),
    ])
    # fmt:on

    if despike:
        despike_wf = init_despike_wf(
            TR=TR,
            cifti=False,
            mem_gb=mem_gbx["timeseries"],
            omp_nthreads=omp_nthreads,
            name="despike_wf",
        )

        # fmt:off
        if dummy_scans:
            workflow.connect([
                (remove_dummy_scans, despike_wf, [
                    ("bold_file_dropped_TR", "inputnode.bold_file"),
                ]),
            ])
        else:
            workflow.connect([(downcast_data, despike_wf, [("bold_file", "inputnode.bold_file")])])

        workflow.connect([
            (despike_wf, denoise_bold, [("outputnode.bold_file", "preprocessed_bold")]),
        ])
        # fmt:on

    elif dummy_scans:
        # fmt:off
        workflow.connect([
            (remove_dummy_scans, denoise_bold, [("bold_file_dropped_TR", "preprocessed_bold")]),
        ])
        # fmt:on
    else:
        # fmt:off
        workflow.connect([
            (downcast_data, denoise_bold, [("bold_file", "preprocessed_bold")]),
        ])
        # fmt:on

    # fmt:off
    workflow.connect([
        (downcast_data, denoise_bold, [("bold_mask", "mask")]),
        (flag_motion_outliers, denoise_bold, [("temporal_mask", "temporal_mask")]),
    ])
    # fmt:on

    # residual smoothing
    # fmt:off
    workflow.connect([
        (censor_interpolated_data, resd_smoothing_wf, [
            ("censored_denoised_bold", "inputnode.bold_file"),
        ]),
    ])
    # fmt:on

    # functional connect workflow
    # fmt:off
    workflow.connect([
        (downcast_data, connectivity_wf, [
            ("bold_file", "inputnode.bold_file"),
            ("bold_mask", "inputnode.bold_mask"),
            ("boldref", "inputnode.boldref"),
        ]),
        (inputnode, connectivity_wf, [
            ("template_to_t1w_xfm", "inputnode.template_to_t1w_xfm"),
            ("t1w_to_native_xfm", "inputnode.t1w_to_native_xfm"),
        ]),
        (censor_interpolated_data, connectivity_wf, [
            ("censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
    ])

    # reho and alff
    workflow.connect([
        (downcast_data, reho_wf, [("bold_mask", "inputnode.bold_mask")]),
        (censor_interpolated_data, reho_wf, [
            ("censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
    ])

    if bandpass_filter:
        workflow.connect([
            (downcast_data, alff_wf, [("bold_mask", "inputnode.bold_mask")]),
            (censor_interpolated_data, alff_wf, [
                ("censored_denoised_bold", "inputnode.denoised_bold"),
            ]),
        ])

    # qc report
    workflow.connect([
        (flag_motion_outliers, qc_report_wf, [
            ("temporal_mask", "inputnode.temporal_mask"),
            ("filtered_motion", "inputnode.filtered_motion"),
        ]),
        (denoise_bold, qc_report_wf, [
            ("interpolated_filtered_bold", "inputnode.interpolated_filtered_bold"),
        ]),
        (censor_interpolated_data, qc_report_wf, [
            ("censored_denoised_bold", "inputnode.censored_denoised_bold"),
        ]),
    ])
    # fmt:on

    # write derivatives
    # fmt:off
    workflow.connect([
        (consolidate_confounds_node, postproc_derivatives_wf, [
            ("out_file", "inputnode.confounds_file"),
        ]),
        (denoise_bold, postproc_derivatives_wf, [
            ("interpolated_filtered_bold", "inputnode.interpolated_filtered_bold"),
        ]),
        (censor_interpolated_data, postproc_derivatives_wf, [
            ("censored_denoised_bold", "inputnode.censored_denoised_bold"),
        ]),
        (qc_report_wf, postproc_derivatives_wf, [("outputnode.qc_file", "inputnode.qc_file")]),
        (resd_smoothing_wf, outputnode, [("outputnode.smoothed_bold", "smoothed_denoised_bold")]),
        (resd_smoothing_wf, postproc_derivatives_wf, [
            ("outputnode.smoothed_bold", "inputnode.smoothed_bold"),
        ]),
        (flag_motion_outliers, postproc_derivatives_wf, [
            ("filtered_motion", "inputnode.filtered_motion"),
            ("filtered_motion_metadata", "inputnode.filtered_motion_metadata"),
            ("temporal_mask", "inputnode.temporal_mask"),
            ("tmask_metadata", "inputnode.tmask_metadata"),
        ]),
        (reho_wf, postproc_derivatives_wf, [("outputnode.reho", "inputnode.reho")]),
        (connectivity_wf, postproc_derivatives_wf, [
            ("outputnode.atlas_names", "inputnode.atlas_names"),
            ("outputnode.correlations", "inputnode.correlations"),
            ("outputnode.timeseries", "inputnode.timeseries"),
            ("outputnode.coverage", "inputnode.coverage"),
        ]),
    ])
    # fmt:on

    if bandpass_filter:
        # fmt:off
        workflow.connect([
            (alff_wf, postproc_derivatives_wf, [
                ("outputnode.alff", "inputnode.alff"),
                ("outputnode.smoothed_alff", "inputnode.smoothed_alff"),
            ]),
        ])
        # fmt:on

    ds_design_matrix_plot = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            dismiss_entities=["space", "res", "den", "desc"],
            datatype="figures",
            suffix="design",
            extension=".svg",
        ),
        name="ds_design_matrix_plot",
        run_without_submitting=False,
    )

    ds_report_connectivity = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            desc="connectivityplot",
            datatype="figures",
        ),
        name="ds_report_connectivity",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (plot_design_matrix_node, ds_design_matrix_plot, [("design_matrix_figure", "in_file")]),
        (connectivity_wf, ds_report_connectivity, [("outputnode.connectplot", "in_file")]),
    ])
    # fmt:on

    # executive summary workflow
    if dcan_qc:
        execsummary_functional_plots_wf = init_execsummary_functional_plots_wf(
            preproc_nifti=bold_file,
            t1w_available=True,
            t2w_available=False,
            output_dir=output_dir,
            layout=layout,
            name="execsummary_functional_plots_wf",
        )

        # fmt:off
        workflow.connect([
            # Use inputnode for executive summary instead of downcast_data
            # because T1w is used as name source.
            (inputnode, execsummary_functional_plots_wf, [
                ("boldref", "inputnode.boldref"),
                ("t1w", "inputnode.t1w"),
                ("t2w", "inputnode.t2w"),
            ]),
        ])
        # fmt:on

    return workflow


def _create_mem_gb(bold_fname):
    bold_size_gb = os.path.getsize(bold_fname) / (1024**3)
    bold_tlen = nb.load(bold_fname).shape[-1]
    mem_gbz = {
        "derivative": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "timeseries": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }

    if mem_gbz["timeseries"] < 4.0:
        mem_gbz["timeseries"] = 6.0
        mem_gbz["resampled"] = 2
    elif mem_gbz["timeseries"] > 8.0:
        mem_gbz["timeseries"] = 8.0
        mem_gbz["resampled"] = 3

    return mem_gbz
