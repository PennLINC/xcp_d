# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for post-processing CIFTI-format BOLD data."""
import os

import nibabel as nb
from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from num2words import num2words

from xcp_d.interfaces.utils import ConvertTo32
from xcp_d.utils.confounds import get_custom_confounds
from xcp_d.utils.doc import fill_doc
from xcp_d.workflows.connectivity import init_functional_connectivity_cifti_wf
from xcp_d.workflows.execsummary import init_execsummary_functional_plots_wf
from xcp_d.workflows.outputs import init_postproc_derivatives_wf
from xcp_d.workflows.plotting import init_qc_report_wf
from xcp_d.workflows.postprocessing import (
    init_denoise_bold_wf,
    init_despike_wf,
    init_prepare_confounds_wf,
)
from xcp_d.workflows.restingstate import init_alff_wf, init_reho_cifti_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_postprocess_cifti_wf(
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
    dummy_scans,
    fd_thresh,
    despike,
    dcan_qc,
    run_data,
    t1w_available,
    t2w_available,
    n_runs,
    min_coverage,
    omp_nthreads,
    layout=None,
    name="cifti_postprocess_wf",
):
    """Organize the cifti processing workflow.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            import os

            from xcp_d.utils.bids import collect_data, collect_run_data
            from xcp_d.workflows.cifti import init_postprocess_cifti_wf
            from xcp_d.utils.doc import download_example_data

            fmri_dir = download_example_data()

            layout, subj_data = collect_data(
                bids_dir=fmri_dir,
                input_type="fmriprep",
                participant_label="01",
                task="imagery",
                bids_validate=False,
                cifti=True,
            )

            bold_file = subj_data["bold"][0]
            custom_confounds_folder = os.path.join(fmri_dir, "sub-01/func")
            run_data = collect_run_data(
                layout=layout,
                input_type="fmriprep",
                bold_file=bold_file,
                cifti=True,
                primary_anat="T1w",
            )

            wf = init_postprocess_cifti_wf(
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
                dummy_scans=2,
                fd_thresh=0.2,
                despike=True,
                dcan_qc=True,
                run_data=run_data,
                t1w_available=True,
                t2w_available=True,
                n_runs=1,
                min_coverage=0.5,
                omp_nthreads=1,
                layout=layout,
                name="cifti_postprocess_wf",
            )
            wf.inputs.inputnode.t1w = subj_data["t1w"]

    Parameters
    ----------
    bold_file
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
        This will already be estimated before this workflow.
    %(params)s
    %(output_dir)s
    %(custom_confounds_folder)s
    %(dummy_scans)s
    %(fd_thresh)s
    %(despike)s
    %(dcan_qc)s
    run_data : dict
    t1w_available
    t2w_available
    n_runs
        Number of runs being postprocessed by XCP-D.
        This is just used for the boilerplate, as this workflow only posprocesses one run.
    %(min_coverage)s
    %(omp_nthreads)s
    %(layout)s
    %(name)s
        Default is "cifti_postprocess_wf".

    Inputs
    ------
    bold_file
        CIFTI file
    %(boldref)s
    %(custom_confounds_file)s
    t1w
        Preprocessed T1w image, warped to standard space.
        Fed from the subject workflow.
    t2w
        Preprocessed T2w image, warped to standard space.
        Fed from the subject workflow.
    %(fmriprep_confounds_file)s
    %(dummy_scans)s

    Outputs
    -------
    %(name_source)s
    preprocessed_bold : :obj:`str`
        The preprocessed BOLD file, after dummy scan removal.
    %(fmriprep_confounds_file)s
        After dummy scan removal.
    %(filtered_motion)s
    %(temporal_mask)s
    %(uncensored_denoised_bold)s
    %(interpolated_filtered_bold)s
    %(censored_denoised_bold)s
    %(smoothed_denoised_bold)s
    %(boldref)s
    bold_mask
        This will not be defined.
    %(anat_to_native_xfm)s
        This will not be defined.
    %(atlas_names)s
    %(timeseries)s
    %(timeseries_ciftis)s

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
                "custom_confounds_file",
                "t1w",
                "t2w",
                "fmriprep_confounds_file",
                "dummy_scans",
            ],
        ),
        name="inputnode",
    )

    inputnode.inputs.bold_file = bold_file
    inputnode.inputs.boldref = run_data["boldref"]
    inputnode.inputs.fmriprep_confounds_file = run_data["confounds"]
    inputnode.inputs.dummy_scans = dummy_scans

    # Load custom confounds
    # We need to run this function directly to access information in the confounds that is
    # used for the boilerplate.
    custom_confounds_file = get_custom_confounds(
        custom_confounds_folder,
        run_data["confounds"],
    )

    workflow = Workflow(name=name)

    workflow.__desc__ = (
        f"For each of the {num2words(n_runs)} BOLD runs found per subject "
        "(across all tasks and sessions), the following post-processing was performed."
    )

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
                "bold_mask",  # will not be defined
                "anat_to_native_xfm",  # will not be defined
                "atlas_names",
                "timeseries",
                "timeseries_ciftis",
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
            ("boldref", "boldref"),
        ]),
        (inputnode, downcast_data, [("bold_file", "bold_file")]),
    ])
    # fmt:on

    prepare_confounds_wf = init_prepare_confounds_wf(
        output_dir=output_dir,
        TR=TR,
        params=params,
        dummy_scans=dummy_scans,
        motion_filter_type=motion_filter_type,
        band_stop_min=band_stop_min,
        band_stop_max=band_stop_max,
        motion_filter_order=motion_filter_order,
        head_radius=head_radius,
        fd_thresh=fd_thresh,
        custom_confounds_file=custom_confounds_file,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        name="prepare_confounds_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, prepare_confounds_wf, [
            ("bold_file", "inputnode.name_source"),
            ("fmriprep_confounds_file", "inputnode.fmriprep_confounds_file"),
        ]),
        (downcast_data, prepare_confounds_wf, [
            ("bold_file", "inputnode.preprocessed_bold"),
        ]),
        (prepare_confounds_wf, outputnode, [
            ("outputnode.filtered_motion", "filtered_motion"),
            ("outputnode.temporal_mask", "temporal_mask"),
            ("outputnode.fmriprep_confounds_file", "fmriprep_confounds_file"),
            ("outputnode.preprocessed_bold", "preprocessed_bold"),
        ]),
    ])
    # fmt:on

    denoise_bold_wf = init_denoise_bold_wf(
        TR=TR,
        low_pass=low_pass,
        high_pass=high_pass,
        bpf_order=bpf_order,
        bandpass_filter=bandpass_filter,
        smoothing=smoothing,
        cifti=True,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        name="denoise_bold_wf",
    )

    # fmt:off
    workflow.connect([
        (prepare_confounds_wf, denoise_bold_wf, [
            ("outputnode.temporal_mask", "inputnode.temporal_mask"),
            ("outputnode.confounds_file", "inputnode.confounds_file"),
        ]),
        (denoise_bold_wf, outputnode, [
            ("outputnode.uncensored_denoised_bold", "uncensored_denoised_bold"),
            ("outputnode.interpolated_filtered_bold", "interpolated_filtered_bold"),
            ("outputnode.censored_denoised_bold", "censored_denoised_bold"),
            ("outputnode.smoothed_denoised_bold", "smoothed_denoised_bold"),
        ]),
    ])
    # fmt:on

    if despike:
        despike_wf = init_despike_wf(
            TR=TR,
            cifti=True,
            mem_gb=mem_gbx["timeseries"],
            omp_nthreads=omp_nthreads,
            name="despike_wf",
        )

        # fmt:off
        workflow.connect([
            (prepare_confounds_wf, despike_wf, [
                ("outputnode.preprocessed_bold", "inputnode.bold_file"),
            ]),
            (despike_wf, denoise_bold_wf, [
                ("outputnode.bold_file", "inputnode.preprocessed_bold"),
            ]),
        ])
        # fmt:on

    else:
        # fmt:off
        workflow.connect([
            (prepare_confounds_wf, denoise_bold_wf, [
                ("outputnode.preprocessed_bold", "inputnode.preprocessed_bold"),
            ]),
        ])
        # fmt:on

    connectivity_wf = init_functional_connectivity_cifti_wf(
        min_coverage=min_coverage,
        output_dir=output_dir,
        mem_gb=mem_gbx["timeseries"],
        name="connectivity_wf",
        omp_nthreads=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, connectivity_wf, [("bold_file", "inputnode.name_source")]),
        (denoise_bold_wf, connectivity_wf, [
            ("outputnode.censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
        (connectivity_wf, outputnode, [
            ("outputnode.atlas_names", "atlas_names"),
            ("outputnode.timeseries", "timeseries"),
            ("outputnode.timeseries_ciftis", "timeseries_ciftis"),
        ]),
    ])
    # fmt:on

    if bandpass_filter:
        alff_wf = init_alff_wf(
            name_source=bold_file,
            output_dir=output_dir,
            TR=TR,
            low_pass=low_pass,
            high_pass=high_pass,
            smoothing=smoothing,
            cifti=True,
            mem_gb=mem_gbx["timeseries"],
            omp_nthreads=omp_nthreads,
            name="alff_wf",
        )

        # fmt:off
        workflow.connect([
            (denoise_bold_wf, alff_wf, [
                ("outputnode.censored_denoised_bold", "inputnode.denoised_bold"),
            ]),
        ])
        # fmt:on

    reho_wf = init_reho_cifti_wf(
        name_source=bold_file,
        output_dir=output_dir,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        name="reho_wf",
    )

    # fmt:off
    workflow.connect([
        (denoise_bold_wf, reho_wf, [
            ("outputnode.censored_denoised_bold", "inputnode.denoised_bold"),
        ]),
    ])
    # fmt:on

    qc_report_wf = init_qc_report_wf(
        output_dir=output_dir,
        TR=TR,
        head_radius=head_radius,
        mem_gb=mem_gbx["timeseries"],
        omp_nthreads=omp_nthreads,
        dcan_qc=dcan_qc,
        cifti=True,
        name="qc_report_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [("bold_file", "inputnode.name_source")]),
        (prepare_confounds_wf, qc_report_wf, [
            ("outputnode.preprocessed_bold", "inputnode.preprocessed_bold"),
            ("outputnode.dummy_scans", "inputnode.dummy_scans"),
            ("outputnode.fmriprep_confounds_file", "inputnode.fmriprep_confounds_file"),
            ("outputnode.temporal_mask", "inputnode.temporal_mask"),
            ("outputnode.filtered_motion", "inputnode.filtered_motion"),
        ]),
        (denoise_bold_wf, qc_report_wf, [
            ("outputnode.uncensored_denoised_bold", "inputnode.uncensored_denoised_bold"),
            ("outputnode.interpolated_filtered_bold", "inputnode.interpolated_filtered_bold"),
            ("outputnode.censored_denoised_bold", "inputnode.censored_denoised_bold"),
        ]),
    ])
    # fmt:on

    postproc_derivatives_wf = init_postproc_derivatives_wf(
        smoothing=smoothing,
        name_source=bold_file,
        bandpass_filter=bandpass_filter,
        params=params,
        cifti=True,
        dcan_qc=dcan_qc,
        output_dir=output_dir,
        low_pass=low_pass,
        high_pass=high_pass,
        fd_thresh=fd_thresh,
        motion_filter_type=motion_filter_type,
        TR=TR,
        name="postproc_derivatives_wf",
    )

    # fmt:off
    workflow.connect([
        (denoise_bold_wf, postproc_derivatives_wf, [
            ("outputnode.interpolated_filtered_bold", "inputnode.interpolated_filtered_bold"),
            ("outputnode.censored_denoised_bold", "inputnode.censored_denoised_bold"),
            ("outputnode.smoothed_denoised_bold", "inputnode.smoothed_denoised_bold"),
        ]),
        (qc_report_wf, postproc_derivatives_wf, [("outputnode.qc_file", "inputnode.qc_file")]),
        (prepare_confounds_wf, postproc_derivatives_wf, [
            ("outputnode.confounds_file", "inputnode.confounds_file"),
            ("outputnode.filtered_motion", "inputnode.filtered_motion"),
            ("outputnode.filtered_motion_metadata", "inputnode.filtered_motion_metadata"),
            ("outputnode.temporal_mask", "inputnode.temporal_mask"),
            ("outputnode.temporal_mask_metadata", "inputnode.temporal_mask_metadata"),
        ]),
        (reho_wf, postproc_derivatives_wf, [("outputnode.reho", "inputnode.reho")]),
        (connectivity_wf, postproc_derivatives_wf, [
            ("outputnode.atlas_names", "inputnode.atlas_names"),
            ("outputnode.coverage_ciftis", "inputnode.coverage_ciftis"),
            ("outputnode.timeseries_ciftis", "inputnode.timeseries_ciftis"),
            ("outputnode.correlation_ciftis", "inputnode.correlation_ciftis"),
            ("outputnode.coverage", "inputnode.coverage"),
            ("outputnode.timeseries", "inputnode.timeseries"),
            ("outputnode.correlations", "inputnode.correlations"),
        ]),
    ])

    if bandpass_filter:
        workflow.connect([
            (alff_wf, postproc_derivatives_wf, [
                ("outputnode.alff", "inputnode.alff"),
                ("outputnode.smoothed_alff", "inputnode.smoothed_alff"),
            ]),
        ])
    # fmt:on

    # executive summary workflow
    if dcan_qc:
        execsummary_functional_plots_wf = init_execsummary_functional_plots_wf(
            preproc_nifti=run_data["nifti_file"],
            t1w_available=t1w_available,
            t2w_available=t2w_available,
            output_dir=output_dir,
            layout=layout,
            name="execsummary_functional_plots_wf",
        )

        # Use inputnode for executive summary instead of downcast_data
        # because T1w is used as name source.
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
    return {
        "derivative": bold_size_gb,
        "resampled": bold_size_gb * 4,
        "timeseries": bold_size_gb * (max(bold_tlen / 100, 1.0) + 4),
    }
