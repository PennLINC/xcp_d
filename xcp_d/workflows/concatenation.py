"""Workflows for concatenating postprocessed data."""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.concatenation import (
    CleanNameSource,
    ConcatenateInputs,
    FilterOutFailedRuns,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import _select_first
from xcp_d.workflows.plotting import init_qc_report_wf


@fill_doc
def init_concatenate_data_wf(
    output_dir,
    motion_filter_type,
    band_stop_max,
    band_stop_min,
    motion_filter_order,
    fd_thresh,
    mem_gb,
    omp_nthreads,
    TR,
    smooth,
    cifti,
    dcan_qc,
    name="concatenate_data_wf",
):
    """Concatenate postprocessed data.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.concatenation import init_concatenate_data_wf

            wf = init_concatenate_data_wf(
                output_dir=".",
                motion_filter_type=None,
                motion_filter_order=4,
                band_stop_min=12,
                band_stop_max=20,
                fd_thresh=0.2,
                mem_gb=0.1,
                omp_nthreads=1,
                TR=2,
                smooth=False,
                cifti=False,
                dcan_qc=True,
                name="concatenate_data_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    %(motion_filter_type)s
    %(cifti)s
    %(name)s
        Default is "concatenate_data_wf".

    Inputs
    ------
    name_source : :obj:`list` of :obj:`str`
        The preprocessed BOLD files that were post-processed with XCP-D.
        These are used as the bases for concatenated output filenames.
    preprocessed_bold : :obj:`list` of :obj:`str`
        The preprocessed BOLD files, after dummy volume removal.
    confounds_file : :obj:`list` of :obj:`str`
        TSV files with selected confounds for individual BOLD runs.
    filtered_motion : :obj:`list` of :obj:`str`
        TSV files with filtered motion parameters, used for FD calculation.
    temporal_mask : :obj:`list` of :obj:`str`
        TSV files with high-motion outliers indexed.
    uncensored_denoised_bold : :obj:`list` of :obj:`str`
        Denoised BOLD data.
    filtered_denoised_bold : :obj:`list` of :obj:`str`
        Denoised BOLD data.
    bold_mask : :obj:`list` of :obj:`str` or :obj:`~nipype.interfaces.base.Undefined`
        Brain mask files for each of the BOLD runs.
        This will be a list of paths for NIFTI inputs, or a list of Undefineds for CIFTI ones.
    t1w_mask : :obj:`str`
    boldref
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "preprocessed_bold",
                "confounds_file",
                "filtered_motion",
                "temporal_mask",
                "uncensored_denoised_bold",
                "filtered_denoised_bold",
                "smoothed_denoised_bold",
                "bold_mask",  # only for niftis, from postproc workflows
                "boldref",  # only for niftis, from postproc workflows
                "t1w_to_native_xform",  # only for niftis, from postproc workflows
                "t1w_mask",  # only for niftis, from data collection
                "template_to_t1w_xform",  # only for niftis, from data collection
            ],
        ),
        name="inputnode",
    )

    clean_name_source = pe.Node(
        CleanNameSource(),
        name="clean_name_source",
    )

    # fmt:off
    workflow.connect([(inputnode, clean_name_source, [("name_source", "name_source")])])
    # fmt:on

    filter_out_failed_runs = pe.Node(
        FilterOutFailedRuns(),
        name="filter_out_failed_runs",
    )

    # fmt:off
    workflow.connect([
        (inputnode, filter_out_failed_runs, [
            ("preprocessed_bold", "preprocessed_bold"),
            ("confounds_file", "confounds_file"),
            ("filtered_motion", "filtered_motion"),
            ("temporal_mask", "temporal_mask"),
            ("uncensored_denoised_bold", "uncensored_denoised_bold"),
            ("filtered_denoised_bold", "filtered_denoised_bold"),
            ("smoothed_denoised_bold", "smoothed_denoised_bold"),
            ("bold_mask", "bold_mask"),
            ("boldref", "boldref"),
            ("t1w_to_native_xform", "t1w_to_native_xform"),
        ])
    ])
    # fmt:on

    concatenate_inputs = pe.Node(
        ConcatenateInputs(cifti=cifti),
        name="concatenate_inputs",
    )

    # fmt:off
    workflow.connect([
        (filter_out_failed_runs, concatenate_inputs, [
            ("preprocessed_bold", "preprocessed_bold"),
            ("confounds_file", "confounds_file"),
            ("filtered_motion", "filtered_motion"),
            ("temporal_mask", "temporal_mask"),
            ("uncensored_denoised_bold", "uncensored_denoised_bold"),
            ("filtered_denoised_bold", "filtered_denoised_bold"),
            ("smoothed_denoised_bold", "smoothed_denoised_bold"),
        ]),
    ])
    # fmt:on

    # Now, I need to take the concatenation node's outputs and run the QC report workflow on
    # each of them.
    qc_report_wf = init_qc_report_wf(
        output_dir=output_dir,
        TR=TR,
        motion_filter_type=motion_filter_type,
        band_stop_max=band_stop_max,
        band_stop_min=band_stop_min,
        motion_filter_order=motion_filter_order,
        fd_thresh=fd_thresh,
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
        cifti=cifti,
        dcan_qc=dcan_qc,
        name="concat_qc_report_wf",
    )

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [
            ("template_to_t1w_xform", "inputnode.template_to_t1w"),
            ("t1w_mask", "inputnode.t1w_mask"),
        ]),
        (clean_name_source, qc_report_wf, [("name_source", "inputnode.name_source")]),
        (filter_out_failed_runs, qc_report_wf, [
            # nifti-only inputs
            (("bold_mask", _select_first), "inputnode.bold_mask"),
            (("boldref", _select_first), "inputnode.boldref"),
            (("t1w_to_native_xform", _select_first), "inputnode.t1w_to_native"),
        ]),
        (concatenate_inputs, qc_report_wf, [
            ("preprocessed_bold", "inputnode.preprocessed_bold"),
            ("filtered_denoised_bold", "inputnode.filtered_denoised_bold"),
            ("uncensored_denoised_bold", "inputnode.uncensored_denoised_bold"),
            ("filtered_motion", "inputnode.filtered_motion"),
            ("temporal_mask", "inputnode.tmask"),
        ]),
    ])
    # fmt:on

    ds_confounds_file = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["space", "cohort", "den", "res"],
            datatype="func",
            suffix="design",
            extension=".tsv",
        ),
        name="ds_confounds_file",
        run_without_submitting=True,
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([
        (clean_name_source, ds_confounds_file, [("name_source", "source_file")]),
        (concatenate_inputs, ds_confounds_file, [("confounds_file", "in_file")]),
    ])
    # fmt:on

    ds_filtered_motion = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["atlas", "den", "res", "space", "cohort", "desc"],
            desc="filtered" if motion_filter_type else None,
            suffix="motion",
            extension=".tsv",
        ),
        name="ds_filtered_motion",
        run_without_submitting=True,
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([
        (clean_name_source, ds_filtered_motion, [("name_source", "source_file")]),
        (concatenate_inputs, ds_filtered_motion, [("filtered_motion", "in_file")]),
    ])
    # fmt:on

    ds_temporal_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["atlas", "den", "res", "space", "cohort", "desc"],
            suffix="outliers",
            extension=".tsv",
        ),
        name="ds_temporal_mask",
        run_without_submitting=True,
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([
        (clean_name_source, ds_temporal_mask, [("name_source", "source_file")]),
        (concatenate_inputs, ds_temporal_mask, [("temporal_mask", "in_file")]),
    ])
    # fmt:on

    if cifti:
        ds_filtered_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                desc="denoised",
                den="91k",
                extension=".dtseries.nii",
            ),
            name="ds_filtered_denoised_bold",
            run_without_submitting=True,
            mem_gb=2,
        )

        if smooth:
            ds_smoothed_denoised_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["den"],
                    desc="smoothDenoised",
                    den="91k",
                    extension=".dtseries.nii",
                ),
                name="ds_smoothed_denoised_bold",
                run_without_submitting=True,
                mem_gb=2,
            )
    else:
        ds_filtered_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="denoised",
                extension=".nii.gz",
                compression=True,
            ),
            name="ds_filtered_denoised_bold",
            run_without_submitting=True,
            mem_gb=2,
        )
        if smooth:
            ds_smoothed_denoised_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    desc="smoothDenoised",
                    extension=".nii.gz",
                    compression=True,
                ),
                name="ds_smoothed_denoised_bold",
                run_without_submitting=True,
                mem_gb=2,
            )

    # fmt:off
    workflow.connect([
        (clean_name_source, ds_filtered_denoised_bold, [("name_source", "source_file")]),
        (concatenate_inputs, ds_filtered_denoised_bold, [("filtered_denoised_bold", "in_file")]),
    ])
    # fmt:on

    if smooth:
        # fmt:off
        workflow.connect([
            (clean_name_source, ds_smoothed_denoised_bold, [("name_source", "source_file")]),
            (concatenate_inputs, ds_smoothed_denoised_bold, [
                ("smoothed_denoised_bold", "in_file"),
            ]),
        ])
        # fmt:on

    return workflow
