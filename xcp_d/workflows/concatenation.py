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
from xcp_d.utils.utils import _select_first, estimate_brain_radius
from xcp_d.workflows.plotting import init_qc_report_wf


@fill_doc
def init_concatenate_data_wf(
    output_dir,
    motion_filter_type,
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
    fd_thresh
    mem_gb
    omp_nthreads
    TR
    smooth
    %(cifti)s
    dcan_qc
    %(name)s
        Default is "concatenate_data_wf".

    Inputs
    ------
    name_source : :obj:`list` of :obj:`str`
        The preprocessed BOLD files that were post-processed with XCP-D.
        These are used as the bases for concatenated output filenames.
    preprocessed_bold : :obj:`list` of :obj:`str`
        The preprocessed BOLD files, after dummy volume removal.
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
    head_radius
    atlas_names
    timeseries
    timeseries_ciftis
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = """\
    Postprocessing derivatives from multi-run tasks were then concatenated across runs.
    """

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "preprocessed_bold",
                "fmriprep_confounds_file",
                "filtered_motion",
                "temporal_mask",
                "uncensored_denoised_bold",
                "filtered_denoised_bold",
                "smoothed_denoised_bold",
                "head_radius",
                "bold_mask",  # only for niftis, from postproc workflows
                "boldref",  # only for niftis, from postproc workflows
                "t1w_to_native_xform",  # only for niftis, from postproc workflows
                "t1w_mask",  # only for niftis, from data collection
                "template_to_t1w_xform",  # only for niftis, from data collection
                "atlas_names",  # this will be exactly the same across runs
                "timeseries",
                "timeseries_ciftis",  # only for ciftis, from postproc workflows
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

    get_head_radius = pe.Node(
        niu.Function(
            function=estimate_brain_radius,
            input_names=["mask_file", "head_radius"],
            output_names=["head_radius"],
        ),
        name="get_head_radius",
    )

    # fmt:off
    workflow.connect([
        (inputnode, get_head_radius, [
            ("t1w_mask", "mask_file"),
            ("head_radius", "head_radius"),
        ]),
    ])
    # fmt:on

    filter_out_failed_runs = pe.Node(
        FilterOutFailedRuns(),
        name="filter_out_failed_runs",
    )

    # fmt:off
    workflow.connect([
        (inputnode, filter_out_failed_runs, [
            ("preprocessed_bold", "preprocessed_bold"),
            ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ("filtered_motion", "filtered_motion"),
            ("temporal_mask", "temporal_mask"),
            ("uncensored_denoised_bold", "uncensored_denoised_bold"),
            ("filtered_denoised_bold", "filtered_denoised_bold"),
            ("smoothed_denoised_bold", "smoothed_denoised_bold"),
            ("bold_mask", "bold_mask"),
            ("boldref", "boldref"),
            ("t1w_to_native_xform", "t1w_to_native_xform"),
            ("atlas_names", "atlas_names"),
            ("timeseries", "timeseries"),
            ("timeseries_ciftis", "timeseries_ciftis"),
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
            ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ("filtered_motion", "filtered_motion"),
            ("temporal_mask", "temporal_mask"),
            ("uncensored_denoised_bold", "uncensored_denoised_bold"),
            ("filtered_denoised_bold", "filtered_denoised_bold"),
            ("smoothed_denoised_bold", "smoothed_denoised_bold"),
            ("timeseries", "timeseries"),
            ("timeseries_ciftis", "timeseries_ciftis"),
        ]),
    ])
    # fmt:on

    # Now, run the QC report workflow on the concatenated BOLD file.
    qc_report_wf = init_qc_report_wf(
        output_dir=output_dir,
        TR=TR,
        motion_filter_type=motion_filter_type,
        fd_thresh=fd_thresh,
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
        cifti=cifti,
        dcan_qc=dcan_qc,
        name="concat_qc_report_wf",
    )
    qc_report_wf.inputs.inputnode.dummy_scans = 0

    # fmt:off
    workflow.connect([
        (inputnode, qc_report_wf, [
            ("template_to_t1w_xform", "inputnode.template_to_t1w"),
            ("t1w_mask", "inputnode.t1w_mask"),
        ]),
        (clean_name_source, qc_report_wf, [("name_source", "inputnode.name_source")]),
        (get_head_radius, qc_report_wf, [("head_radius", "inputnode.head_radius")]),
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
            ("fmriprep_confounds_file", "inputnode.fmriprep_confounds_file"),
            ("filtered_motion", "inputnode.filtered_motion"),
            ("temporal_mask", "inputnode.tmask"),
        ]),
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

    ds_timeseries = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["desc"],
            suffix="timeseries",
            extension=".tsv",
        ),
        name="ds_timeseries",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )

    # fmt:off
    workflow.connect([
        (clean_name_source, ds_timeseries, [("name_source", "source_file")]),
        (filter_out_failed_runs, ds_timeseries, [(("atlas_names", _select_first), "atlas")]),
        (concatenate_inputs, ds_timeseries, [("timeseries", "in_file")]),
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

        ds_timeseries_cifti_files = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc", "den"],
                den="91k",
                suffix="timeseries",
                extension=".ptseries.nii",
            ),
            name="ds_timeseries_cifti_files",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        # fmt:off
        workflow.connect([
            (clean_name_source, ds_timeseries_cifti_files, [("name_source", "source_file")]),
            (filter_out_failed_runs, ds_timeseries_cifti_files, [
                (("atlas_names", _select_first), "atlas"),
            ]),
            (concatenate_inputs, ds_timeseries_cifti_files, [("timeseries_ciftis", "in_file")]),
        ])
        # fmt:on

        if smooth:
            ds_smoothed_denoised_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["den"],
                    desc="denoisedSmoothed",
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
                    desc="denoisedSmoothed",
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
