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
    mem_gb,
    omp_nthreads,
    TR,
    head_radius,
    smoothing,
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
                mem_gb=0.1,
                omp_nthreads=1,
                TR=2,
                head_radius=50,
                smoothing=None,
                cifti=False,
                dcan_qc=True,
                name="concatenate_data_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    %(motion_filter_type)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(TR)s
    %(head_radius)s
    %(smoothing)s
    %(cifti)s
    %(dcan_qc)s
    %(name)s
        Default is "concatenate_data_wf".

    Inputs
    ------
    %(name_source)s
        One list entry for each run.
        These are used as the bases for concatenated output filenames.
    preprocessed_bold : :obj:`list` of :obj:`str`
        The preprocessed BOLD files, after dummy volume removal.
    %(filtered_motion)s
        One list entry for each run.
    %(temporal_mask)s
        One list entry for each run.
    %(uncensored_denoised_bold)s
        One list entry for each run.
    %(interpolated_filtered_bold)s
        One list entry for each run.
    %(censored_denoised_bold)s
        One list entry for each run.
    bold_mask : :obj:`list` of :obj:`str` or :obj:`~nipype.interfaces.base.Undefined`
        Brain mask files for each of the BOLD runs.
        This will be a list of paths for NIFTI inputs, or a list of Undefineds for CIFTI ones.
    anat_brainmask : :obj:`str`
    %(template_to_anat_xfm)s
    %(boldref)s
    %(atlas_names)s
        This will be a list of lists, with one sublist for each run.
    %(timeseries)s
        This will be a list of lists, with one sublist for each run.
    %(timeseries_ciftis)s
        This will be a list of lists, with one sublist for each run.
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = """
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
                "interpolated_filtered_bold",
                "censored_denoised_bold",
                "smoothed_denoised_bold",
                "bold_mask",  # only for niftis, from postproc workflows
                "boldref",  # only for niftis, from postproc workflows
                "anat_to_native_xfm",  # only for niftis, from postproc workflows
                "anat_brainmask",  # only for niftis, from data collection
                "template_to_anat_xfm",  # only for niftis, from data collection
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
            ("interpolated_filtered_bold", "interpolated_filtered_bold"),
            ("censored_denoised_bold", "censored_denoised_bold"),
            ("smoothed_denoised_bold", "smoothed_denoised_bold"),
            ("bold_mask", "bold_mask"),
            ("boldref", "boldref"),
            ("anat_to_native_xfm", "anat_to_native_xfm"),
            ("atlas_names", "atlas_names"),
            ("timeseries", "timeseries"),
            ("timeseries_ciftis", "timeseries_ciftis"),
        ])
    ])
    # fmt:on

    concatenate_inputs = pe.Node(
        ConcatenateInputs(),
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
            ("interpolated_filtered_bold", "interpolated_filtered_bold"),
            ("censored_denoised_bold", "censored_denoised_bold"),
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
        head_radius=head_radius,
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
            ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
            ("anat_brainmask", "inputnode.anat_brainmask"),
        ]),
        (clean_name_source, qc_report_wf, [("name_source", "inputnode.name_source")]),
        (filter_out_failed_runs, qc_report_wf, [
            # nifti-only inputs
            (("bold_mask", _select_first), "inputnode.bold_mask"),
            (("boldref", _select_first), "inputnode.boldref"),
            (("anat_to_native_xfm", _select_first), "inputnode.anat_to_native_xfm"),
        ]),
        (concatenate_inputs, qc_report_wf, [
            ("preprocessed_bold", "inputnode.preprocessed_bold"),
            ("uncensored_denoised_bold", "inputnode.uncensored_denoised_bold"),
            ("interpolated_filtered_bold", "inputnode.interpolated_filtered_bold"),
            ("censored_denoised_bold", "inputnode.censored_denoised_bold"),
            ("fmriprep_confounds_file", "inputnode.fmriprep_confounds_file"),
            ("filtered_motion", "inputnode.filtered_motion"),
            ("temporal_mask", "inputnode.temporal_mask"),
            ("run_index", "inputnode.run_index"),
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
        ds_censored_filtered_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                desc="denoised",
                den="91k",
                extension=".dtseries.nii",
            ),
            name="ds_censored_filtered_bold",
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

        if smoothing:
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

        if dcan_qc:
            ds_interpolated_filtered_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["den"],
                    desc="interpolated",
                    den="91k",
                    extension=".dtseries.nii",
                ),
                name="ds_interpolated_filtered_bold",
                run_without_submitting=True,
                mem_gb=2,
            )

    else:
        ds_censored_filtered_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="denoised",
                extension=".nii.gz",
                compression=True,
            ),
            name="ds_censored_filtered_bold",
            run_without_submitting=True,
            mem_gb=2,
        )
        if smoothing:
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

        if dcan_qc:
            ds_interpolated_filtered_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    desc="interpolated",
                    extension=".nii.gz",
                    compression=True,
                ),
                name="ds_interpolated_filtered_bold",
                run_without_submitting=True,
                mem_gb=2,
            )

    # fmt:off
    workflow.connect([
        (clean_name_source, ds_censored_filtered_bold, [("name_source", "source_file")]),
        (concatenate_inputs, ds_censored_filtered_bold, [("censored_denoised_bold", "in_file")]),
    ])
    # fmt:on

    if smoothing:
        # fmt:off
        workflow.connect([
            (clean_name_source, ds_smoothed_denoised_bold, [("name_source", "source_file")]),
            (concatenate_inputs, ds_smoothed_denoised_bold, [
                ("smoothed_denoised_bold", "in_file"),
            ]),
        ])
        # fmt:on

    if dcan_qc:
        # fmt:off
        workflow.connect([
            (clean_name_source, ds_interpolated_filtered_bold, [("name_source", "source_file")]),
            (concatenate_inputs, ds_interpolated_filtered_bold, [
                ("interpolated_filtered_bold", "in_file"),
            ]),
        ])
        # fmt:on

    return workflow
