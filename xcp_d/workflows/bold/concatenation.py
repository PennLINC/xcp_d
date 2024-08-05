"""Workflows for concatenating postprocessed data."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.concatenation import (
    CleanNameSource,
    ConcatenateInputs,
    FilterOutFailedRuns,
)
from xcp_d.interfaces.connectivity import TSVConnect
from xcp_d.utils.bids import _make_xcpd_uri, _make_xcpd_uri_lol
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import _make_dictionary, _select_first
from xcp_d.workflows.bold.plotting import init_qc_report_wf


@fill_doc
def init_concatenate_data_wf(TR, head_radius, name="concatenate_data_wf"):
    """Concatenate postprocessed data across runs and directions.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.concatenation import init_concatenate_data_wf

            with mock_config():
                wf = init_concatenate_data_wf(
                    TR=2,
                    head_radius=50,
                    name="concatenate_data_wf",
                )

    Parameters
    ----------
    %(TR)s
    %(head_radius)s
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
    %(denoised_interpolated_bold)s
        One list entry for each run.
    %(censored_denoised_bold)s
        One list entry for each run.
    bold_mask : :obj:`list` of :obj:`str` or :obj:`~nipype.interfaces.base.Undefined`
        Brain mask files for each of the BOLD runs.
        This will be a list of paths for NIFTI inputs, or a list of Undefineds for CIFTI ones.
    anat_brainmask : :obj:`str`
        Anatomically-derived brain mask in the same standard space as the BOLD mask.
    %(template_to_anat_xfm)s
    %(boldref)s
    %(timeseries)s
        This will be a list of lists, with one sublist for each run.
    %(timeseries_ciftis)s
        This will be a list of lists, with one sublist for each run.
    """
    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir
    motion_filter_type = config.workflow.motion_filter_type
    smoothing = config.workflow.smoothing
    file_format = config.workflow.file_format
    fd_thresh = config.workflow.fd_thresh
    atlases = config.execution.atlases

    workflow.__desc__ = """
Postprocessing derivatives from multi-run tasks were then concatenated across runs and directions.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "preprocessed_bold",
                "fmriprep_confounds_file",
                "filtered_motion",
                "temporal_mask",
                "denoised_bold",
                "denoised_interpolated_bold",
                "censored_denoised_bold",
                "smoothed_denoised_bold",
                "bold_mask",  # only for niftis, from postproc workflows
                "boldref",  # only for niftis, from postproc workflows
                "anat_native",  # only for niftis, from data collection
                "anat_brainmask",  # only for niftis, from data collection
                "template_to_anat_xfm",  # only for niftis, from data collection
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
    workflow.connect([(inputnode, clean_name_source, [("name_source", "name_source")])])

    filter_runs = pe.Node(
        FilterOutFailedRuns(),
        name="filter_runs",
    )

    workflow.connect([
        (inputnode, filter_runs, [
            ("preprocessed_bold", "preprocessed_bold"),
            ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ("filtered_motion", "filtered_motion"),
            ("temporal_mask", "temporal_mask"),
            ("denoised_bold", "denoised_bold"),
            ("denoised_interpolated_bold", "denoised_interpolated_bold"),
            ("censored_denoised_bold", "censored_denoised_bold"),
            ("smoothed_denoised_bold", "smoothed_denoised_bold"),
            ("bold_mask", "bold_mask"),
            ("boldref", "boldref"),
            ("timeseries", "timeseries"),
            ("timeseries_ciftis", "timeseries_ciftis"),
        ])
    ])  # fmt:skip

    concatenate_inputs = pe.Node(
        ConcatenateInputs(),
        name="concatenate_inputs",
    )

    workflow.connect([
        (filter_runs, concatenate_inputs, [
            ("preprocessed_bold", "preprocessed_bold"),
            ("fmriprep_confounds_file", "fmriprep_confounds_file"),
            ("filtered_motion", "filtered_motion"),
            ("temporal_mask", "temporal_mask"),
            ("denoised_bold", "denoised_bold"),
            ("denoised_interpolated_bold", "denoised_interpolated_bold"),
            ("censored_denoised_bold", "censored_denoised_bold"),
            ("smoothed_denoised_bold", "smoothed_denoised_bold"),
            ("timeseries", "timeseries"),
            ("timeseries_ciftis", "timeseries_ciftis"),
        ]),
    ])  # fmt:skip

    # Now, run the QC report workflow on the concatenated BOLD file.
    qc_report_wf = init_qc_report_wf(
        TR=TR,
        head_radius=head_radius,
        name="concat_qc_report_wf",
    )
    qc_report_wf.inputs.inputnode.dummy_scans = 0

    workflow.connect([
        (inputnode, qc_report_wf, [
            ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
            ("anat_native", "inputnode.anat"),
            ("anat_brainmask", "inputnode.anat_brainmask"),
        ]),
        (clean_name_source, qc_report_wf, [("name_source", "inputnode.name_source")]),
        (filter_runs, qc_report_wf, [
            # nifti-only inputs
            (("bold_mask", _select_first), "inputnode.bold_mask"),
            (("boldref", _select_first), "inputnode.boldref"),
        ]),
        (concatenate_inputs, qc_report_wf, [
            ("preprocessed_bold", "inputnode.preprocessed_bold"),
            ("denoised_interpolated_bold", "inputnode.denoised_interpolated_bold"),
            ("censored_denoised_bold", "inputnode.censored_denoised_bold"),
            ("fmriprep_confounds_file", "inputnode.fmriprep_confounds_file"),
            ("filtered_motion", "inputnode.filtered_motion"),
            ("temporal_mask", "inputnode.temporal_mask"),
            ("run_index", "inputnode.run_index"),
        ]),
    ])  # fmt:skip

    ds_filtered_motion = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["segmentation", "den", "res", "space", "cohort", "desc"],
            desc="filtered" if motion_filter_type else None,
            suffix="motion",
            extension=".tsv",
        ),
        name="ds_filtered_motion",
        run_without_submitting=True,
        mem_gb=1,
    )
    workflow.connect([
        (clean_name_source, ds_filtered_motion, [("name_source", "source_file")]),
        (concatenate_inputs, ds_filtered_motion, [("filtered_motion", "in_file")]),
        (filter_runs, ds_filtered_motion, [
            (("filtered_motion", _make_xcpd_uri, output_dir), "Sources"),
        ]),
    ])  # fmt:skip

    if fd_thresh > 0:
        ds_temporal_mask = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["segmentation", "den", "res", "space", "cohort", "desc"],
                suffix="outliers",
                extension=".tsv",
            ),
            name="ds_temporal_mask",
            run_without_submitting=True,
            mem_gb=1,
        )

        workflow.connect([
            (clean_name_source, ds_temporal_mask, [("name_source", "source_file")]),
            (concatenate_inputs, ds_temporal_mask, [("temporal_mask", "in_file")]),
            (filter_runs, ds_temporal_mask, [
                (("temporal_mask", _make_xcpd_uri, output_dir), "Sources"),
            ]),
        ])  # fmt:skip

    if file_format == "cifti":
        ds_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["den"],
                desc="denoised",
                den="91k",
                extension=".dtseries.nii",
            ),
            name="ds_denoised_bold",
            run_without_submitting=True,
            mem_gb=2,
        )

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

    else:
        ds_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="denoised",
                extension=".nii.gz",
                compression=True,
            ),
            name="ds_denoised_bold",
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

    workflow.connect([
        (clean_name_source, ds_denoised_bold, [("name_source", "source_file")]),
        (concatenate_inputs, ds_denoised_bold, [("denoised_bold", "in_file")]),
        (filter_runs, ds_denoised_bold, [
            (("denoised_bold", _make_xcpd_uri, output_dir), "Sources"),
        ]),
    ])  # fmt:skip

    if smoothing:
        workflow.connect([
            (clean_name_source, ds_smoothed_denoised_bold, [("name_source", "source_file")]),
            (concatenate_inputs, ds_smoothed_denoised_bold, [
                ("smoothed_denoised_bold", "in_file"),
            ]),
            (filter_runs, ds_smoothed_denoised_bold, [
                (("smoothed_denoised_bold", _make_xcpd_uri, output_dir), "Sources"),
            ]),
        ])  # fmt:skip

    # Functional connectivity outputs
    if atlases:
        make_timeseries_dict = pe.MapNode(
            niu.Function(
                function=_make_dictionary,
                input_names=["Sources"],
                output_names=["metadata"],
            ),
            run_without_submitting=True,
            mem_gb=1,
            name="make_timeseries_dict",
            iterfield=["Sources"],
        )
        workflow.connect([
            (filter_runs, make_timeseries_dict, [
                (("timeseries", _make_xcpd_uri_lol, output_dir), "Sources"),
            ]),
        ])  # fmt:skip

        ds_timeseries = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["desc", "den", "res"],
                statistic="mean",
                suffix="timeseries",
                extension=".tsv",
                # Metadata
                SamplingFrequency="TR",
            ),
            name="ds_timeseries",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["segmentation", "in_file", "meta_dict"],
        )
        ds_timeseries.inputs.segmentation = atlases

        workflow.connect([
            (clean_name_source, ds_timeseries, [("name_source", "source_file")]),
            (concatenate_inputs, ds_timeseries, [("timeseries", "in_file")]),
            (make_timeseries_dict, ds_timeseries, [("metadata", "meta_dict")]),
        ])  # fmt:skip

        correlate_timeseries = pe.MapNode(
            TSVConnect(),
            run_without_submitting=True,
            mem_gb=1,
            name="correlate_timeseries",
            iterfield=["timeseries"],
        )
        workflow.connect([
            (concatenate_inputs, correlate_timeseries, [
                ("timeseries", "timeseries"),
                ("temporal_mask", "temporal_mask"),
            ]),
        ])  # fmt:skip

        make_correlations_dict = pe.MapNode(
            niu.Function(
                function=_make_dictionary,
                input_names=["Sources"],
                output_names=["metadata"],
            ),
            run_without_submitting=True,
            mem_gb=1,
            name="make_correlations_dict",
            iterfield=["Sources"],
        )
        workflow.connect([
            (ds_timeseries, make_correlations_dict, [
                (("out_file", _make_xcpd_uri, output_dir), "Sources"),
            ]),
        ])  # fmt:skip

        ds_correlations = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["desc"],
                statistic="pearsoncorrelation",
                suffix="relmat",
                extension=".tsv",
            ),
            name="ds_correlations",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["segmentation", "in_file", "meta_dict"],
        )
        ds_correlations.inputs.segmentation = atlases

        workflow.connect([
            (clean_name_source, ds_correlations, [("name_source", "source_file")]),
            (correlate_timeseries, ds_correlations, [("correlations", "in_file")]),
            (make_correlations_dict, ds_correlations, [("metadata", "meta_dict")]),
        ])  # fmt:skip

        if file_format == "cifti":
            make_timeseries_ciftis_dict = pe.MapNode(
                niu.Function(
                    function=_make_dictionary,
                    input_names=["Sources"],
                    output_names=["metadata"],
                ),
                run_without_submitting=True,
                mem_gb=1,
                name="make_timeseries_ciftis_dict",
                iterfield=["Sources"],
            )
            workflow.connect([
                (filter_runs, make_timeseries_ciftis_dict, [
                    (("timeseries_ciftis", _make_xcpd_uri_lol, output_dir), "Sources"),
                ]),
            ])  # fmt:skip

            ds_timeseries_cifti_files = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                    check_hdr=False,
                    dismiss_entities=["desc", "den"],
                    den="91k",
                    statistic="mean",
                    suffix="timeseries",
                    extension=".ptseries.nii",
                ),
                name="ds_timeseries_cifti_files",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["segmentation", "in_file", "meta_dict"],
            )
            ds_timeseries_cifti_files.inputs.segmentation = atlases

            workflow.connect([
                (clean_name_source, ds_timeseries_cifti_files, [("name_source", "source_file")]),
                (concatenate_inputs, ds_timeseries_cifti_files, [
                    ("timeseries_ciftis", "in_file"),
                ]),
                (make_timeseries_ciftis_dict, ds_timeseries_cifti_files, [
                    ("metadata", "meta_dict"),
                ]),
            ])  # fmt:skip

    return workflow
