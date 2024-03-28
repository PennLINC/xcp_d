# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for collecting and saving xcp_d outputs."""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.bids import BIDSURI, DerivativesDataSink, GenerateMetadata
from xcp_d.interfaces.utils import FilterUndefined
from xcp_d.utils.bids import get_entity
from xcp_d.utils.doc import fill_doc


@fill_doc
def init_copy_inputs_to_outputs_wf(name="copy_inputs_to_outputs_wf"):
    """Copy files from the preprocessing derivatives to the output folder, with no modifications.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.outputs import init_copy_inputs_to_outputs_wf

            with mock_config():
                wf = init_copy_inputs_to_outputs_wf()

    Parameters
    ----------
    %(name)s
        Default is "copy_inputs_to_outputs_wf".

    Inputs
    ------
    lh_pial_surf
    rh_pial_surf
    lh_wm_surf
    rh_wm_surf
    sulcal_depth
    sulcal_curv
    cortical_thickness
    cortical_thickness_corr
    myelin
    myelin_smoothed
    """
    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
                "sulcal_depth",
                "sulcal_curv",
                "cortical_thickness",
                "cortical_thickness_corr",
                "myelin",
                "myelin_smoothed",
            ],
        ),
        name="inputnode",
    )

    # Place the surfaces in a single node.
    collect_files = pe.Node(
        niu.Merge(10),
        name="collect_files",
    )
    workflow.connect([
        (inputnode, collect_files, [
            # fsLR-space surface mesh files
            ("lh_pial_surf", "in1"),
            ("rh_pial_surf", "in2"),
            ("lh_wm_surf", "in3"),
            ("rh_wm_surf", "in4"),
            # fsLR-space surface shape files
            ("sulcal_depth", "in5"),
            ("sulcal_curv", "in6"),
            ("cortical_thickness", "in7"),
            ("cortical_thickness_corr", "in8"),
            ("myelin", "in9"),
            ("myelin_smoothed", "in10"),
        ]),
    ])  # fmt:skip

    filter_out_undefined = pe.Node(
        FilterUndefined(),
        name="filter_out_undefined",
    )
    workflow.connect([(collect_files, filter_out_undefined, [("out", "inlist")])])

    ds_copied_outputs = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
        ),
        name="ds_copied_outputs",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["in_file", "source_file"],
    )
    workflow.connect([
        (filter_out_undefined, ds_copied_outputs, [
            ("outlist", "in_file"),
            ("outlist", "source_file"),
        ]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_postproc_derivatives_wf(
    name_source,
    source_metadata,
    exact_scans,
    custom_confounds_file,
    name="postproc_derivatives_wf",
):
    """Write out the xcp_d derivatives in BIDS format.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.outputs import init_postproc_derivatives_wf

            with mock_config():
                wf = init_postproc_derivatives_wf(
                    name_source="/path/to/file.nii.gz",
                    source_metadata={},
                    exact_scans=[],
                    custom_confounds_file=None,
                    name="postproc_derivatives_wf",
                )

    Parameters
    ----------
    name_source : :obj:`str`
        bold or cifti files
    source_metadata : :obj:`dict`
    %(exact_scans)s
    custom_confounds_file
        Only used for Sources metadata.
    %(name)s
        Default is "connectivity_wf".

    Inputs
    ------
    atlas_files
    %(timeseries)s
    %(correlations)s
    %(coverage)s
    %(timeseries_ciftis)s
    %(correlation_ciftis)s
    %(coverage_ciftis)s
    qc_file
        LINC-style quality control file
    %(denoised_interpolated_bold)s
    %(censored_denoised_bold)s
    %(smoothed_denoised_bold)s
    alff
        alff nifti
    parcellated_alff
    smoothed_alff
        smoothed alff
    reho
    parcellated_reho
    confounds_file
    confounds_metadata
    %(filtered_motion)s
    motion_metadata
    %(temporal_mask)s
    temporal_mask_metadata
    %(dummy_scans)s
    """
    workflow = Workflow(name=name)

    bandpass_filter = config.workflow.bandpass_filter
    low_pass = config.workflow.low_pass
    high_pass = config.workflow.high_pass
    bpf_order = config.workflow.bpf_order
    fd_thresh = config.workflow.fd_thresh
    motion_filter_type = config.workflow.motion_filter_type
    smoothing = config.workflow.smoothing
    params = config.workflow.params
    atlases = config.execution.atlases
    cifti = config.workflow.cifti
    dcan_qc = config.workflow.dcan_qc
    output_dir = config.execution.xcp_d_dir

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # preprocessing files to use as sources
                "fmriprep_confounds_file",
                # postprocessed outputs
                "atlas_files",  # for Sources
                "confounds_file",
                "confounds_metadata",
                "coverage",
                "timeseries",
                "correlations",
                "correlations_exact",
                "qc_file",
                "censored_denoised_bold",
                "smoothed_denoised_bold",
                "denoised_interpolated_bold",
                "alff",
                "parcellated_alff",
                "smoothed_alff",
                "reho",
                "parcellated_reho",
                "filtered_motion",
                "motion_metadata",
                "temporal_mask",
                "temporal_mask_metadata",
                "dummy_scans",
                # cifti-only inputs
                "coverage_ciftis",
                "timeseries_ciftis",
                "correlation_ciftis",
                "correlation_ciftis_exact",
            ],
        ),
        name="inputnode",
    )

    # Outputs that may be used by the concatenation workflow, in which case we want the actual
    # output filenames for the Sources metadata field.
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "filtered_motion",
                "temporal_mask",
                "denoised_interpolated_bold",
                "censored_denoised_bold",
                "smoothed_denoised_bold",
                "timeseries",
                "timeseries_ciftis",
            ],
        ),
        name="outputnode",
    )

    # Create dictionary of basic information
    cleaned_data_dictionary = {
        "NuisanceParameters": params,
        **source_metadata,
    }
    software_filters = None
    if bandpass_filter:
        software_filters = {}
        if low_pass > 0 and high_pass > 0:
            software_filters["Bandpass filter"] = {
                "Low-pass cutoff (Hz)": low_pass,
                "High-pass cutoff (Hz)": high_pass,
                "Filter order": bpf_order,
            }
        elif high_pass > 0:
            software_filters["High-pass filter"] = {
                "cutoff (Hz)": high_pass,
                "Filter order": bpf_order,
            }
        elif low_pass > 0:
            software_filters["Low-pass filter"] = {
                "cutoff (Hz)": low_pass,
                "Filter order": bpf_order,
            }

    # Determine cohort (if there is one) in the original data
    cohort = get_entity(name_source, "cohort")

    filtered_motion_sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(config.execution.xcp_d_dir.absolute()),
        ),
        name="filtered_motion_sources",
    )

    workflow.connect([
        (inputnode, filtered_motion_sources, [("fmriprep_confounds_file", "in_file")]),
    ])  # fmt:skip

    ds_filtered_motion = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
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
        (inputnode, ds_filtered_motion, [
            ("motion_metadata", "meta_dict"),
            ("filtered_motion", "in_file"),
        ]),
        (filtered_motion_sources, ds_filtered_motion, [("out", "Sources")]),
        (ds_filtered_motion, outputnode, [("out_file", "filtered_motion")]),
    ])  # fmt:skip

    merge_dense_src = pe.Node(
        BIDSURI(
            numinputs=(1 + (1 if fd_thresh > 0 else 0) + (1 if params != "none" else 0)),
            dataset_links=config.execution.dataset_links,
            out_dir=str(config.execution.xcp_d_dir.absolute()),
        ),
        name="merge_dense_src",
        run_without_submitting=True,
        mem_gb=1,
    )
    merge_dense_src.inputs.in1 = name_source

    if fd_thresh > 0:
        temporal_mask_sources = pe.Node(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
            ),
            name="temporal_mask_sources",
        )
        ds_temporal_mask = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["segmentation", "den", "res", "space", "cohort", "desc"],
                suffix="outliers",
                extension=".tsv",
                source_file=name_source,
                # Metadata
                Threshold=fd_thresh,
            ),
            name="ds_temporal_mask",
            run_without_submitting=True,
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, ds_temporal_mask, [
                ("temporal_mask_metadata", "meta_dict"),
                ("temporal_mask", "in_file"),
            ]),
            (ds_filtered_motion, temporal_mask_sources, [("out_file", "in1")]),
            (temporal_mask_sources, ds_temporal_mask, [("out", "Sources")]),
            (ds_temporal_mask, outputnode, [("out_file", "temporal_mask")]),
            (ds_temporal_mask, merge_dense_src, [("out_file", "in2")]),
        ])  # fmt:skip

    if params != "none":
        confounds_src = pe.Node(
            BIDSURI(
                numinputs=(1 + (1 if fd_thresh > 0 else 0) + (1 if custom_confounds_file else 0)),
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
            ),
            name="confounds_src",
            run_without_submitting=True,
            mem_gb=1,
        )
        workflow.connect([(inputnode, confounds_src, [("fmriprep_confounds_file", "in1")])])
        if fd_thresh > 0:
            workflow.connect([(ds_temporal_mask, confounds_src, [("out_file", "in2")])])

            if custom_confounds_file:
                confounds_src.inputs.in3 = custom_confounds_file

        elif custom_confounds_file:
            confounds_src.inputs.in2 = custom_confounds_file

        ds_confounds = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                dismiss_entities=["space", "cohort", "den", "res"],
                datatype="func",
                suffix="design",
                extension=".tsv",
            ),
            name="ds_confounds",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_confounds, [
                ("confounds_file", "in_file"),
                ("confounds_metadata", "meta_dict"),
            ]),
            (confounds_src, ds_confounds, [("out", "Sources")]),
            (ds_confounds, merge_dense_src, [("out_file", f"in{3 if fd_thresh > 0 else 2}")]),
        ])  # fmt:skip

    # Write out derivatives via DerivativesDataSink
    ds_denoised_bold = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["den"],
            cohort=cohort,
            desc="denoised",
            den="91k" if cifti else None,
            extension=".dtseries.nii" if cifti else ".nii.gz",
            # Metadata
            meta_dict=cleaned_data_dictionary,
            SoftwareFilters=software_filters,
        ),
        name="ds_denoised_bold",
        run_without_submitting=True,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_denoised_bold, [("censored_denoised_bold", "in_file")]),
        (merge_dense_src, ds_denoised_bold, [("out", "Sources")]),
        (ds_denoised_bold, outputnode, [("out_file", "censored_denoised_bold")]),
    ])  # fmt:skip

    if dcan_qc and (fd_thresh > 0):
        interpolated_denoised_bold_sources = pe.Node(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
            ),
            name="interpolated_denoised_bold_sources",
        )
        ds_interpolated_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                dismiss_entities=["den"],
                desc="interpolated",
                den="91k" if cifti else None,
                extension=".dtseries.nii" if cifti else ".nii.gz",
                # Metadata
                meta_dict=cleaned_data_dictionary,
            ),
            name="ds_interpolated_denoised_bold",
            run_without_submitting=True,
            mem_gb=2,
        )
        workflow.connect([
            (inputnode, ds_interpolated_denoised_bold, [
                ("denoised_interpolated_bold", "in_file"),
            ]),
            (ds_denoised_bold, interpolated_denoised_bold_sources, [("out_file", "in1")]),
            (interpolated_denoised_bold_sources, ds_interpolated_denoised_bold, [
                ("out", "Sources"),
            ]),
            (ds_interpolated_denoised_bold, outputnode, [
                ("out_file", "denoised_interpolated_bold"),
            ]),
        ])  # fmt:skip

    else:
        workflow.connect([
            (inputnode, outputnode, [
                ("denoised_interpolated_bold", "denoised_interpolated_bold"),
            ]),
        ])  # fmt:skip

    ds_qc_file = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            dismiss_entities=["desc", "den", "res"],
            cohort=cohort,
            den="91k" if cifti else None,
            desc="linc",
            suffix="qc",
            extension=".tsv",
        ),
        name="ds_qc_file",
        run_without_submitting=True,
        mem_gb=1,
    )
    workflow.connect([(inputnode, ds_qc_file, [("qc_file", "in_file")])])

    if smoothing:
        smoothed_bold_sources = pe.Node(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
            ),
            name="smoothed_bold_sources",
        )
        ds_smoothed_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                dismiss_entities=["den"],
                cohort=cohort,
                den="91k" if cifti else None,
                desc="denoisedSmoothed",
                extension=".dtseries.nii" if cifti else ".nii.gz",
                check_hdr=False,
                # Metadata
                SoftwareFilters=software_filters,
                FWHM=smoothing,
            ),
            name="ds_smoothed_bold",
            run_without_submitting=True,
            mem_gb=2,
        )
        workflow.connect([
            (inputnode, ds_smoothed_bold, [("smoothed_denoised_bold", "in_file")]),
            (ds_denoised_bold, smoothed_bold_sources, [("out_file", "in1")]),
            (smoothed_bold_sources, ds_smoothed_bold, [("out", "Sources")]),
            (ds_smoothed_bold, outputnode, [("out_file", "smoothed_denoised_bold")]),
        ])  # fmt:skip

    # Connectivity workflow outputs
    if atlases:
        make_atlas_dict = pe.MapNode(
            GenerateMetadata(
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
                input_names=["Sources"],
            ),
            run_without_submitting=True,
            mem_gb=1,
            name="make_atlas_dict",
            iterfield=["Sources"],
        )
        workflow.connect([(inputnode, make_atlas_dict, [("atlas_files", "Sources")])])

        # Convert Sources to a dictionary, to play well with parcellation MapNodes.
        add_denoised_to_src = pe.MapNode(
            GenerateMetadata(
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
                input_names=["Sources"],
            ),
            run_without_submitting=True,
            mem_gb=1,
            name="add_denoised_to_src",
            iterfield=["metadata"],
        )
        workflow.connect([
            (ds_denoised_bold, add_denoised_to_src, [("out_file", "Sources")]),
            (make_atlas_dict, add_denoised_to_src, [("metadata", "metadata")]),
        ])  # fmt:skip

        # TODO: Add brain mask to Sources (for NIfTIs).
        ds_coverage = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                dismiss_entities=["desc", "den", "res"],
                cohort=cohort,
                statistic="coverage",
                suffix="bold",
                extension=".tsv",
            ),
            name="ds_coverage",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["segmentation", "in_file", "meta_dict"],
        )
        ds_coverage.inputs.segmentation = atlases
        workflow.connect([
            (inputnode, ds_coverage, [("coverage", "in_file")]),
            (make_atlas_dict, ds_coverage, [("metadata", "meta_dict")]),
        ])  # fmt:skip

        add_coverage_to_src = pe.MapNode(
            GenerateMetadata(
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
                input_names=["Sources"],
            ),
            run_without_submitting=True,
            mem_gb=1,
            name="add_coverage_to_src",
            iterfield=["metadata", "Sources"],
        )
        workflow.connect([
            (add_denoised_to_src, add_coverage_to_src, [("metadata", "metadata")]),
            (ds_coverage, add_coverage_to_src, [("out_file", "Sources")]),
        ])  # fmt:skip

        ds_timeseries = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                dismiss_entities=["desc", "den", "res"],
                cohort=cohort,
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
            (inputnode, ds_timeseries, [("timeseries", "in_file")]),
            (add_coverage_to_src, ds_timeseries, [("metadata", "meta_dict")]),
            (ds_timeseries, outputnode, [("out_file", "timeseries")]),
        ])  # fmt:skip

        make_corrs_meta_dict = pe.MapNode(
            GenerateMetadata(
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
                input_names=["Sources", "NodeFiles"],
            ),
            run_without_submitting=True,
            mem_gb=1,
            name="make_corrs_meta_dict",
            iterfield=["Sources", "NodeFiles"],
        )
        workflow.connect([
            (inputnode, make_corrs_meta_dict, [("atlas_files", "NodeFiles")]),
            (ds_timeseries, make_corrs_meta_dict, [("out_file", "Sources")]),
        ])  # fmt:skip

        ds_correlations = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                dismiss_entities=["desc", "den", "res"],
                cohort=cohort,
                statistic="pearsoncorrelation",
                suffix="relmat",
                extension=".tsv",
                # Metadata
                RelationshipMeasure="Pearson correlation coefficient",
                Weighted=True,
                Directed=False,
                ValidDiagonal=False,
                StorageFormat="Full",
            ),
            name="ds_correlations",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["segmentation", "in_file", "meta_dict"],
        )
        ds_correlations.inputs.segmentation = atlases

        workflow.connect([
            (inputnode, ds_correlations, [("correlations", "in_file")]),
            (make_corrs_meta_dict, ds_correlations, [("metadata", "meta_dict")]),
        ])  # fmt:skip

        if cifti:
            ds_coverage_ciftis = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=name_source,
                    check_hdr=False,
                    dismiss_entities=["desc"],
                    cohort=cohort,
                    statistic="coverage",
                    suffix="boldmap",
                    extension=".pscalar.nii",
                ),
                name="ds_coverage_ciftis",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["segmentation", "in_file", "meta_dict"],
            )
            ds_coverage_ciftis.inputs.segmentation = atlases

            workflow.connect([
                (inputnode, ds_coverage_ciftis, [("coverage_ciftis", "in_file")]),
                (add_denoised_to_src, ds_coverage_ciftis, [("metadata", "meta_dict")]),
            ])  # fmt:skip

            add_ccoverage_to_src = pe.MapNode(
                GenerateMetadata(
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(config.execution.xcp_d_dir.absolute()),
                    input_names=["Sources", "NodeFiles"],
                ),
                run_without_submitting=True,
                mem_gb=1,
                name="add_ccoverage_to_src",
                iterfield=["metadata", "Sources"],
            )
            workflow.connect([
                (add_denoised_to_src, add_ccoverage_to_src, [("metadata", "metadata")]),
                (ds_coverage_ciftis, add_ccoverage_to_src, [("out_file", "Sources")]),
            ])  # fmt:skip

            ds_timeseries_ciftis = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=name_source,
                    check_hdr=False,
                    dismiss_entities=["desc", "den"],
                    cohort=cohort,
                    den="91k" if cifti else None,
                    statistic="mean",
                    suffix="timeseries",
                    extension=".ptseries.nii",
                ),
                name="ds_timeseries_ciftis",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["segmentation", "in_file", "meta_dict"],
            )
            ds_timeseries_ciftis.inputs.segmentation = atlases

            workflow.connect([
                (inputnode, ds_timeseries_ciftis, [("timeseries_ciftis", "in_file")]),
                (add_ccoverage_to_src, ds_timeseries_ciftis, [("metadata", "meta_dict")]),
                (ds_timeseries_ciftis, outputnode, [("out_file", "timeseries_ciftis")]),
            ])  # fmt:skip

            make_ccorrs_meta_dict = pe.MapNode(
                GenerateMetadata(
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(config.execution.xcp_d_dir.absolute()),
                    input_names=["Sources", "NodeFiles"],
                ),
                run_without_submitting=True,
                mem_gb=1,
                name="make_ccorrs_meta_dict",
                iterfield=["Sources", "NodeFiles"],
            )
            workflow.connect([
                (inputnode, make_ccorrs_meta_dict, [("atlas_files", "NodeFiles")]),
                (ds_timeseries_ciftis, make_ccorrs_meta_dict, [("out_file", "Sources")]),
            ])  # fmt:skip

            ds_correlation_ciftis = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=name_source,
                    check_hdr=False,
                    dismiss_entities=["desc", "den"],
                    cohort=cohort,
                    den="91k" if cifti else None,
                    statistic="pearsoncorrelation",
                    suffix="boldmap",
                    extension=".pconn.nii",
                    # Metadata
                    RelationshipMeasure="Pearson correlation coefficient",
                    Weighted=True,
                    Directed=False,
                    ValidDiagonal=False,
                    StorageFormat="Full",
                ),
                name="ds_correlation_ciftis",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["segmentation", "in_file", "meta_dict"],
            )
            ds_correlation_ciftis.inputs.segmentation = atlases
            workflow.connect([
                (inputnode, ds_correlation_ciftis, [("correlation_ciftis", "in_file")]),
                (make_ccorrs_meta_dict, ds_correlation_ciftis, [("metadata", "meta_dict")]),
            ])  # fmt:skip

        for i_exact_scan, exact_scan in enumerate(exact_scans):
            select_exact_scan_files = pe.MapNode(
                niu.Select(index=i_exact_scan),
                name=f"select_exact_scan_files_{i_exact_scan}",
                iterfield=["inlist"],
            )
            workflow.connect([
                (inputnode, select_exact_scan_files, [("correlations_exact", "inlist")]),
            ])  # fmt:skip

            ds_correlations_exact = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=name_source,
                    dismiss_entities=["desc", "den", "res"],
                    cohort=cohort,
                    statistic="pearsoncorrelation",
                    desc=f"{exact_scan}volumes",
                    suffix="relmat",
                    extension=".tsv",
                ),
                name=f"ds_correlations_exact_{i_exact_scan}",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["segmentation", "in_file"],
            )
            ds_correlations_exact.inputs.segmentation = atlases
            workflow.connect([
                (select_exact_scan_files, ds_correlations_exact, [("out", "in_file")]),
            ])  # fmt:skip

    # Resting state metric outputs
    reho_sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(config.execution.xcp_d_dir.absolute()),
        ),
        name="reho_sources",
    )
    ds_reho = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=name_source,
            check_hdr=False,
            dismiss_entities=["desc", "den"],
            cohort=cohort,
            den="91k" if cifti else None,
            statistic="reho",
            suffix="boldmap",
            extension=".dscalar.nii" if cifti else ".nii.gz",
            # Metadata
            SoftwareFilters=software_filters,
            Neighborhood="vertices",
        ),
        name="ds_reho",
        run_without_submitting=True,
        mem_gb=1,
    )
    workflow.connect([
        (ds_denoised_bold, reho_sources, [("out_file", "in1")]),
        (inputnode, ds_reho, [("reho", "in_file")]),
        (reho_sources, ds_reho, [("out", "Sources")]),
    ])  # fmt:skip

    if atlases:
        add_reho_to_src = pe.MapNode(
            GenerateMetadata(
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
                input_names=["Sources"],
            ),
            run_without_submitting=True,
            mem_gb=1,
            name="add_reho_to_src",
            iterfield=["metadata"],
        )
        workflow.connect([
            (make_atlas_dict, add_reho_to_src, [("metadata", "metadata")]),
            (ds_reho, add_reho_to_src, [("out_file", "Sources")]),
        ])  # fmt:skip

        ds_parcellated_reho = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                dismiss_entities=["desc", "den", "res"],
                cohort=cohort,
                statistic="reho",
                suffix="bold",
                extension=".tsv",
                # Metadata
                SoftwareFilters=software_filters,
                Neighborhood="vertices",
            ),
            name="ds_parcellated_reho",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["segmentation", "in_file", "meta_dict"],
        )
        ds_parcellated_reho.inputs.segmentation = atlases
        workflow.connect([
            (inputnode, ds_parcellated_reho, [("parcellated_reho", "in_file")]),
            (add_reho_to_src, ds_parcellated_reho, [("metadata", "meta_dict")]),
        ])  # fmt:skip

    if bandpass_filter:
        alff_sources = pe.Node(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(config.execution.xcp_d_dir.absolute()),
            ),
            name="alff_sources",
        )

        ds_alff = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=name_source,
                check_hdr=False,
                dismiss_entities=["desc", "den"],
                cohort=cohort,
                den="91k" if cifti else None,
                statistic="alff",
                suffix="boldmap",
                extension=".dscalar.nii" if cifti else ".nii.gz",
                # Metadata
                SoftwareFilters=software_filters,
            ),
            name="ds_alff",
            run_without_submitting=True,
            mem_gb=1,
        )
        workflow.connect([
            (ds_denoised_bold, alff_sources, [("out_file", "in1")]),
            (inputnode, ds_alff, [("alff", "in_file")]),
            (alff_sources, ds_alff, [("out", "Sources")]),
        ])  # fmt:skip

        if smoothing:
            smoothed_alff_sources = pe.Node(
                BIDSURI(
                    numinputs=1,
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(config.execution.xcp_d_dir.absolute()),
                ),
                name="smoothed_alff_sources",
            )

            ds_smoothed_alff = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=name_source,
                    dismiss_entities=["den"],
                    cohort=cohort,
                    desc="smooth",
                    den="91k" if cifti else None,
                    statistic="alff",
                    suffix="boldmap",
                    extension=".dscalar.nii" if cifti else ".nii.gz",
                    check_hdr=False,
                    # Metadata
                    SoftwareFilters=software_filters,
                    FWHM=smoothing,
                ),
                name="ds_smoothed_alff",
                run_without_submitting=True,
                mem_gb=1,
            )
            workflow.connect([
                (ds_alff, smoothed_alff_sources, [("out_file", "in1")]),
                (inputnode, ds_smoothed_alff, [("smoothed_alff", "in_file")]),
                (smoothed_alff_sources, ds_smoothed_alff, [("out", "Sources")]),
            ])  # fmt:skip

        if atlases:
            add_alff_to_src = pe.MapNode(
                GenerateMetadata(
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(config.execution.xcp_d_dir.absolute()),
                    input_names=["Sources"],
                ),
                run_without_submitting=True,
                mem_gb=1,
                name="add_alff_to_src",
                iterfield=["metadata"],
            )
            workflow.connect([
                (make_atlas_dict, add_alff_to_src, [("metadata", "metadata")]),
                (ds_alff, add_alff_to_src, [("out_file", "Sources")]),
            ])  # fmt:skip

            ds_parcellated_alff = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=name_source,
                    dismiss_entities=["desc", "den", "res"],
                    cohort=cohort,
                    statistic="alff",
                    suffix="bold",
                    extension=".tsv",
                ),
                name="ds_parcellated_alff",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["segmentation", "in_file", "meta_dict"],
            )
            ds_parcellated_alff.inputs.segmentation = atlases
            workflow.connect([
                (inputnode, ds_parcellated_alff, [("parcellated_alff", "in_file")]),
                (add_alff_to_src, ds_parcellated_alff, [("metadata", "meta_dict")]),
            ])  # fmt:skip

    return workflow
