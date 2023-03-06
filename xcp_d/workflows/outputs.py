# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for collecting and saving xcp_d outputs."""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.utils.bids import get_entity
from xcp_d.utils.doc import fill_doc


@fill_doc
def init_postproc_derivatives_wf(
    bold_file,
    bandpass_filter,
    low_pass,
    high_pass,
    motion_filter_type,
    smoothing,
    params,
    cifti,
    dcan_qc,
    output_dir,
    TR,
    name="postproc_derivatives_wf",
):
    """Write out the xcp_d derivatives in BIDS format.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.outputs import init_postproc_derivatives_wf

            wf = init_postproc_derivatives_wf(
                bold_file="/path/to/file.nii.gz",
                bandpass_filter=True,
                low_pass=0.1,
                high_pass=0.008,
                motion_filter_type=None,
                smoothing=6,
                params="36P",
                cifti=False,
                dcan_qc=True,
                output_dir=".",
                TR=2.,
                name="postproc_derivatives_wf",
            )

    Parameters
    ----------
    bold_file : str
        bold or cifti files
    low_pass : float
        low pass filter
    high_pass : float
        high pass filter
    %(motion_filter_type)s
    %(smoothing)s
    %(params)s
    %(cifti)s
    %(dcan_qc)s
    output_dir : str
        output directory
    %(TR)s
    %(name)s
        Default is "connectivity_wf".

    Inputs
    ------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    %(timeseries)s
    %(correlations)s
    %(coverage)s
    %(timeseries_ciftis)s
    %(correlation_ciftis)s
    %(coverage_ciftis)s
    qc_file
        LINC-style quality control file
    %(interpolated_filtered_bold)s
    censored_denoised_bold
        clean bold after censoring, regression, interpolation, filtering, and re-censoring
    smoothed_bold
        clean bold after censoring, regression, interpolation, filtering, re-censoring, and
        smoothing
    alff
        alff nifti
    smoothed_alff
        smoothed alff
    reho
    confounds_file
    %(filtered_motion)s
    filtered_motion_metadata
    %(temporal_mask)s
    tmask_metadata
    %(dummy_scans)s
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "atlas_names",
                "confounds_file",
                "coverage",
                "timeseries",
                "correlations",
                "qc_file",
                "censored_denoised_bold",
                "smoothed_bold",
                "interpolated_filtered_bold",
                "alff",
                "smoothed_alff",
                "reho_lh",
                "reho_rh",
                "reho",
                "filtered_motion",
                "filtered_motion_metadata",
                "temporal_mask",
                "tmask_metadata",
                "dummy_scans",
                # cifti-only inputs
                "coverage_ciftis",
                "timeseries_ciftis",
                "correlation_ciftis",
            ],
        ),
        name="inputnode",
    )

    # Create dictionary of basic information
    cleaned_data_dictionary = {
        "RepetitionTime": TR,
        "nuisance parameters": params,
    }
    if bandpass_filter:
        cleaned_data_dictionary["Freq Band"] = [high_pass, low_pass]

    smoothed_data_dictionary = {"FWHM": smoothing}  # Separate dictionary for smoothing

    # Determine cohort (if there is one) in the original data
    cohort = get_entity(bold_file, "cohort")

    ds_temporal_mask = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["atlas", "den", "res", "space", "cohort", "desc"],
            suffix="outliers",
            extension=".tsv",
            source_file=bold_file,
        ),
        name="ds_temporal_mask",
        run_without_submitting=True,
        mem_gb=1,
    )

    # fmt:off
    workflow.connect([(inputnode, ds_temporal_mask, [("tmask_metadata", "meta_dict")])])
    # fmt:on

    ds_filtered_motion = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
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
        (inputnode, ds_filtered_motion, [("filtered_motion_metadata", "meta_dict")]),
    ])
    # fmt:on

    ds_confounds = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            dismiss_entities=["space", "cohort", "den", "res"],
            datatype="func",
            suffix="design",
            extension=".tsv",
        ),
        name="ds_confounds",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_temporal_mask, [("temporal_mask", "in_file")]),
        (inputnode, ds_filtered_motion, [("filtered_motion", "in_file")]),
        (inputnode, ds_confounds, [("confounds_file", "in_file")])
    ])
    # fmt:on

    ds_coverage_files = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            dismiss_entities=["desc"],
            cohort=cohort,
            suffix="coverage",
            extension=".tsv",
        ),
        name="ds_coverage_files",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )
    ds_timeseries = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            dismiss_entities=["desc"],
            cohort=cohort,
            suffix="timeseries",
            extension=".tsv",
        ),
        name="ds_timeseries",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )
    ds_correlations = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            source_file=bold_file,
            dismiss_entities=["desc"],
            cohort=cohort,
            measure="pearsoncorrelation",
            suffix="conmat",
            extension=".tsv",
        ),
        name="ds_correlations",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["atlas", "in_file"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_coverage_files, [
            ("coverage", "in_file"),
            ("atlas_names", "atlas"),
        ]),
        (inputnode, ds_timeseries, [
            ("timeseries", "in_file"),
            ("atlas_names", "atlas"),
        ]),
        (inputnode, ds_correlations, [
            ("correlations", "in_file"),
            ("atlas_names", "atlas"),
        ]),
    ])
    # fmt:on

    # Write out detivatives via DerivativesDataSink
    if not cifti:  # if Nifti
        ds_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                meta_dict=cleaned_data_dictionary,
                source_file=bold_file,
                cohort=cohort,
                desc="denoised",
                extension=".nii.gz",
                compression=True,
            ),
            name="ds_denoised_bold",
            run_without_submitting=True,
            mem_gb=2,
        )

        if dcan_qc:
            ds_interpolated_denoised_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=cleaned_data_dictionary,
                    source_file=bold_file,
                    desc="interpolated",
                    extension=".nii.gz",
                    compression=True,
                ),
                name="ds_interpolated_denoised_bold",
                run_without_submitting=True,
                mem_gb=2,
            )

        ds_qc_file = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=["desc"],
                cohort=cohort,
                desc="linc",
                suffix="qc",
                extension=".csv",
            ),
            name="ds_qc_file",
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_reho = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=["desc"],
                cohort=cohort,
                suffix="reho",
                extension=".nii.gz",
                compression=True,
            ),
            name="ds_reho",
            run_without_submitting=True,
            mem_gb=1,
        )

        if bandpass_filter:
            ds_alff = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=bold_file,
                    dismiss_entities=["desc"],
                    cohort=cohort,
                    suffix="alff",
                    extension=".nii.gz",
                    compression=True,
                ),
                name="ds_alff",
                run_without_submitting=True,
                mem_gb=1,
            )

        if smoothing:  # if smoothed
            # Write out detivatives via DerivativesDataSink
            ds_smoothed_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    cohort=cohort,
                    desc="denoisedSmoothed",
                    extension=".nii.gz",
                    compression=True,
                ),
                name="ds_smoothed_bold",
                run_without_submitting=True,
                mem_gb=2,
            )

            if bandpass_filter:
                ds_smoothed_alff = pe.Node(
                    DerivativesDataSink(
                        base_directory=output_dir,
                        meta_dict=smoothed_data_dictionary,
                        source_file=bold_file,
                        cohort=cohort,
                        desc="smooth",
                        suffix="alff",
                        extension=".nii.gz",
                        compression=True,
                    ),
                    name="ds_smoothed_alff",
                    run_without_submitting=True,
                    mem_gb=1,
                )

    else:  # For cifti files
        # Write out derivatives via DerivativesDataSink
        ds_denoised_bold = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                meta_dict=cleaned_data_dictionary,
                source_file=bold_file,
                dismiss_entities=["den"],
                cohort=cohort,
                desc="denoised",
                den="91k",
                extension=".dtseries.nii",
            ),
            name="ds_denoised_bold",
            run_without_submitting=True,
            mem_gb=2,
        )

        if dcan_qc:
            ds_interpolated_denoised_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=cleaned_data_dictionary,
                    source_file=bold_file,
                    dismiss_entities=["den"],
                    desc="interpolated",
                    den="91k",
                    extension=".dtseries.nii",
                ),
                name="ds_interpolated_denoised_bold",
                run_without_submitting=True,
                mem_gb=2,
            )

        ds_qc_file = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                dismiss_entities=["desc", "den"],
                cohort=cohort,
                den="91k",
                desc="linc",
                suffix="qc",
                extension=".csv",
            ),
            name="ds_qc_file",
            run_without_submitting=True,
            mem_gb=1,
        )

        ds_coverage_cifti_files = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=["desc"],
                cohort=cohort,
                suffix="coverage",
                extension=".pscalar.nii",
            ),
            name="ds_coverage_cifti_files",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )
        ds_timeseries_cifti_files = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=["desc", "den"],
                cohort=cohort,
                den="91k",
                suffix="timeseries",
                extension=".ptseries.nii",
            ),
            name="ds_timeseries_cifti_files",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )
        ds_correlation_cifti_files = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=["desc", "den"],
                cohort=cohort,
                den="91k",
                measure="pearsoncorrelation",
                suffix="conmat",
                extension=".pconn.nii",
            ),
            name="ds_correlation_cifti_files",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_coverage_cifti_files, [
                ("coverage_ciftis", "in_file"),
                ("atlas_names", "atlas"),
            ]),
            (inputnode, ds_timeseries_cifti_files, [
                ("timeseries_ciftis", "in_file"),
                ("atlas_names", "atlas"),
            ]),
            (inputnode, ds_correlation_cifti_files, [
                ("correlation_ciftis", "in_file"),
                ("atlas_names", "atlas"),
            ]),
        ])
        # fmt:on

        ds_reho = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                source_file=bold_file,
                check_hdr=False,
                dismiss_entities=["desc", "den"],
                cohort=cohort,
                den="91k",
                suffix="reho",
                extension=".dscalar.nii",
            ),
            name="ds_reho",
            run_without_submitting=True,
            mem_gb=1,
        )

        if bandpass_filter:
            ds_alff = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    source_file=bold_file,
                    check_hdr=False,
                    dismiss_entities=["desc", "den"],
                    cohort=cohort,
                    den="91k",
                    suffix="alff",
                    extension=".dscalar.nii",
                ),
                name="ds_alff",
                run_without_submitting=True,
                mem_gb=1,
            )

        if smoothing:  # If smoothed
            # Write out detivatives via DerivativesDataSink
            ds_smoothed_bold = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    meta_dict=smoothed_data_dictionary,
                    source_file=bold_file,
                    dismiss_entities=["den"],
                    cohort=cohort,
                    den="91k",
                    desc="denoisedSmoothed",
                    extension=".dtseries.nii",
                    check_hdr=False,
                ),
                name="ds_smoothed_bold",
                run_without_submitting=True,
                mem_gb=2,
            )

            if bandpass_filter:
                ds_smoothed_alff = pe.Node(
                    DerivativesDataSink(
                        base_directory=output_dir,
                        meta_dict=smoothed_data_dictionary,
                        source_file=bold_file,
                        dismiss_entities=["den"],
                        cohort=cohort,
                        desc="smooth",
                        den="91k",
                        suffix="alff",
                        extension=".dscalar.nii",
                        check_hdr=False,
                    ),
                    name="ds_smoothed_alff",
                    run_without_submitting=True,
                    mem_gb=1,
                )

    # fmt:off
    workflow.connect([
        (inputnode, ds_denoised_bold, [("censored_denoised_bold", "in_file")]),
        (inputnode, ds_qc_file, [("qc_file", "in_file")]),
        (inputnode, ds_reho, [("reho", "in_file")]),
    ])
    # fmt:on

    if dcan_qc:
        # fmt:off
        workflow.connect([
            (inputnode, ds_interpolated_denoised_bold, [
                ("interpolated_filtered_bold", "in_file"),
            ]),
        ])
        # fmt:on

    if bandpass_filter:
        workflow.connect([(inputnode, ds_alff, [("alff", "in_file")])])

    if smoothing:
        workflow.connect([(inputnode, ds_smoothed_bold, [("smoothed_bold", "in_file")])])

        if bandpass_filter:
            workflow.connect([(inputnode, ds_smoothed_alff, [("smoothed_alff", "in_file")])])

    return workflow
