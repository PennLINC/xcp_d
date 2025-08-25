# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for collecting and saving functional outputs."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.config import dismiss_hash
from xcp_d.interfaces.bids import BIDSURI, AddHashToTSV, DerivativesDataSink
from xcp_d.utils.bids import get_entity
from xcp_d.utils.doc import fill_doc


@fill_doc
def init_postproc_derivatives_wf(
    name_source,
    source_metadata,
    exact_scans,
    name='postproc_derivatives_wf',
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
                    name="postproc_derivatives_wf",
                )

    Parameters
    ----------
    name_source : :obj:`str`
        bold or cifti files
    source_metadata : :obj:`dict`
    %(exact_scans)s
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
    denoised_bold
    smoothed_denoised_bold
    alff
        alff nifti
    parcellated_alff
    smoothed_alff
        smoothed alff
    reho
    parcellated_reho
    confounds_tsv
    confounds_metadata
    motion_file
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
    smoothing = config.workflow.smoothing
    file_format = config.workflow.file_format
    output_dir = config.execution.output_dir

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # preprocessing files to use as sources
                'preproc_confounds_file',
                # postprocessed outputs
                'atlas_files',  # for Sources
                'confounds_tsv',
                'confounds_metadata',
                'coverage',
                'timeseries',
                'correlations',
                'correlations_exact',
                'qc_file',
                'denoised_bold',
                'smoothed_denoised_bold',
                'alff',
                'parcellated_alff',
                'smoothed_alff',
                'reho',
                'parcellated_reho',
                'motion_file',
                'motion_metadata',
                'temporal_mask',
                'temporal_mask_metadata',
                'dummy_scans',
                # cifti-only inputs
                'coverage_ciftis',
                'timeseries_ciftis',
                'correlation_ciftis',
                'correlation_ciftis_exact',
                # info for filenames
                'atlas_names',
            ],
        ),
        name='inputnode',
    )

    # Outputs that may be used by the concatenation workflow, in which case we want the actual
    # output filenames for the Sources metadata field.
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'motion_file',
                'temporal_mask',
                'denoised_bold',
                'censored_denoised_bold',
                'smoothed_denoised_bold',
                'timeseries',
                'timeseries_ciftis',
            ],
        ),
        name='outputnode',
    )

    bold_sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='sources',
    )
    bold_sources.inputs.in1 = name_source
    confound_sources = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='confounds',
    )
    workflow.connect([(inputnode, confound_sources, [('preproc_confounds_file', 'in1')])])

    # Create dictionary of basic information
    cleaned_data_dictionary = {
        **source_metadata,
    }
    software_filters = None
    if bandpass_filter:
        software_filters = {}
        if low_pass > 0 and high_pass > 0:
            software_filters['Bandpass filter'] = {
                'Low-pass cutoff (Hz)': low_pass,
                'High-pass cutoff (Hz)': high_pass,
                'Filter order': bpf_order,
            }
        elif high_pass > 0:
            software_filters['High-pass filter'] = {
                'cutoff (Hz)': high_pass,
                'Filter order': bpf_order,
            }
        elif low_pass > 0:
            software_filters['Low-pass filter'] = {
                'cutoff (Hz)': low_pass,
                'Filter order': bpf_order,
            }

    # Determine cohort (if there is one) in the original data
    cohort = get_entity(name_source, 'cohort')

    add_hash_motion = pe.Node(
        AddHashToTSV(
            add_to_columns=True,
            add_to_rows=False,
        ),
        name='add_hash_motion',
    )
    workflow.connect([
        (inputnode, add_hash_motion, [
            ('motion_file', 'in_file'),
            ('motion_metadata', 'metadata'),
        ]),
    ])  # fmt:skip

    ds_motion = pe.Node(
        DerivativesDataSink(
            source_file=name_source,
            dismiss_entities=dismiss_hash(
                ['segmentation', 'den', 'res', 'space', 'cohort', 'desc']
            ),
            suffix='motion',
            extension='.tsv',
        ),
        name='ds_motion',
        run_without_submitting=True,
        mem_gb=1,
    )
    workflow.connect([
        (add_hash_motion, ds_motion, [
            ('metadata', 'meta_dict'),
            ('out_file', 'in_file'),
        ]),
        (confound_sources, ds_motion, [('out', 'Sources')]),
        (ds_motion, outputnode, [('out_file', 'motion_file')]),
    ])  # fmt:skip

    merge_dense_src = pe.Node(
        BIDSURI(
            numinputs=(
                1
                + (1 if fd_thresh > 0 else 0)
                + (1 if config.execution.confounds_config != 'none' else 0)
            ),
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='merge_dense_src',
        mem_gb=1,
    )
    workflow.connect([(bold_sources, merge_dense_src, [('out', 'in1')])])

    if fd_thresh > 0:
        motion_src = pe.Node(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            name='motion_src',
        )
        workflow.connect([(ds_motion, motion_src, [('out_file', 'in1')])])

        add_hash_temporal_mask = pe.Node(
            AddHashToTSV(
                add_to_columns=True,
                add_to_rows=False,
            ),
            name='add_hash_temporal_mask',
        )
        workflow.connect([
            (inputnode, add_hash_temporal_mask, [
                ('temporal_mask', 'in_file'),
                ('temporal_mask_metadata', 'metadata'),
            ]),
        ])  # fmt:skip

        ds_temporal_mask = pe.Node(
            DerivativesDataSink(
                dismiss_entities=dismiss_hash(
                    ['segmentation', 'den', 'res', 'space', 'cohort', 'desc']
                ),
                suffix='outliers',
                extension='.tsv',
                source_file=name_source,
                # Metadata
                Threshold=fd_thresh,
            ),
            name='ds_temporal_mask',
            run_without_submitting=True,
            mem_gb=1,
        )

        workflow.connect([
            (add_hash_temporal_mask, ds_temporal_mask, [
                ('metadata', 'meta_dict'),
                ('out_file', 'in_file'),
            ]),
            (motion_src, ds_temporal_mask, [('out', 'Sources')]),
            (ds_temporal_mask, outputnode, [('out_file', 'temporal_mask')]),
            (ds_temporal_mask, merge_dense_src, [('out_file', 'in2')]),
        ])  # fmt:skip

    if config.execution.confounds_config is not None:
        # XXX: I need to combine collected confounds files as Sources here.
        # Not just the preproc_confounds_file.
        confounds_src = pe.Node(
            BIDSURI(
                numinputs=0 + (1 if fd_thresh > 0 else 0),
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            name='confounds_src',
            run_without_submitting=True,
            mem_gb=1,
        )
        workflow.connect([(inputnode, confounds_src, [('confounds_metadata', 'metadata')])])
        if fd_thresh > 0:
            workflow.connect([(ds_temporal_mask, confounds_src, [('out_file', 'in2')])])

        add_hash_confounds = pe.Node(
            AddHashToTSV(
                add_to_columns=True,
                add_to_rows=False,
            ),
            name='add_hash_confounds',
        )
        workflow.connect([
            (inputnode, add_hash_confounds, [('confounds_tsv', 'in_file')]),
            (confounds_src, add_hash_confounds, [('metadata', 'metadata')]),
        ])  # fmt:skip

        ds_confounds = pe.Node(
            DerivativesDataSink(
                source_file=name_source,
                dismiss_entities=dismiss_hash(['space', 'cohort', 'den', 'res']),
                datatype='func',
                suffix='design',
                extension='.tsv',
            ),
            name='ds_confounds',
            run_without_submitting=True,
        )
        workflow.connect([
            (add_hash_confounds, ds_confounds, [
                ('out_file', 'in_file'),
                ('metadata', 'meta_dict'),
            ]),
            (ds_confounds, merge_dense_src, [('out_file', f'in{3 if fd_thresh > 0 else 2}')]),
        ])  # fmt:skip

    # Write out derivatives via DerivativesDataSink
    ds_denoised_bold = pe.Node(
        DerivativesDataSink(
            source_file=name_source,
            dismiss_entities=dismiss_hash(['den']),
            cohort=cohort,
            desc='denoised',
            den='91k' if file_format == 'cifti' else None,
            extension='.dtseries.nii' if file_format == 'cifti' else '.nii.gz',
            # Metadata
            meta_dict=cleaned_data_dictionary,
            SoftwareFilters=software_filters,
        ),
        name='ds_denoised_bold',
        run_without_submitting=True,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_denoised_bold, [('denoised_bold', 'in_file')]),
        (merge_dense_src, ds_denoised_bold, [('out', 'Sources')]),
        (ds_denoised_bold, outputnode, [('out_file', 'denoised_bold')]),
    ])  # fmt:skip

    if config.workflow.linc_qc:
        add_hash_qc_file = pe.Node(
            AddHashToTSV(
                add_to_columns=True,
                add_to_rows=False,
            ),
            name='add_hash_qc_file',
        )
        workflow.connect([(inputnode, add_hash_qc_file, [('qc_file', 'in_file')])])

        ds_qc_file = pe.Node(
            DerivativesDataSink(
                source_file=name_source,
                dismiss_entities=dismiss_hash(['desc', 'den', 'res']),
                cohort=cohort,
                den='91k' if file_format == 'cifti' else None,
                desc='linc',
                suffix='qc',
                extension='.tsv',
            ),
            name='ds_qc_file',
            run_without_submitting=True,
            mem_gb=1,
        )
        workflow.connect([(add_hash_qc_file, ds_qc_file, [('out_file', 'in_file')])])

    if smoothing:
        smoothed_bold_src = pe.Node(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            name='smoothed_bold_src',
            run_without_submitting=True,
            mem_gb=1,
        )
        workflow.connect([(ds_denoised_bold, smoothed_bold_src, [('out_file', 'in1')])])

        # Write out derivatives via DerivativesDataSink
        ds_smoothed_bold = pe.Node(
            DerivativesDataSink(
                source_file=name_source,
                dismiss_entities=dismiss_hash(['den']),
                cohort=cohort,
                den='91k' if file_format == 'cifti' else None,
                desc='denoisedSmoothed',
                extension='.dtseries.nii' if file_format == 'cifti' else '.nii.gz',
                check_hdr=False,
                # Metadata
                SoftwareFilters=software_filters,
                FWHM=smoothing,
            ),
            name='ds_smoothed_bold',
            run_without_submitting=True,
            mem_gb=2,
        )
        workflow.connect([
            (inputnode, ds_smoothed_bold, [('smoothed_denoised_bold', 'in_file')]),
            (smoothed_bold_src, ds_smoothed_bold, [('out', 'Sources')]),
            (ds_smoothed_bold, outputnode, [('out_file', 'smoothed_denoised_bold')]),
        ])  # fmt:skip

    # Connectivity workflow outputs
    if config.execution.atlases:
        make_atlas_dict = pe.MapNode(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            run_without_submitting=True,
            mem_gb=1,
            name='make_atlas_dict',
            iterfield=['in1'],
        )
        workflow.connect([(inputnode, make_atlas_dict, [('atlas_files', 'in1')])])

        # Convert Sources to a dictionary, to play well with parcellation MapNodes.
        add_denoised_to_src = pe.MapNode(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            run_without_submitting=True,
            mem_gb=1,
            name='add_denoised_to_src',
            iterfield=['metadata'],
        )
        workflow.connect([
            (make_atlas_dict, add_denoised_to_src, [('metadata', 'metadata')]),
            (ds_denoised_bold, add_denoised_to_src, [('out_file', 'in1')]),
        ])  # fmt:skip

        # TODO: Add brain mask to Sources (for NIfTIs).
        add_hash_coverage = pe.MapNode(
            AddHashToTSV(
                add_to_columns=True,
                add_to_rows=False,
            ),
            name='add_hash_coverage',
            iterfield=['in_file', 'metadata'],
        )
        workflow.connect([
            (inputnode, add_hash_coverage, [('coverage', 'in_file')]),
            (make_atlas_dict, add_hash_coverage, [('metadata', 'metadata')]),
        ])  # fmt:skip

        ds_coverage = pe.MapNode(
            DerivativesDataSink(
                source_file=name_source,
                dismiss_entities=dismiss_hash(['desc', 'den', 'res']),
                cohort=cohort,
                statistic='coverage',
                suffix='bold',
                extension='.tsv',
            ),
            name='ds_coverage',
            run_without_submitting=True,
            mem_gb=1,
            iterfield=['segmentation', 'in_file', 'meta_dict'],
        )
        workflow.connect([
            (inputnode, ds_coverage, [('atlas_names', 'segmentation')]),
            (add_hash_coverage, ds_coverage, [
                ('out_file', 'in_file'),
                ('metadata', 'meta_dict'),
            ]),
        ])  # fmt:skip

        add_coverage_to_src = pe.MapNode(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            run_without_submitting=True,
            mem_gb=1,
            name='add_coverage_to_src',
            iterfield=['metadata', 'in1'],
        )
        workflow.connect([
            (add_denoised_to_src, add_coverage_to_src, [('metadata', 'metadata')]),
            (ds_coverage, add_coverage_to_src, [('out_file', 'in1')]),
        ])  # fmt:skip

        add_hash_timeseries = pe.MapNode(
            AddHashToTSV(
                add_to_columns=True,
                add_to_rows=False,
            ),
            name='add_hash_timeseries',
            iterfield=['in_file', 'metadata'],
        )
        workflow.connect([
            (inputnode, add_hash_timeseries, [('timeseries', 'in_file')]),
            (add_coverage_to_src, add_hash_timeseries, [('metadata', 'metadata')]),
        ])  # fmt:skip

        ds_timeseries = pe.MapNode(
            DerivativesDataSink(
                source_file=name_source,
                dismiss_entities=dismiss_hash(['desc', 'den', 'res']),
                cohort=cohort,
                statistic='mean',
                suffix='timeseries',
                extension='.tsv',
                # Metadata
                SamplingFrequency='TR',
            ),
            name='ds_timeseries',
            run_without_submitting=True,
            mem_gb=1,
            iterfield=['segmentation', 'in_file', 'meta_dict'],
        )
        workflow.connect([
            (inputnode, ds_timeseries, [('atlas_names', 'segmentation')]),
            (add_hash_timeseries, ds_timeseries, [
                ('out_file', 'in_file'),
                ('metadata', 'meta_dict'),
            ]),
            (ds_timeseries, outputnode, [('out_file', 'timeseries')]),
        ])  # fmt:skip

        if 'all' in config.workflow.correlation_lengths:
            make_corrs_meta_dict1 = pe.MapNode(
                BIDSURI(
                    numinputs=1,
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(output_dir),
                ),
                run_without_submitting=True,
                mem_gb=1,
                name='make_corrs_meta_dict1',
                iterfield=['in1'],
            )
            workflow.connect([(ds_timeseries, make_corrs_meta_dict1, [('out_file', 'in1')])])

            make_corrs_meta_dict2 = pe.MapNode(
                BIDSURI(
                    numinputs=1,
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(output_dir),
                    field='NodeFiles',
                ),
                run_without_submitting=True,
                mem_gb=1,
                name='make_corrs_meta_dict2',
                iterfield=['in1', 'metadata'],
            )
            workflow.connect([
                (inputnode, make_corrs_meta_dict2, [('atlas_files', 'in1')]),
                (make_corrs_meta_dict1, make_corrs_meta_dict2, [('metadata', 'metadata')]),
            ])  # fmt:skip

            add_hash_correlations = pe.MapNode(
                AddHashToTSV(
                    add_to_columns=True,
                    add_to_rows=True,
                ),
                name='add_hash_correlations',
                iterfield=['in_file', 'metadata'],
            )
            workflow.connect([
                (inputnode, add_hash_correlations, [('correlations', 'in_file')]),
                (make_corrs_meta_dict2, add_hash_correlations, [('metadata', 'metadata')]),
            ])  # fmt:skip

            ds_correlations = pe.MapNode(
                DerivativesDataSink(
                    source_file=name_source,
                    dismiss_entities=dismiss_hash(['desc', 'den', 'res']),
                    cohort=cohort,
                    statistic='pearsoncorrelation',
                    suffix='relmat',
                    extension='.tsv',
                    # Metadata
                    RelationshipMeasure='Pearson correlation coefficient',
                    Weighted=True,
                    Directed=False,
                    ValidDiagonal=False,
                    StorageFormat='Full',
                ),
                name='ds_correlations',
                run_without_submitting=True,
                mem_gb=1,
                iterfield=['segmentation', 'in_file', 'meta_dict'],
            )
            workflow.connect([
                (inputnode, ds_correlations, [('atlas_names', 'segmentation')]),
                (add_hash_correlations, ds_correlations, [
                    ('out_file', 'in_file'),
                    ('metadata', 'meta_dict'),
                ]),
            ])  # fmt:skip

        if file_format == 'cifti':
            ds_coverage_ciftis = pe.MapNode(
                DerivativesDataSink(
                    source_file=name_source,
                    check_hdr=False,
                    dismiss_entities=dismiss_hash(['desc']),
                    cohort=cohort,
                    statistic='coverage',
                    suffix='boldmap',
                    extension='.pscalar.nii',
                ),
                name='ds_coverage_ciftis',
                run_without_submitting=True,
                mem_gb=1,
                iterfield=['segmentation', 'in_file', 'meta_dict'],
            )
            workflow.connect([
                (inputnode, ds_coverage_ciftis, [
                    ('atlas_names', 'segmentation'),
                    ('coverage_ciftis', 'in_file'),
                ]),
                (add_denoised_to_src, ds_coverage_ciftis, [('metadata', 'meta_dict')]),
            ])  # fmt:skip

            add_ccoverage_to_src = pe.MapNode(
                BIDSURI(
                    numinputs=1,
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(output_dir),
                ),
                run_without_submitting=True,
                mem_gb=1,
                name='add_ccoverage_to_src',
                iterfield=['metadata', 'in1'],
            )
            workflow.connect([
                (add_denoised_to_src, add_ccoverage_to_src, [('metadata', 'metadata')]),
                (ds_coverage_ciftis, add_ccoverage_to_src, [('out_file', 'in1')]),
            ])  # fmt:skip

            ds_timeseries_ciftis = pe.MapNode(
                DerivativesDataSink(
                    source_file=name_source,
                    check_hdr=False,
                    dismiss_entities=dismiss_hash(['desc', 'den']),
                    cohort=cohort,
                    den='91k' if file_format == 'cifti' else None,
                    statistic='mean',
                    suffix='timeseries',
                    extension='.ptseries.nii',
                ),
                name='ds_timeseries_ciftis',
                run_without_submitting=True,
                mem_gb=1,
                iterfield=['segmentation', 'in_file', 'meta_dict'],
            )
            workflow.connect([
                (inputnode, ds_timeseries_ciftis, [
                    ('atlas_names', 'segmentation'),
                    ('timeseries_ciftis', 'in_file'),
                ]),
                (add_ccoverage_to_src, ds_timeseries_ciftis, [('metadata', 'meta_dict')]),
                (ds_timeseries_ciftis, outputnode, [('out_file', 'timeseries_ciftis')]),
            ])  # fmt:skip

            if 'all' in config.workflow.correlation_lengths:
                make_ccorrs_meta_dict1 = pe.MapNode(
                    BIDSURI(
                        numinputs=1,
                        dataset_links=config.execution.dataset_links,
                        out_dir=str(output_dir),
                    ),
                    run_without_submitting=True,
                    mem_gb=1,
                    name='make_ccorrs_meta_dict1',
                    iterfield=['in1'],
                )
                workflow.connect([
                    (ds_timeseries_ciftis, make_ccorrs_meta_dict1, [('out_file', 'in1')]),
                ])  # fmt:skip

                make_ccorrs_meta_dict2 = pe.MapNode(
                    BIDSURI(
                        numinputs=1,
                        dataset_links=config.execution.dataset_links,
                        out_dir=str(output_dir),
                        field='NodeFiles',
                    ),
                    run_without_submitting=True,
                    mem_gb=1,
                    name='make_ccorrs_meta_dict2',
                    iterfield=['in1', 'metadata'],
                )
                workflow.connect([
                    (inputnode, make_ccorrs_meta_dict2, [('atlas_files', 'in1')]),
                    (make_ccorrs_meta_dict1, make_ccorrs_meta_dict2, [('metadata', 'metadata')]),
                ])  # fmt:skip

                ds_correlation_ciftis = pe.MapNode(
                    DerivativesDataSink(
                        source_file=name_source,
                        check_hdr=False,
                        dismiss_entities=dismiss_hash(['desc', 'den']),
                        cohort=cohort,
                        den='91k' if file_format == 'cifti' else None,
                        statistic='pearsoncorrelation',
                        suffix='boldmap',
                        extension='.pconn.nii',
                        # Metadata
                        RelationshipMeasure='Pearson correlation coefficient',
                        Weighted=True,
                        Directed=False,
                        ValidDiagonal=False,
                        StorageFormat='Full',
                    ),
                    name='ds_correlation_ciftis',
                    run_without_submitting=True,
                    mem_gb=1,
                    iterfield=['segmentation', 'in_file', 'meta_dict'],
                )
                workflow.connect([
                    (inputnode, ds_correlation_ciftis, [
                        ('atlas_names', 'segmentation'),
                        ('correlation_ciftis', 'in_file'),
                    ]),
                    (make_ccorrs_meta_dict2, ds_correlation_ciftis, [('metadata', 'meta_dict')]),
                ])  # fmt:skip

        for i_exact_scan, exact_scan in enumerate(exact_scans):
            select_exact_scan_files = pe.MapNode(
                niu.Select(index=i_exact_scan),
                name=f'select_exact_scan_files_{i_exact_scan}',
                iterfield=['inlist'],
            )
            workflow.connect([
                (inputnode, select_exact_scan_files, [('correlations_exact', 'inlist')]),
            ])  # fmt:skip

            add_hash_correlations_exact = pe.MapNode(
                AddHashToTSV(
                    add_to_columns=True,
                    add_to_rows=True,
                ),
                name=f'add_hash_correlations_exact_{i_exact_scan}',
                iterfield=['in_file'],
            )
            workflow.connect([
                (select_exact_scan_files, add_hash_correlations_exact, [('out', 'in_file')]),
            ])  # fmt:skip

            ds_correlations_exact = pe.MapNode(
                DerivativesDataSink(
                    source_file=name_source,
                    dismiss_entities=dismiss_hash(['desc', 'den', 'res']),
                    cohort=cohort,
                    statistic='pearsoncorrelation',
                    desc=f'{exact_scan}volumes',
                    suffix='relmat',
                    extension='.tsv',
                ),
                name=f'ds_correlations_exact_{i_exact_scan}',
                run_without_submitting=True,
                mem_gb=1,
                iterfield=['segmentation', 'in_file'],
            )
            workflow.connect([
                (inputnode, ds_correlations_exact, [('atlas_names', 'segmentation')]),
                (add_hash_correlations_exact, ds_correlations_exact, [('out_file', 'in_file')]),
            ])  # fmt:skip

    # Resting state metric outputs
    denoised_src = pe.Node(
        BIDSURI(
            numinputs=1,
            dataset_links=config.execution.dataset_links,
            out_dir=str(output_dir),
        ),
        name='denoised_src',
        run_without_submitting=True,
        mem_gb=1,
    )
    workflow.connect([(ds_denoised_bold, denoised_src, [('out_file', 'in1')])])

    ds_reho = pe.Node(
        DerivativesDataSink(
            source_file=name_source,
            check_hdr=False,
            dismiss_entities=dismiss_hash(['desc', 'den']),
            cohort=cohort,
            den='91k' if file_format == 'cifti' else None,
            statistic='reho',
            suffix='boldmap',
            extension='.dscalar.nii' if file_format == 'cifti' else '.nii.gz',
            # Metadata
            SoftwareFilters=software_filters,
            Neighborhood='vertices',
        ),
        name='ds_reho',
        run_without_submitting=True,
        mem_gb=1,
    )
    workflow.connect([
        (inputnode, ds_reho, [('reho', 'in_file')]),
        (denoised_src, ds_reho, [('out', 'Sources')]),
    ])  # fmt:skip

    if config.execution.atlases:
        add_reho_to_src = pe.MapNode(
            BIDSURI(
                numinputs=1,
                dataset_links=config.execution.dataset_links,
                out_dir=str(output_dir),
            ),
            run_without_submitting=True,
            mem_gb=1,
            name='add_reho_to_src',
            iterfield=['metadata'],
        )
        workflow.connect([
            (make_atlas_dict, add_reho_to_src, [('metadata', 'metadata')]),
            (ds_reho, add_reho_to_src, [('out_file', 'in1')]),
        ])  # fmt:skip

        add_hash_parcellated_reho = pe.MapNode(
            AddHashToTSV(
                add_to_columns=True,
                add_to_rows=False,
            ),
            name='add_hash_parcellated_reho',
            iterfield=['in_file', 'metadata'],
        )
        workflow.connect([
            (inputnode, add_hash_parcellated_reho, [('parcellated_reho', 'in_file')]),
            (add_reho_to_src, add_hash_parcellated_reho, [('metadata', 'metadata')]),
        ])  # fmt:skip

        ds_parcellated_reho = pe.MapNode(
            DerivativesDataSink(
                source_file=name_source,
                dismiss_entities=dismiss_hash(['desc', 'den', 'res']),
                cohort=cohort,
                statistic='reho',
                suffix='bold',
                extension='.tsv',
                # Metadata
                SoftwareFilters=software_filters,
                Neighborhood='vertices',
            ),
            name='ds_parcellated_reho',
            run_without_submitting=True,
            mem_gb=1,
            iterfield=['segmentation', 'in_file', 'meta_dict'],
        )
        workflow.connect([
            (inputnode, ds_parcellated_reho, [('atlas_names', 'segmentation')]),
            (add_hash_parcellated_reho, ds_parcellated_reho, [
                ('out_file', 'in_file'),
                ('metadata', 'meta_dict'),
            ]),
        ])  # fmt:skip

    if bandpass_filter:
        ds_alff = pe.Node(
            DerivativesDataSink(
                source_file=name_source,
                check_hdr=False,
                dismiss_entities=dismiss_hash(['desc', 'den']),
                cohort=cohort,
                den='91k' if file_format == 'cifti' else None,
                statistic='alff',
                suffix='boldmap',
                extension='.dscalar.nii' if file_format == 'cifti' else '.nii.gz',
                # Metadata
                SoftwareFilters=software_filters,
            ),
            name='ds_alff',
            run_without_submitting=True,
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, ds_alff, [('alff', 'in_file')]),
            (denoised_src, ds_alff, [('out', 'Sources')]),
        ])  # fmt:skip

        if smoothing:
            alff_src = pe.Node(
                BIDSURI(
                    numinputs=1,
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(output_dir),
                ),
                name='alff_src',
                run_without_submitting=True,
                mem_gb=1,
            )
            workflow.connect([(ds_alff, alff_src, [('out_file', 'in1')])])

            ds_smoothed_alff = pe.Node(
                DerivativesDataSink(
                    source_file=name_source,
                    dismiss_entities=dismiss_hash(['den']),
                    cohort=cohort,
                    desc='smooth',
                    den='91k' if file_format == 'cifti' else None,
                    statistic='alff',
                    suffix='boldmap',
                    extension='.dscalar.nii' if file_format == 'cifti' else '.nii.gz',
                    check_hdr=False,
                    # Metadata
                    SoftwareFilters=software_filters,
                    FWHM=smoothing,
                ),
                name='ds_smoothed_alff',
                run_without_submitting=True,
                mem_gb=1,
            )
            workflow.connect([
                (inputnode, ds_smoothed_alff, [('smoothed_alff', 'in_file')]),
                (alff_src, ds_smoothed_alff, [('out', 'Sources')]),
            ])  # fmt:skip

        if config.execution.atlases:
            add_alff_to_src = pe.MapNode(
                BIDSURI(
                    numinputs=1,
                    dataset_links=config.execution.dataset_links,
                    out_dir=str(output_dir),
                ),
                run_without_submitting=True,
                mem_gb=1,
                name='add_alff_to_src',
                iterfield=['metadata'],
            )
            workflow.connect([
                (make_atlas_dict, add_alff_to_src, [('metadata', 'metadata')]),
                (ds_alff, add_alff_to_src, [('out_file', 'in1')]),
            ])  # fmt:skip

            add_hash_parcellated_alff = pe.MapNode(
                AddHashToTSV(
                    add_to_columns=True,
                    add_to_rows=False,
                ),
                name='add_hash_parcellated_alff',
                iterfield=['in_file', 'metadata'],
            )
            workflow.connect([
                (inputnode, add_hash_parcellated_alff, [('parcellated_alff', 'in_file')]),
                (add_alff_to_src, add_hash_parcellated_alff, [('metadata', 'metadata')]),
            ])  # fmt:skip

            ds_parcellated_alff = pe.MapNode(
                DerivativesDataSink(
                    source_file=name_source,
                    dismiss_entities=dismiss_hash(['desc', 'den', 'res']),
                    cohort=cohort,
                    statistic='alff',
                    suffix='bold',
                    extension='.tsv',
                ),
                name='ds_parcellated_alff',
                run_without_submitting=True,
                mem_gb=1,
                iterfield=['segmentation', 'in_file', 'meta_dict'],
            )
            workflow.connect([
                (inputnode, ds_parcellated_alff, [('atlas_names', 'segmentation')]),
                (add_hash_parcellated_alff, ds_parcellated_alff, [
                    ('out_file', 'in_file'),
                    ('metadata', 'meta_dict'),
                ]),
            ])  # fmt:skip

    return workflow
