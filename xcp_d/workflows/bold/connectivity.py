# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for extracting time series and computing functional connectivity."""

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.utils.atlas import select_atlases
from xcp_d.utils.boilerplate import describe_atlases
from xcp_d.utils.doc import fill_doc
from xcp_d.workflows.parcellation import init_parcellate_cifti_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_functional_connectivity_nifti_wf(mem_gb, name="connectivity_wf"):
    """Extract BOLD time series and compute functional connectivity.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.connectivity import init_functional_connectivity_nifti_wf

            with mock_config():
                config.execution.atlases = ["Glasser", "Gordon"]

                wf = init_functional_connectivity_nifti_wf(
                    mem_gb={"resampled": 0.1, "timeseries": 1.0},
                )

    Parameters
    ----------
    mem_gb : :obj:`dict`
        Dictionary of memory allocations.
    %(name)s
        Default is "connectivity_wf".

    Inputs
    ------
    %(name_source)s
    denoised_bold
        clean bold after filtered out nuisscance and filtering
    %(temporal_mask)s
    alff
    reho
    %(atlases)s
    atlas_files
    atlas_labels_files

    Outputs
    -------
    %(coverage)s
    %(timeseries)s
    %(correlations)s
    %(correlations_exact)s
    parcellated_alff
    parcellated_reho
    """
    from xcp_d.interfaces.connectivity import ConnectPlot, NiftiParcellate, TSVConnect

    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir
    bandpass_filter = config.workflow.bandpass_filter
    min_coverage = config.workflow.min_coverage

    atlas_str = describe_atlases(config.execution.atlases)

    workflow.__desc__ = f"""
Processed functional timeseries were extracted from the residual BOLD signal
with *Nilearn's* *NiftiLabelsMasker* for the following atlases: {atlas_str}.
Corresponding pair-wise functional connectivity between all regions was computed for each atlas,
which was operationalized as the Pearson's correlation of each parcel's unsmoothed timeseries.
In cases of partial coverage, uncovered voxels (values of all zeros or NaNs) were either
ignored (when the parcel had >{min_coverage * 100}% coverage)
or were set to zero (when the parcel had <{min_coverage * 100}% coverage).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "bold_mask",
                "denoised_bold",
                "temporal_mask",
                "alff",  # may be Undefined
                "reho",
                "atlases",
                "atlas_files",
                "atlas_labels_files",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "coverage",
                "timeseries",
                "correlations",
                "correlations_exact",
                "parcellated_alff",
                "parcellated_reho",
            ],
        ),
        name="outputnode",
    )

    parcellate_data = pe.MapNode(
        NiftiParcellate(min_coverage=min_coverage),
        name="parcellate_data",
        iterfield=["atlas", "atlas_labels"],
        mem_gb=mem_gb["timeseries"],
    )
    workflow.connect([
        (inputnode, parcellate_data, [
            ("denoised_bold", "filtered_file"),
            ("bold_mask", "mask"),
            ("atlas_files", "atlas"),
            ("atlas_labels_files", "atlas_labels"),
        ]),
        (parcellate_data, outputnode, [
            ("coverage", "coverage"),
            ("timeseries", "timeseries"),
        ]),
    ])  # fmt:skip

    if config.workflow.output_correlations:
        functional_connectivity = pe.MapNode(
            TSVConnect(),
            name="functional_connectivity",
            iterfield=["timeseries"],
            mem_gb=mem_gb["timeseries"],
        )
        workflow.connect([
            (inputnode, functional_connectivity, [("temporal_mask", "temporal_mask")]),
            (parcellate_data, functional_connectivity, [("timeseries", "timeseries")]),
            (functional_connectivity, outputnode, [
                ("correlations", "correlations"),
                ("correlations_exact", "correlations_exact"),
            ]),
        ])  # fmt:skip

        connectivity_plot = pe.Node(
            ConnectPlot(),
            name="connectivity_plot",
            mem_gb=mem_gb["resampled"],
        )
        workflow.connect([
            (inputnode, connectivity_plot, [
                ("atlases", "atlases"),
                ("atlas_labels_files", "atlas_tsvs"),
            ]),
            (functional_connectivity, connectivity_plot, [("correlations", "correlations_tsv")]),
        ])  # fmt:skip

        ds_connectivity_plot = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="connectivityplot",
                datatype="figures",
            ),
            name="ds_connectivity_plot",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_connectivity_plot, [("name_source", "source_file")]),
            (connectivity_plot, ds_connectivity_plot, [("connectplot", "in_file")]),
        ])  # fmt:skip

    parcellate_reho = pe.MapNode(
        NiftiParcellate(min_coverage=min_coverage),
        name="parcellate_reho",
        iterfield=["atlas", "atlas_labels"],
        mem_gb=mem_gb["resampled"],
    )
    workflow.connect([
        (inputnode, parcellate_reho, [
            ("reho", "filtered_file"),
            ("bold_mask", "mask"),
            ("atlas_files", "atlas"),
            ("atlas_labels_files", "atlas_labels"),
        ]),
        (parcellate_reho, outputnode, [("timeseries", "parcellated_reho")]),
    ])  # fmt:skip

    if bandpass_filter:
        parcellate_alff = pe.MapNode(
            NiftiParcellate(min_coverage=min_coverage),
            name="parcellate_alff",
            iterfield=["atlas", "atlas_labels"],
            mem_gb=mem_gb["resampled"],
        )
        workflow.connect([
            (inputnode, parcellate_alff, [
                ("alff", "filtered_file"),
                ("bold_mask", "mask"),
                ("atlas_files", "atlas"),
                ("atlas_labels_files", "atlas_labels"),
            ]),
            (parcellate_alff, outputnode, [("timeseries", "parcellated_alff")]),
        ])  # fmt:skip

    return workflow


@fill_doc
def init_functional_connectivity_cifti_wf(mem_gb, exact_scans, name="connectivity_wf"):
    """Extract CIFTI time series.

    This will parcellate the CIFTI file using the selected atlases and compute functional
    connectivity between all regions for the selected atlases.
    It will also parcellate ReHo and ALFF maps if they are provided.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d import config
            from xcp_d.tests.tests import mock_config
            from xcp_d.workflows.connectivity import init_functional_connectivity_cifti_wf

            with mock_config():
                config.execution.atlases = ["Glasser", "Gordon"]

                wf = init_functional_connectivity_cifti_wf(
                    mem_gb={"resampled": 0.1, "timeseries": 1.0},
                    exact_scans=[30, 40],
                )

    Parameters
    ----------
    mem_gb : :obj:`dict`
        Dictionary of memory allocations.
    exact_scans : :obj:`list`
        List of exact scans to compute correlations for.
    %(name)s
        Default is "connectivity_wf".

    Inputs
    ------
    %(name_source)s
    denoised_bold
        Clean CIFTI after filtering and nuisance regression.
        The CIFTI file is in the same standard space as the atlases,
        so no transformations will be applied to the data before parcellation.
    %(temporal_mask)s
    alff
    reho
    %(atlases)s
    atlas_files
    atlas_labels_files

    Outputs
    -------
    %(coverage_ciftis)s
    %(timeseries_ciftis)s
    %(correlation_ciftis)s
    correlation_ciftis_exact
    %(coverage)s
    %(timeseries)s
    %(correlations)s
    correlations_exact
    parcellated_reho
    parcellated_alff
    """
    from xcp_d.interfaces.censoring import ReduceCifti
    from xcp_d.interfaces.connectivity import CiftiToTSV, ConnectPlot
    from xcp_d.interfaces.plotting import PlotCiftiParcellation
    from xcp_d.interfaces.workbench import CiftiCorrelation

    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir
    bandpass_filter = config.workflow.bandpass_filter
    min_coverage = config.workflow.min_coverage

    atlas_str = describe_atlases(config.execution.atlases)

    workflow.__desc__ = f"""
Processed functional timeseries were extracted from residual BOLD using
Connectome Workbench [@marcus2011informatics] for the following atlases: {atlas_str}.
Corresponding pair-wise functional connectivity between all regions was computed for each atlas,
which was operationalized as the Pearson's correlation of each parcel's unsmoothed timeseries with
the Connectome Workbench.
In cases of partial coverage, uncovered vertices (values of all zeros or NaNs) were either
ignored (when the parcel had >{min_coverage * 100}% coverage)
or were set to zero (when the parcel had <{min_coverage * 100}% coverage).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "denoised_bold",
                "temporal_mask",
                "alff",  # may be Undefined
                "reho",
                "atlases",
                "atlas_files",
                "atlas_labels_files",
                # for plotting, if the anatomical workflow is enabled
                "lh_midthickness",
                "rh_midthickness",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "coverage_ciftis",
                "timeseries_ciftis",
                "correlation_ciftis",
                "correlation_ciftis_exact",
                "coverage",
                "timeseries",
                "correlations",
                "correlations_exact",
                "parcellated_alff",
                "parcellated_reho",
            ],
        ),
        name="outputnode",
    )

    parcellate_bold_wf = init_parcellate_cifti_wf(
        mem_gb=mem_gb,
        compute_mask=True,
        name="parcellate_bold_wf",
    )
    workflow.connect([
        (inputnode, parcellate_bold_wf, [
            ("denoised_bold", "inputnode.in_file"),
            ("atlas_files", "inputnode.atlas_files"),
            ("atlas_labels_files", "inputnode.atlas_labels_files"),
        ]),
        (parcellate_bold_wf, outputnode, [
            ("outputnode.parcellated_cifti", "timeseries_ciftis"),
            ("outputnode.parcellated_tsv", "timeseries"),
            ("outputnode.coverage_cifti", "coverage_ciftis"),
            ("outputnode.coverage_tsv", "coverage"),
        ]),
    ])  # fmt:skip

    # Filter out subcortical atlases
    cortical_atlases = select_atlases(atlases=config.execution.atlases, subset="cortical")
    if cortical_atlases:
        plot_coverage = pe.Node(
            PlotCiftiParcellation(
                base_desc="coverage",
                cortical_atlases=cortical_atlases,
                vmin=0,
                vmax=1,
            ),
            name="plot_coverage",
            mem_gb=mem_gb["resampled"],
        )
        workflow.connect([
            (inputnode, plot_coverage, [
                ("atlases", "labels"),
                ("lh_midthickness", "lh_underlay"),
                ("rh_midthickness", "rh_underlay"),
            ]),
            (parcellate_bold_wf, plot_coverage, [("outputnode.coverage_cifti", "in_files")]),
        ])  # fmt:skip

        ds_plot_coverage = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                datatype="figures",
            ),
            name="ds_plot_coverage",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_plot_coverage, [("name_source", "source_file")]),
            (plot_coverage, ds_plot_coverage, [
                ("out_file", "in_file"),
                ("desc", "desc"),
            ]),
        ])  # fmt:skip

    # Reduce the CIFTI before calculating correlations
    parcellated_bold_buffer = pe.MapNode(
        niu.IdentityInterface(fields=["parcellated_cifti"]),
        name="parcellated_bold_buffer",
        iterfield=["parcellated_cifti"],
    )
    if config.workflow.output_interpolated:
        # If we want interpolated time series, the parcellated CIFTI will have interpolated values,
        # but the correlation matrices should only include low-motion volumes.
        remove_outliers = pe.MapNode(
            ReduceCifti(column="framewise_displacement"),
            name="remove_outliers",
            iterfield=["in_file"],
        )
        workflow.connect([
            (inputnode, remove_outliers, [("temporal_mask", "temporal_mask")]),
            (parcellate_bold_wf, remove_outliers, [("outputnode.parcellated_cifti", "in_file")]),
            (remove_outliers, parcellated_bold_buffer, [("out_file", "parcellated_cifti")]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (parcellate_bold_wf, parcellated_bold_buffer, [
                ("outputnode.parcellated_cifti", "parcellated_cifti"),
            ]),
        ])  # fmt:skip

    if config.workflow.output_correlations:
        # Correlate the parcellated data
        correlate_bold = pe.MapNode(
            CiftiCorrelation(),
            name="correlate_bold",
            iterfield=["in_file"],
        )
        workflow.connect([
            (parcellated_bold_buffer, correlate_bold, [("parcellated_cifti", "in_file")]),
            (correlate_bold, outputnode, [("out_file", "correlation_ciftis")]),
        ])  # fmt:skip

        # Convert correlation pconn file to TSV
        dconn_to_tsv = pe.MapNode(
            CiftiToTSV(),
            name="dconn_to_tsv",
            iterfield=["in_file", "atlas_labels"],
        )
        workflow.connect([
            (inputnode, dconn_to_tsv, [("atlas_labels_files", "atlas_labels")]),
            (correlate_bold, dconn_to_tsv, [("out_file", "in_file")]),
            (dconn_to_tsv, outputnode, [("out_file", "correlations")]),
        ])  # fmt:skip

        # Plot up to four connectivity matrices
        connectivity_plot = pe.Node(
            ConnectPlot(),
            name="connectivity_plot",
            mem_gb=mem_gb["resampled"],
        )
        workflow.connect([
            (inputnode, connectivity_plot, [
                ("atlases", "atlases"),
                ("atlas_labels_files", "atlas_tsvs"),
            ]),
            (dconn_to_tsv, connectivity_plot, [("out_file", "correlations_tsv")]),
        ])  # fmt:skip

        ds_connectivity_plot = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="connectivityplot",
                datatype="figures",
            ),
            name="ds_connectivity_plot",
            run_without_submitting=False,
            mem_gb=0.1,
        )
        workflow.connect([
            (inputnode, ds_connectivity_plot, [("name_source", "source_file")]),
            (connectivity_plot, ds_connectivity_plot, [("connectplot", "in_file")]),
        ])  # fmt:skip

    # Perform exact-time correlations
    if exact_scans:
        collect_exact_ciftis = pe.Node(
            niu.Merge(len(exact_scans)),
            name="collect_exact_ciftis",
        )
        workflow.connect([
            (collect_exact_ciftis, outputnode, [("out", "correlation_ciftis_exact")]),
        ])  # fmt:skip

        collect_exact_tsvs = pe.Node(
            niu.Merge(len(exact_scans)),
            name="collect_exact_tsvs",
        )
        workflow.connect([(collect_exact_tsvs, outputnode, [("out", "correlations_exact")])])

        for i_exact_scan, exact_scan in enumerate(exact_scans):
            reduce_exact_bold = pe.MapNode(
                ReduceCifti(column=f"exact_{exact_scan}"),
                name=f"reduce_bold_{exact_scan}volumes",
                iterfield=["in_file"],
            )
            workflow.connect([
                (inputnode, reduce_exact_bold, [("temporal_mask", "temporal_mask")]),
                (parcellated_bold_buffer, reduce_exact_bold, [("parcellated_cifti", "in_file")]),
            ])  # fmt:skip

            # Correlate the parcellated data
            correlate_exact_bold = pe.MapNode(
                CiftiCorrelation(),
                name=f"correlate_bold_{exact_scan}volumes",
                iterfield=["in_file"],
            )
            workflow.connect([
                (reduce_exact_bold, correlate_exact_bold, [("out_file", "in_file")]),
                (correlate_exact_bold, collect_exact_ciftis, [
                    ("out_file", f"in{i_exact_scan + 1}"),
                ]),
            ])  # fmt:skip

            # Convert correlation pconn file to TSV
            exact_dconn_to_tsv = pe.MapNode(
                CiftiToTSV(),
                name=f"dconn_to_tsv_{exact_scan}volumes",
                iterfield=["in_file", "atlas_labels"],
            )
            workflow.connect([
                (inputnode, exact_dconn_to_tsv, [("atlas_labels_files", "atlas_labels")]),
                (correlate_exact_bold, exact_dconn_to_tsv, [("out_file", "in_file")]),
                (exact_dconn_to_tsv, collect_exact_tsvs, [("out_file", f"in{i_exact_scan + 1}")]),
            ])  # fmt:skip

    parcellate_reho_wf = init_parcellate_cifti_wf(
        mem_gb=mem_gb,
        compute_mask=False,
        name="parcellate_reho_wf",
    )
    workflow.connect([
        (inputnode, parcellate_reho_wf, [
            ("reho", "inputnode.in_file"),
            ("atlas_files", "inputnode.atlas_files"),
            ("atlas_labels_files", "inputnode.atlas_labels_files"),
        ]),
        (parcellate_bold_wf, parcellate_reho_wf, [
            ("outputnode.vertexwise_coverage", "inputnode.vertexwise_coverage"),
            ("outputnode.coverage_cifti", "inputnode.coverage_cifti"),
        ]),
        (parcellate_reho_wf, outputnode, [("outputnode.parcellated_tsv", "parcellated_reho")]),
    ])  # fmt:skip

    if cortical_atlases:
        plot_parcellated_reho = pe.Node(
            PlotCiftiParcellation(
                base_desc="reho",
                cortical_atlases=cortical_atlases,
            ),
            name="plot_parcellated_reho",
            mem_gb=mem_gb["resampled"],
        )
        workflow.connect([
            (inputnode, plot_parcellated_reho, [
                ("atlases", "labels"),
                ("lh_midthickness", "lh_underlay"),
                ("rh_midthickness", "rh_underlay"),
            ]),
            (parcellate_reho_wf, plot_parcellated_reho, [
                ("outputnode.parcellated_cifti", "in_files"),
            ]),
        ])  # fmt:skip

        ds_plot_reho = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                datatype="figures",
            ),
            name="ds_plot_reho",
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_plot_reho, [("name_source", "source_file")]),
            (plot_parcellated_reho, ds_plot_reho, [
                ("desc", "desc"),
                ("out_file", "in_file"),
            ]),
        ])  # fmt:skip

    if bandpass_filter:
        parcellate_alff_wf = init_parcellate_cifti_wf(
            mem_gb=mem_gb,
            compute_mask=False,
            name="parcellate_alff_wf",
        )
        workflow.connect([
            (inputnode, parcellate_alff_wf, [
                ("alff", "inputnode.in_file"),
                ("atlas_files", "inputnode.atlas_files"),
                ("atlas_labels_files", "inputnode.atlas_labels_files"),
            ]),
            (parcellate_bold_wf, parcellate_alff_wf, [
                ("outputnode.vertexwise_coverage", "inputnode.vertexwise_coverage"),
                ("outputnode.coverage_cifti", "inputnode.coverage_cifti"),
            ]),
            (parcellate_alff_wf, outputnode, [("outputnode.parcellated_tsv", "parcellated_alff")]),
        ])  # fmt:skip

        if cortical_atlases:
            plot_parcellated_alff = pe.Node(
                PlotCiftiParcellation(
                    base_desc="alff",
                    cortical_atlases=cortical_atlases,
                ),
                name="plot_parcellated_alff",
                mem_gb=mem_gb["resampled"],
            )
            workflow.connect([
                (inputnode, plot_parcellated_alff, [
                    ("atlases", "labels"),
                    ("lh_midthickness", "lh_underlay"),
                    ("rh_midthickness", "rh_underlay"),
                ]),
                (parcellate_alff_wf, plot_parcellated_alff, [
                    ("outputnode.parcellated_cifti", "in_files"),
                ]),
            ])  # fmt:skip

            ds_plot_alff = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    datatype="figures",
                ),
                name="ds_plot_alff",
                run_without_submitting=False,
            )
            workflow.connect([
                (inputnode, ds_plot_alff, [("name_source", "source_file")]),
                (plot_parcellated_alff, ds_plot_alff, [
                    ("out_file", "in_file"),
                    ("desc", "desc"),
                ]),
            ])  # fmt:skip

    return workflow
