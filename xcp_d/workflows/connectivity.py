# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for extracting time series and computing functional connectivity."""
from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import IndexImage
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_nifti, select_atlases
from xcp_d.utils.boilerplate import describe_atlases
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.utils import get_std2bold_xfms

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_load_atlases_wf(name="load_atlases_wf"):
    """Load atlases and warp them to the same space as the BOLD file.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.connectivity import init_load_atlases_wf

            with mock_config():
                wf = init_load_atlases_wf()

    Parameters
    ----------
    %(name)s
        Default is "load_atlases_wf".

    Inputs
    ------
    %(name_source)s
    bold_file

    Outputs
    -------
    atlas_files
    atlas_labels_files
    """
    from xcp_d.interfaces.bids import CopyAtlas

    workflow = Workflow(name=name)
    atlases = config.execution.atlases
    output_dir = config.execution.xcp_d_dir
    cifti = config.workflow.cifti
    omp_nthreads = config.nipype.omp_nthreads

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "bold_file",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "atlas_files",
                "atlas_labels_files",
            ],
        ),
        name="outputnode",
    )

    # get atlases via load_data
    atlas_file_grabber = pe.MapNode(
        Function(
            input_names=["atlas"],
            output_names=["atlas_file", "atlas_labels_file", "atlas_metadata_file"],
            function=get_atlas_cifti if cifti else get_atlas_nifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas"],
    )
    atlas_file_grabber.inputs.atlas = atlases

    atlas_buffer = pe.Node(niu.IdentityInterface(fields=["atlas_file"]), name="atlas_buffer")

    if not cifti:
        get_transforms_to_bold_space = pe.Node(
            Function(
                input_names=["bold_file"],
                output_names=["transformfile"],
                function=get_std2bold_xfms,
            ),
            name="get_transforms_to_bold_space",
        )

        workflow.connect([
            (inputnode, get_transforms_to_bold_space, [("name_source", "bold_file")]),
        ])  # fmt:skip

        # ApplyTransforms needs a 3D image for the reference image.
        grab_first_volume = pe.Node(
            IndexImage(index=0),
            name="grab_first_volume",
        )

        workflow.connect([(inputnode, grab_first_volume, [("bold_file", "in_file")])])

        # Using the generated transforms, apply them to get everything in the correct MNI form
        warp_atlases_to_bold_space = pe.MapNode(
            ApplyTransforms(
                interpolation="GenericLabel",
                input_image_type=3,
                dimension=3,
            ),
            name="warp_atlases_to_bold_space",
            iterfield=["input_image"],
            mem_gb=2,
            n_procs=omp_nthreads,
        )

        workflow.connect([
            (grab_first_volume, warp_atlases_to_bold_space, [("out_file", "reference_image")]),
            (atlas_file_grabber, warp_atlases_to_bold_space, [("atlas_file", "input_image")]),
            (get_transforms_to_bold_space, warp_atlases_to_bold_space, [
                ("transformfile", "transforms"),
            ]),
            (warp_atlases_to_bold_space, atlas_buffer, [("output_image", "atlas_file")]),
        ])  # fmt:skip

    else:
        workflow.connect([(atlas_file_grabber, atlas_buffer, [("atlas_file", "atlas_file")])])

    ds_atlas = pe.MapNode(
        CopyAtlas(output_dir=output_dir),
        name="ds_atlas",
        iterfield=["in_file", "atlas"],
        run_without_submitting=True,
    )
    ds_atlas.inputs.atlas = atlases

    workflow.connect([
        (inputnode, ds_atlas, [("name_source", "name_source")]),
        (atlas_buffer, ds_atlas, [("atlas_file", "in_file")]),
        (ds_atlas, outputnode, [("out_file", "atlas_files")]),
    ])  # fmt:skip

    ds_atlas_labels_file = pe.MapNode(
        CopyAtlas(output_dir=output_dir),
        name="ds_atlas_labels_file",
        iterfield=["in_file", "atlas"],
        run_without_submitting=True,
    )
    ds_atlas_labels_file.inputs.atlas = atlases

    workflow.connect([
        (inputnode, ds_atlas_labels_file, [("name_source", "name_source")]),
        (atlas_file_grabber, ds_atlas_labels_file, [("atlas_labels_file", "in_file")]),
        (ds_atlas_labels_file, outputnode, [("out_file", "atlas_labels_files")]),
    ])  # fmt:skip

    ds_atlas_metadata = pe.MapNode(
        CopyAtlas(output_dir=output_dir),
        name="ds_atlas_metadata",
        iterfield=["in_file", "atlas"],
        run_without_submitting=True,
    )
    ds_atlas_metadata.inputs.atlas = atlases

    workflow.connect([
        (inputnode, ds_atlas_metadata, [("name_source", "name_source")]),
        (atlas_file_grabber, ds_atlas_metadata, [("atlas_metadata_file", "in_file")]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_parcellate_surfaces_wf(files_to_parcellate, name="parcellate_surfaces_wf"):
    """Parcellate surface files and write them out to the output directory.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.connectivity import init_parcellate_surfaces_wf

            with mock_config():
                wf = init_parcellate_surfaces_wf(
                    files_to_parcellate=["sulcal_depth", "sulcal_curv", "cortical_thickness"],
                    name="parcellate_surfaces_wf",
                )

    Parameters
    ----------
    files_to_parcellate : :obj:`list` of :obj:`str`
        List of surface file types to parcellate
        (e.g., "sulcal_depth", "sulcal_curv", "cortical_thickness").
    %(name)s

    Inputs
    ------
    sulcal_depth
    sulcal_curv
    cortical_thickness
    cortical_thickness_corr
    myelin
    myelin_smoothed
    """
    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir
    atlases = config.execution.atlases

    SURF_DESCS = {
        "sulcal_depth": "sulc",
        "sulcal_curv": "curv",
        "cortical_thickness": "thickness",
        "cortical_thickness_corr": "thicknessCorrected",
        "myelin": "myelin",
        "myelin_smoothed": "myelinSmoothed",
    }

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
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

    selected_atlases = select_atlases(atlases=atlases, subset="cortical")

    if not selected_atlases:
        LOGGER.warning(
            "No cortical atlases have been selected, so surface metrics will not be parcellated."
        )
        # If no cortical atlases are selected, inputnode could go unconnected, so add explicitly.
        workflow.add_nodes([inputnode])

        return workflow

    # Get CIFTI atlases via load_data
    atlas_file_grabber = pe.MapNode(
        Function(
            input_names=["atlas"],
            output_names=["atlas_file", "atlas_labels_file", "atlas_metadata_file"],
            function=get_atlas_cifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas"],
    )
    atlas_file_grabber.inputs.atlas = selected_atlases

    for file_to_parcellate in files_to_parcellate:
        parcellate_surface_wf = init_parcellate_cifti_wf(
            compute_mask=True,
            name=f"parcellate_{file_to_parcellate}_wf",
        )
        workflow.connect([
            (inputnode, parcellate_surface_wf, [(file_to_parcellate, "inputnode.in_file")]),
            (atlas_file_grabber, parcellate_surface_wf, [
                ("atlas_file", "inputnode.atlas_files"),
                ("atlas_labels_file", "inputnode.atlas_labels_files"),
            ]),
        ])  # fmt:skip

        # Write out the parcellated files
        ds_parcellated_surface = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["hemi", "desc", "den", "res"],
                desc=SURF_DESCS[file_to_parcellate],
                statistic="mean",
                suffix="morph",
                extension=".tsv",
            ),
            name=f"ds_parcellated_{file_to_parcellate}",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["segmentation", "in_file"],
        )
        ds_parcellated_surface.inputs.segmentation = selected_atlases

        workflow.connect([
            (inputnode, ds_parcellated_surface, [(file_to_parcellate, "source_file")]),
            (parcellate_surface_wf, ds_parcellated_surface, [
                ("outputnode.parcellated_tsv", "in_file"),
            ]),
        ])  # fmt:skip

    return workflow


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
                wf = init_functional_connectivity_nifti_wf()

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

    # fmt:off
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
    ])
    # fmt:on

    functional_connectivity = pe.MapNode(
        TSVConnect(),
        name="functional_connectivity",
        iterfield=["timeseries"],
        mem_gb=mem_gb["timeseries"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, functional_connectivity, [("temporal_mask", "temporal_mask")]),
        (parcellate_data, functional_connectivity, [("timeseries", "timeseries")]),
        (functional_connectivity, outputnode, [
            ("correlations", "correlations"),
            ("correlations_exact", "correlations_exact"),
        ]),
    ])
    # fmt:on

    parcellate_reho = pe.MapNode(
        NiftiParcellate(min_coverage=min_coverage),
        name="parcellate_reho",
        iterfield=["atlas", "atlas_labels"],
        mem_gb=mem_gb["resampled"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, parcellate_reho, [
            ("reho", "filtered_file"),
            ("bold_mask", "mask"),
            ("atlas_files", "atlas"),
            ("atlas_labels_files", "atlas_labels"),
        ]),
        (parcellate_reho, outputnode, [("timeseries", "parcellated_reho")]),
    ])
    # fmt:on

    if bandpass_filter:
        parcellate_alff = pe.MapNode(
            NiftiParcellate(min_coverage=min_coverage),
            name="parcellate_alff",
            iterfield=["atlas", "atlas_labels"],
            mem_gb=mem_gb["resampled"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, parcellate_alff, [
                ("alff", "filtered_file"),
                ("bold_mask", "mask"),
                ("atlas_files", "atlas"),
                ("atlas_labels_files", "atlas_labels"),
            ]),
            (parcellate_alff, outputnode, [("timeseries", "parcellated_alff")]),
        ])
        # fmt:on

    # Create a node to plot the matrices
    if config.execution.atlases:
        connectivity_plot = pe.Node(
            ConnectPlot(),
            name="connectivity_plot",
            mem_gb=mem_gb["resampled"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, connectivity_plot, [
                ("atlases", "atlases"),
                ("atlas_labels_files", "atlas_tsvs"),
            ]),
            (functional_connectivity, connectivity_plot, [("correlations", "correlations_tsv")]),
        ])
        # fmt:on

        ds_connectivity_plot = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                desc="connectivityplot",
                datatype="figures",
            ),
            name="ds_connectivity_plot",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_connectivity_plot, [("name_source", "source_file")]),
            (connectivity_plot, ds_connectivity_plot, [("connectplot", "in_file")]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_functional_connectivity_cifti_wf(mem_gb, name="connectivity_wf"):
    """Extract CIFTI time series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.connectivity import init_functional_connectivity_cifti_wf

            with mock_config():
                wf = init_functional_connectivity_cifti_wf()

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
    from xcp_d.interfaces.connectivity import CiftiToTSV, ConnectPlot
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

    # Correlate the parcellated data
    cifti_correlate = pe.MapNode(
        CiftiCorrelation(),
        name="cifti_correlate",
        iterfield=["in_file"],
    )
    workflow.connect([
        (parcellate_bold_wf, cifti_correlate, [("outputnode.parcellated_cifti", "in_file")]),
        (cifti_correlate, outputnode, [("out_file", "correlation_ciftis")]),
    ])  # fmt:skip

    # Convert correlation pconn file to TSV
    dconn_to_tsv = pe.MapNode(
        CiftiToTSV(),
        name="dconn_to_tsv",
        iterfield=["in_file", "atlas_labels"],
    )
    workflow.connect([
        (inputnode, dconn_to_tsv, [("atlas_labels_files", "atlas_labels")]),
        (cifti_correlate, dconn_to_tsv, [("out_file", "in_file")]),
        (dconn_to_tsv, outputnode, [("out_file", "correlations")]),
    ])  # fmt:skip

    # TODO: Add exact correlation calculation back in

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
        (parcellate_reho_wf, outputnode, [
            ("outputnode.parcellated_tsv", "parcellated_reho"),
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
            (parcellate_alff_wf, outputnode, [
                ("outputnode.parcellated_tsv", "parcellated_alff"),
            ]),
        ])  # fmt:skip

    # Create a node to plot the matrices
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
        (dconn_to_tsv, connectivity_plot, [("correlations", "correlations_tsv")]),
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

    return workflow


def init_parcellate_cifti_wf(
    mem_gb,
    compute_mask=True,
    name="parcellate_cifti_wf",
):
    from xcp_d.interfaces.connectivity import CiftiToTSV
    from xcp_d.interfaces.workbench import CiftiMath, CiftiParcellateWorkbench
    from xcp_d.utils.utils import create_cifti_mask

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "in_file",
                "atlas_files",
                "atlas_labels_files",
                "vertexwise_coverage",
                "coverage_cifti",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "parcellated_cifti",
                "parcellated_tsv",
                "vertexwise_coverage",
                "coverage_cifti",
                "coverage_tsv",
            ],
        ),
        name="outputnode",
    )

    # Replace vertices with all zeros with NaNs using Python.
    coverage_buffer = pe.Node(
        niu.IdentityInterface(fields=["vertexwise_coverage", "coverage_cifti"]),
        name="coverage_buffer",
    )
    if compute_mask:
        # Write out a vertex-wise binary coverage map using Python.
        vertexwise_coverage = pe.Node(
            niu.Function(
                input_names=["data_file"],
                output_names=["mask_file"],
                function=create_cifti_mask,
            ),
            name="vertexwise_coverage",
        )
        workflow.connect([
            (inputnode, vertexwise_coverage, [("in_file", "data_file")]),
            (vertexwise_coverage, coverage_buffer, [("mask_file", "vertexwise_coverage")]),
            (vertexwise_coverage, outputnode, [("mask_file", "vertexwise_coverage")]),
        ])  # fmt:skip

        parcellate_coverage = pe.MapNode(
            CiftiParcellateWorkbench(
                direction="COLUMN",
                only_numeric=True,
                out_file="parcellated_atlas.pscalar.nii",
            ),
            name="parcellate_coverage",
            iterfield=["atlas"],
        )
        workflow.connect([
            (inputnode, parcellate_coverage, [("atlas_files", "atlas_label")]),
            (coverage_buffer, parcellate_coverage, [("coverage", "in_file")]),
            (parcellate_coverage, coverage_buffer, [("out_file", "coverage_cifti")]),
            (parcellate_coverage, outputnode, [("out_file", "coverage_cifti")]),
        ])  # fmt:skip

        coverage_to_tsv = pe.MapNode(
            CiftiToTSV(),
            name="coverage_to_tsv",
            iterfield=["in_file", "atlas_labels"],
        )
        workflow.connect([
            (inputnode, coverage_to_tsv, [("atlas_labels_files", "atlas_labels")]),
            (parcellate_coverage, coverage_to_tsv, [("out_file", "in_file")]),
            (coverage_to_tsv, outputnode, [("out_file", "coverage_tsv")]),
        ])  # fmt:skip
    else:
        workflow.connect([
            (inputnode, coverage_buffer, [
                ("vertexwise_coverage", "vertexwise_coverage"),
                ("coverage_cifti", "coverage_cifti"),
            ]),
        ])  # fmt:skip

    # Parcellate the data file using the vertex-wise coverage.
    parcellate_data = pe.MapNode(
        CiftiParcellateWorkbench(
            direction="COLUMN",
            only_numeric=True,
            out_file="parcellated_data.pscalar.nii",
        ),
        name="parcellate_data",
        iterfield=["atlas", "cifti_weights"],
        mem_gb=mem_gb["resampled"],
    )
    workflow.connect([
        (inputnode, parcellate_data, [
            ("denoised_bold", "in_file"),
            ("atlas_files", "atlas_label"),
        ]),
        (coverage_buffer, parcellate_data, [("vertexwise_coverage", "cifti_weights")]),
    ])  # fmt:skip

    # Threshold node coverage values based on coverage threshold.
    threshold_coverage = pe.MapNode(
        CiftiMath(expression=f"data > {config.workflow.min_coverage}"),
        name="threshold_coverage",
        iterfield=["data"],
        mem_gb=mem_gb["resampled"],
    )
    workflow.connect([(coverage_buffer, threshold_coverage, [("coverage_cifti", "data")])])

    # Mask out uncovered nodes from parcellated denoised data
    mask_parcellated_data = pe.MapNode(
        CiftiMath(expression="data * mask"),
        name="mask_parcellated_data",
        iterfield=["data", "mask"],
        mem_gb=mem_gb["resampled"],
    )
    workflow.connect([
        (parcellate_data, mask_parcellated_data, [("out_file", "data")]),
        (threshold_coverage, mask_parcellated_data, [("out_file", "mask")]),
        (mask_parcellated_data, outputnode, [("out_file", "parcellated_cifti")]),
    ])  # fmt:skip

    # Convert the parcellated CIFTI to a TSV file
    cifti_to_tsv = pe.MapNode(
        CiftiToTSV(),
        name="cifti_to_tsv",
        iterfield=["in_file", "atlas_labels"],
    )
    workflow.connect([
        (inputnode, cifti_to_tsv, [("atlas_labels_files", "atlas_labels")]),
        (mask_parcellated_data, cifti_to_tsv, [("out_file", "in_file")]),
        (cifti_to_tsv, outputnode, [("out_file", "parcellated_tsv")]),
    ])  # fmt:skip

    return workflow
