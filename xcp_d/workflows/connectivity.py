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
from xcp_d.interfaces.connectivity import (
    CiftiConnect,
    CiftiParcellate,
    ConnectPlot,
    NiftiParcellate,
    TSVConnect,
)
from xcp_d.interfaces.nilearn import IndexImage
from xcp_d.interfaces.workbench import (
    CiftiChangeMapping,
    CiftiCreateDenseFromTemplate,
    CiftiParcellateWorkbench,
)
from xcp_d.utils.atlas import (
    copy_atlas,
    get_atlas_cifti,
    get_atlas_nifti,
    select_atlases,
)
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

            from xcp_d.workflows.connectivity import init_load_atlases_wf

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
    parcellated_atlas_files
    """
    workflow = Workflow(name=name)
    atlases = config.workflow.atlases
    output_dir = config.execution.xcp_d_dir
    cifti = config.workflow.cifti
    mem_gb = config.nipype.memory_gb
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
                "parcellated_atlas_files",  # only used for CIFTIs
            ],
        ),
        name="outputnode",
    )

    # get atlases via pkgrf
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

        # fmt:off
        workflow.connect([
            (inputnode, get_transforms_to_bold_space, [("name_source", "bold_file")]),
        ])
        # fmt:on

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
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (grab_first_volume, warp_atlases_to_bold_space, [("out_file", "reference_image")]),
            (atlas_file_grabber, warp_atlases_to_bold_space, [("atlas_file", "input_image")]),
            (get_transforms_to_bold_space, warp_atlases_to_bold_space, [
                ("transformfile", "transforms"),
            ]),
            (warp_atlases_to_bold_space, atlas_buffer, [("output_image", "atlas_file")]),
        ])
        # fmt:on

    else:
        # Add empty vertices to atlas for locations in data, but not in atlas
        # (e.g., subcortical regions for cortex-only atlases)
        resample_atlas_to_data = pe.MapNode(
            CiftiCreateDenseFromTemplate(),
            name="resample_atlas_to_data",
            n_procs=omp_nthreads,
            iterfield=["label"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, resample_atlas_to_data, [("bold_file", "template_cifti")]),
            (atlas_file_grabber, resample_atlas_to_data, [("atlas_file", "label")]),
            (resample_atlas_to_data, atlas_buffer, [("cifti_out", "atlas_file")]),
        ])
        # fmt:on

        # Change the atlas to a scalar file.
        convert_to_dscalar = pe.MapNode(
            CiftiChangeMapping(
                direction="ROW",
                scalar=True,
                cifti_out="atlas.dscalar.nii",
            ),
            name="convert_to_dscalar",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
            iterfield=["data_cifti"],
        )
        # fmt:off
        workflow.connect([
            (resample_atlas_to_data, convert_to_dscalar, [("cifti_out", "data_cifti")]),
        ])
        # fmt:on

        # Convert atlas from dlabel to pscalar format.
        # The pscalar version of the atlas is later used for its ParcelAxis.
        parcellate_atlas = pe.MapNode(
            CiftiParcellateWorkbench(
                direction="COLUMN",
                only_numeric=True,
                out_file="parcellated_atlas.pscalar.nii",
            ),
            name="parcellate_atlas",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
            iterfield=["in_file", "atlas_label"],
        )

        # fmt:off
        workflow.connect([
            (atlas_file_grabber, parcellate_atlas, [("atlas_file", "atlas_label")]),
            (convert_to_dscalar, parcellate_atlas, [("cifti_out", "in_file")]),
            (parcellate_atlas, outputnode, [("out_file", "parcellated_atlas_files")]),
        ])
        # fmt:on

    ds_atlas = pe.MapNode(
        Function(
            function=copy_atlas,
            input_names=[
                "name_source",
                "in_file",
                "output_dir",
                "atlas",
            ],
            output_names=["out_file"],
        ),
        name="ds_atlas",
        iterfield=["in_file", "atlas"],
        run_without_submitting=True,
    )
    ds_atlas.inputs.output_dir = output_dir
    ds_atlas.inputs.atlas = atlases

    # fmt:off
    workflow.connect([
        (inputnode, ds_atlas, [("name_source", "name_source")]),
        (atlas_buffer, ds_atlas, [("atlas_file", "in_file")]),
        (ds_atlas, outputnode, [("out_file", "atlas_files")]),
    ])
    # fmt:on

    ds_atlas_labels_file = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
            dismiss_entities=[
                "datatype",
                "subject",
                "session",
                "task",
                "run",
                "desc",
                "space",
                "res",
                "den",
                "cohort",
            ],
            allowed_entities=["atlas"],
            suffix="dseg",
            extension=".tsv",
        ),
        name="ds_atlas_labels_file",
        iterfield=["atlas", "in_file"],
        run_without_submitting=True,
    )
    ds_atlas_labels_file.inputs.atlas = atlases

    # fmt:off
    workflow.connect([
        (inputnode, ds_atlas_labels_file, [("name_source", "source_file")]),
        (atlas_file_grabber, ds_atlas_labels_file, [("atlas_labels_file", "in_file")]),
        (ds_atlas_labels_file, outputnode, [("out_file", "atlas_labels_files")]),
    ])
    # fmt:on

    ds_atlas_metadata = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
            dismiss_entities=[
                "datatype",
                "subject",
                "session",
                "task",
                "run",
                "desc",
                "space",
                "res",
                "den",
                "cohort",
            ],
            allowed_entities=["atlas"],
            suffix="dseg",
            extension=".json",
        ),
        name="ds_atlas_metadata",
        iterfield=["atlas", "in_file"],
        run_without_submitting=True,
    )
    ds_atlas_metadata.inputs.atlas = atlases

    # fmt:off
    workflow.connect([
        (inputnode, ds_atlas_metadata, [("name_source", "source_file")]),
        (atlas_file_grabber, ds_atlas_metadata, [("atlas_metadata_file", "in_file")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_parcellate_surfaces_wf(files_to_parcellate, name="parcellate_surfaces_wf"):
    """Parcellate surface files and write them out to the output directory.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.connectivity import init_parcellate_surfaces_wf

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
    atlases = config.workflow.atlases
    min_coverage = config.workflow.min_coverage
    mem_gb = config.nipype.memory_gb
    omp_nthreads = config.nipype.omp_nthreads

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

    # Get CIFTI atlases via pkgrf
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
        resample_atlas_to_surface = pe.MapNode(
            CiftiCreateDenseFromTemplate(),
            name=f"resample_atlas_to_{file_to_parcellate}",
            n_procs=omp_nthreads,
            iterfield=["label"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, resample_atlas_to_surface, [(file_to_parcellate, "template_cifti")]),
            (atlas_file_grabber, resample_atlas_to_surface, [("atlas_file", "label")]),
        ])
        # fmt:on

        parcellate_atlas = pe.MapNode(
            CiftiParcellateWorkbench(
                direction="COLUMN",
                only_numeric=True,
                out_file="parcellated_atlas.pscalar.nii",
            ),
            name=f"parcellate_atlas_for_{file_to_parcellate}",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
            iterfield=["atlas_label"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, parcellate_atlas, [(file_to_parcellate, "in_file")]),
            (resample_atlas_to_surface, parcellate_atlas, [("cifti_out", "atlas_label")]),
        ])
        # fmt:on

        # Parcellate the ciftis
        parcellate_surface = pe.MapNode(
            CiftiParcellate(min_coverage=min_coverage),
            mem_gb=mem_gb,
            name=f"parcellate_{file_to_parcellate}",
            n_procs=omp_nthreads,
            iterfield=["atlas_labels", "atlas", "parcellated_atlas"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, parcellate_surface, [(file_to_parcellate, "data_file")]),
            (resample_atlas_to_surface, parcellate_surface, [("cifti_out", "atlas")]),
            (atlas_file_grabber, parcellate_surface, [("atlas_labels_file", "atlas_labels")]),
            (parcellate_atlas, parcellate_surface, [("out_file", "parcellated_atlas")]),
        ])
        # fmt:on

        # Write out the parcellated files
        ds_parcellated_surface = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                dismiss_entities=["hemi", "desc"],
                desc=SURF_DESCS[file_to_parcellate],
                suffix="morph",
                extension=".tsv",
            ),
            name=f"ds_parcellated_{file_to_parcellate}",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["atlas", "in_file"],
        )
        ds_parcellated_surface.inputs.atlas = selected_atlases

        # fmt:off
        workflow.connect([
            (inputnode, ds_parcellated_surface, [(file_to_parcellate, "source_file")]),
            (parcellate_surface, ds_parcellated_surface, [("timeseries", "in_file")]),
        ])
        # fmt:on

    return workflow


@fill_doc
def init_functional_connectivity_nifti_wf(name="connectivity_wf"):
    """Extract BOLD time series and compute functional connectivity.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.connectivity import init_functional_connectivity_nifti_wf

            wf = init_functional_connectivity_nifti_wf()

    Parameters
    ----------
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
    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir
    bandpass_filter = config.workflow.bandpass_filter
    min_coverage = config.workflow.min_coverage
    mem_gb = config.nipype.memory_gb

    workflow.__desc__ = f"""
Processed functional timeseries were extracted from the residual BOLD signal
with *Nilearn's* *NiftiLabelsMasker* for the following atlases:
the Schaefer Supplemented with Subcortical Structures (4S) atlas
[@Schaefer_2017,@pauli2018high,@king2019functional,@najdenovska2018vivo,@glasser2013minimal] at
10 different resolutions (156, 256, 356, 456, 556, 656, 756, 856, 956, and 1056 parcels),
the Glasser atlas [@Glasser_2016], the Gordon atlas [@Gordon_2014],
the Tian subcortical atlas [@tian2020topographic], and the HCP CIFTI subcortical atlas
[@glasser2013minimal].
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
        mem_gb=(3 * mem_gb) if mem_gb is not None else mem_gb,
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
        mem_gb=(3 * mem_gb) if mem_gb is not None else mem_gb,
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
        mem_gb=mem_gb,
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
            mem_gb=mem_gb,
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
    connectivity_plot = pe.Node(
        ConnectPlot(),
        name="connectivity_plot",
        mem_gb=mem_gb,
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
def init_functional_connectivity_cifti_wf(name="connectivity_wf"):
    """Extract CIFTI time series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.connectivity import init_functional_connectivity_cifti_wf
            wf = init_functional_connectivity_cifti_wf()

    Parameters
    ----------
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
    parcellated_atlas_files

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
    workflow = Workflow(name=name)

    output_dir = config.execution.xcp_d_dir
    bandpass_filter = config.workflow.bandpass_filter
    min_coverage = config.workflow.min_coverage
    mem_gb = config.nipype.memory_gb
    omp_nthreads = config.nipype.omp_nthreads

    workflow.__desc__ = f"""
Processed functional timeseries were extracted from residual BOLD using
Connectome Workbench [@hcppipelines] for the following atlases:
the Schaefer Supplemented with Subcortical Structures (4S) atlas
[@Schaefer_2017,@pauli2018high,@king2019functional,@najdenovska2018vivo,@glasser2013minimal] at
10 different resolutions (156, 256, 356, 456, 556, 656, 756, 856, 956, and 1056 parcels),
the Glasser atlas [@Glasser_2016], the Gordon atlas [@Gordon_2014],
the Tian subcortical atlas [@tian2020topographic], and the HCP CIFTI subcortical atlas
[@glasser2013minimal].
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
                "parcellated_atlas_files",
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

    parcellate_data = pe.MapNode(
        CiftiParcellate(min_coverage=min_coverage),
        name="parcellate_data",
        iterfield=["atlas", "atlas_labels", "parcellated_atlas"],
        mem_gb=(3 * mem_gb) if mem_gb is not None else mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, parcellate_data, [
            ("denoised_bold", "data_file"),
            ("atlas_files", "atlas"),
            ("atlas_labels_files", "atlas_labels"),
            ("parcellated_atlas_files", "parcellated_atlas"),
        ]),
        (parcellate_data, outputnode, [
            ("coverage", "coverage"),
            ("coverage_ciftis", "coverage_ciftis"),
            ("timeseries", "timeseries"),
            ("timeseries_ciftis", "timeseries_ciftis"),
        ]),
    ])
    # fmt:on

    functional_connectivity = pe.MapNode(
        CiftiConnect(),
        name="functional_connectivity",
        iterfield=["timeseries", "parcellated_atlas"],
        mem_gb=(3 * mem_gb) if mem_gb is not None else mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, functional_connectivity, [
            ("temporal_mask", "temporal_mask"),
            ("denoised_bold", "data_file"),
            ("parcellated_atlas_files", "parcellated_atlas"),
        ]),
        (parcellate_data, functional_connectivity, [("timeseries", "timeseries")]),
        (functional_connectivity, outputnode, [
            ("correlations", "correlations"),
            ("correlation_ciftis", "correlation_ciftis"),
            ("correlations_exact", "correlations_exact"),
            ("correlation_ciftis_exact", "correlation_ciftis_exact"),
        ]),
    ])
    # fmt:on

    parcellate_reho = pe.MapNode(
        CiftiParcellate(min_coverage=min_coverage),
        mem_gb=mem_gb,
        name="parcellate_reho",
        n_procs=omp_nthreads,
        iterfield=["atlas_labels", "atlas", "parcellated_atlas"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, parcellate_reho, [
            ("reho", "data_file"),
            ("atlas_files", "atlas"),
            ("atlas_labels_files", "atlas_labels"),
            ("parcellated_atlas_files", "parcellated_atlas"),
        ]),
        (parcellate_reho, outputnode, [("timeseries", "parcellated_reho")]),
    ])
    # fmt:on

    if bandpass_filter:
        parcellate_alff = pe.MapNode(
            CiftiParcellate(min_coverage=min_coverage),
            mem_gb=mem_gb,
            name="parcellate_alff",
            n_procs=omp_nthreads,
            iterfield=["atlas_labels", "atlas", "parcellated_atlas"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, parcellate_alff, [
                ("alff", "data_file"),
                ("atlas_files", "atlas"),
                ("atlas_labels_files", "atlas_labels"),
                ("parcellated_atlas_files", "parcellated_atlas"),
            ]),
            (parcellate_alff, outputnode, [("timeseries", "parcellated_alff")]),
        ])
        # fmt:on

    # Create a node to plot the matrixes
    connectivity_plot = pe.Node(
        ConnectPlot(),
        name="connectivity_plot",
        mem_gb=mem_gb,
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
