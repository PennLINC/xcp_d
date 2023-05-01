# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for extracting time series and computing functional connectivity."""
import nilearn as nl
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.connectivity import CiftiConnect, ConnectPlot, NiftiConnect
from xcp_d.interfaces.workbench import CiftiCreateDenseFromTemplate, CiftiParcellate
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_names, get_atlas_nifti
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.modified_data import cast_cifti_to_int16
from xcp_d.utils.utils import get_std2bold_xfms


@fill_doc
def init_load_atlases_wf(
    output_dir,
    cifti,
    mem_gb,
    omp_nthreads,
    name="load_atlases_wf",
):
    """Load atlases and warp them to the same space as the BOLD file."""
    workflow = Workflow(name=name)

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
                "atlas_names",
                "atlas_files",
                "atlas_labels_files",
                "parcellated_atlas_files",  # only used for CIFTIs
            ],
        ),
        name="outputnode",
    )

    atlas_name_grabber = pe.Node(
        Function(output_names=["atlas_names"], function=get_atlas_names),
        name="atlas_name_grabber",
    )

    workflow.connect([(atlas_name_grabber, outputnode, [("atlas_names", "atlas_names")])])

    # get atlases via pkgrf
    atlas_file_grabber = pe.MapNode(
        Function(
            input_names=["atlas_name"],
            output_names=["atlas_file", "atlas_labels_file"],
            function=get_atlas_cifti if cifti else get_atlas_nifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas_name"],
    )

    # fmt:off
    workflow.connect([
        (atlas_name_grabber, atlas_file_grabber, [("atlas_names", "atlas_name")]),
        (atlas_file_grabber, outputnode, [("atlas_labels_file", "atlas_labels_files")]),
    ])
    # fmt:on

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
            (inputnode, warp_atlases_to_bold_space, [("boldref", "reference_image")]),
            (atlas_file_grabber, warp_atlases_to_bold_space, [("atlas_file", "input_image")]),
            (get_transforms_to_bold_space, warp_atlases_to_bold_space, [
                ("transformfile", "transforms"),
            ]),
            (warp_atlases_to_bold_space, outputnode, [("output_image", "atlas_files")]),
            (warp_atlases_to_bold_space, atlas_buffer, [("output_image", "atlas_file")]),
        ])
        # fmt:on

    else:
        resample_atlas_to_data = pe.MapNode(
            CiftiCreateDenseFromTemplate(),
            name="resample_atlas_to_data",
            n_procs=omp_nthreads,
            iterfield=["label"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, resample_atlas_to_data, [("denoised_bold", "template_cifti")]),
            (atlas_file_grabber, resample_atlas_to_data, [("atlas_file", "label")]),
            (resample_atlas_to_data, outputnode, [("cifti_out", "atlas_files")]),
        ])
        # fmt:on

        parcellate_atlas = pe.MapNode(
            CiftiParcellate(
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
            (resample_atlas_to_data, parcellate_atlas, [("cifti_out", "in_file")]),
            (parcellate_atlas, outputnode, [("out_file", "parcellated_atlas_files")]),
        ])
        # fmt:on

        cast_atlas_to_int16 = pe.MapNode(
            Function(
                function=cast_cifti_to_int16,
                input_names=["in_file"],
                output_names=["out_file"],
            ),
            name="cast_atlas_to_int16",
            iterfield=["in_file"],
        )

        # fmt:off
        workflow.connect([
            (resample_atlas_to_data, cast_atlas_to_int16, [("cifti_out", "in_file")]),
            (cast_atlas_to_int16, atlas_buffer, [("out_file", "atlas_file")]),
        ])
        # fmt:on

    ds_atlas = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["datatype", "subject", "session", "task", "run", "desc"],
            suffix="dseg",
            extension=".nii.gz",
        ),
        name="ds_atlas",
        iterfield=["atlas", "in_file"],
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_atlas, [("name_source", "source_file")]),
        (atlas_name_grabber, ds_atlas, [("atlas_names", "atlas")]),
        (atlas_buffer, ds_atlas, [("atlas_file", "in_file")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_functional_connectivity_nifti_wf(
    output_dir,
    alff_available,
    min_coverage,
    mem_gb,
    name="connectivity_wf",
):
    """Extract BOLD time series and compute functional connectivity.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.connectivity import init_functional_connectivity_nifti_wf

            wf = init_functional_connectivity_nifti_wf(
                output_dir=".",
                alff_available=True,
                min_coverage=0.5,
                mem_gb=0.1,
                omp_nthreads=1,
                name="connectivity_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    alff_available
    %(min_coverage)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "connectivity_wf".

    Inputs
    ------
    %(name_source)s
    %(boldref)s
    denoised_bold
        clean bold after filtered out nuisscance and filtering
    alff
    reho

    Outputs
    -------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    %(timeseries)s
    %(correlations)s
    %(coverage)s
    parcellated_alff
    parcellated_reho
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = f"""
Processed functional timeseries were extracted from the residual BOLD signal
with *Nilearn's* [version {nl.__version__}, @abraham2014machine] *NiftiLabelsMasker* for the
following atlases:
the Schaefer 17-network 100, 200, 300, 400, 500, 600, 700, 800, 900, and 1000 parcel
atlas [@Schaefer_2017], the Glasser atlas [@Glasser_2016],
the Gordon atlas [@Gordon_2014], and the Tian subcortical artlas [@tian2020topographic].
Corresponding pair-wise functional connectivity between all regions was computed for each atlas,
which was operationalized as the Pearson's correlation of each parcel's unsmoothed timeseries.
In cases of partial coverage, uncovered voxels (values of all zeros or NaNs) were either
ignored, when the parcel had >{min_coverage * 100}% coverage,
or were set to zero,  when the parcel had <{min_coverage * 100}% coverage.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "bold_mask",
                "boldref",
                "denoised_bold",
                "alff",  # may be Undefined
                "reho",
                "atlas_names",
                "atlas_files",
                "atlas_labels_files",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "atlas_names",
                "timeseries",
                "correlations",
                "coverage",
                "parcellated_alff",
                "parcellated_reho",
            ],
        ),
        name="outputnode",
    )

    functional_connectivity = pe.MapNode(
        NiftiConnect(min_coverage=min_coverage, correlate=True),
        name="functional_connectivity",
        iterfield=["atlas", "atlas_labels"],
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, functional_connectivity, [
            ("denoised_bold", "filtered_file"),
            ("bold_mask", "mask"),
            ("atlas_files", "atlas"),
            ("atlas_labels_files", "atlas_labels"),
        ]),
        (functional_connectivity, outputnode, [
            ("timeseries", "timeseries"),
            ("correlations", "correlations"),
            ("coverage", "coverage"),
        ]),
    ])
    # fmt:on

    parcellate_reho = pe.MapNode(
        NiftiConnect(min_coverage=min_coverage, correlate=False),
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

    if alff_available:
        parcellate_alff = pe.MapNode(
            NiftiConnect(min_coverage=min_coverage, correlate=False),
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
            ("denoised_bold", "in_file"),
            ("atlas_names", "atlas_names"),
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
def init_functional_connectivity_cifti_wf(
    output_dir,
    alff_available,
    min_coverage,
    mem_gb,
    omp_nthreads,
    name="connectivity_wf",
):
    """Extract CIFTI time series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.connectivity import init_functional_connectivity_cifti_wf
            wf = init_functional_connectivity_cifti_wf(
                output_dir=".",
                alff_available=True,
                min_coverage=0.5,
                mem_gb=0.1,
                omp_nthreads=1,
                name="connectivity_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    alff_available
    %(min_coverage)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "connectivity_wf".

    Inputs
    ------
    %(name_source)s
    denoised_bold
        Clean CIFTI after filtering and nuisance regression.
        The CIFTI file is in the same standard space as the atlases,
        so no transformations will be applied to the data before parcellation.
    alff
    reho

    Outputs
    -------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    %(timeseries)s
    %(timeseries_ciftis)s
    %(correlations)s
    %(correlation_ciftis)s
    %(coverage)s
    %(coverage_ciftis)s
    parcellated_reho
    parcellated_alff
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = f"""
Processed functional timeseries were extracted from residual BOLD using
Connectome Workbench [@hcppipelines] for the following atlases:
the Schaefer 17-network 100, 200, 300, 400, 500, 600, 700, 800, 900, and 1000 parcel
atlas [@Schaefer_2017], the Glasser atlas [@Glasser_2016],
the Gordon atlas [@Gordon_2014], and the Tian subcortical artlas [@tian2020topographic].
Corresponding pair-wise functional connectivity between all regions was computed for each atlas,
which was operationalized as the Pearson's correlation of each parcel's unsmoothed timeseries with
the Connectome Workbench.
In cases of partial coverage, uncovered vertices (values of all zeros or NaNs) were either
ignored, when the parcel had >{min_coverage * 100}% coverage,
or were set to zero, when the parcel had <{min_coverage * 100}% coverage.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "denoised_bold",
                "alff",  # may be Undefined
                "reho",
                "atlas_names",
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
                "atlas_names",
                "coverage_ciftis",
                "timeseries_ciftis",
                "correlation_ciftis",
                "coverage",
                "timeseries",
                "correlations",
                "connectplot",
                "parcellated_alff",
                "parcellated_reho",
            ],
        ),
        name="outputnode",
    )

    functional_connectivity = pe.MapNode(
        CiftiConnect(min_coverage=min_coverage, correlate=True),
        mem_gb=mem_gb,
        name="functional_connectivity",
        n_procs=omp_nthreads,
        iterfield=["atlas_labels", "atlas_file", "parcellated_atlas"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, functional_connectivity, [
            ("denoised_bold", "data_file"),
            ("atlas_files", "atlas_file"),
            ("atlas_labels_files", "atlas_labels"),
            ("parcellated_atlas_files", "parcellated_atlas"),
        ]),
        (functional_connectivity, outputnode, [
            ("coverage_ciftis", "coverage_ciftis"),
            ("timeseries_ciftis", "timeseries_ciftis"),
            ("correlation_ciftis", "correlation_ciftis"),
            ("coverage", "coverage"),
            ("timeseries", "timeseries"),
            ("correlations", "correlations"),
        ]),
    ])
    # fmt:on

    parcellate_reho = pe.MapNode(
        CiftiConnect(min_coverage=min_coverage, correlate=False),
        mem_gb=mem_gb,
        name="parcellate_reho",
        n_procs=omp_nthreads,
        iterfield=["atlas_labels", "atlas_file", "parcellated_atlas"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, parcellate_reho, [
            ("reho", "data_file"),
            ("atlas_files", "atlas_file"),
            ("atlas_labels_files", "atlas_labels"),
            ("parcellated_atlas_files", "parcellated_atlas"),
        ]),
        (parcellate_reho, outputnode, [("timeseries", "parcellated_reho")]),
    ])
    # fmt:on

    if alff_available:
        parcellate_alff = pe.MapNode(
            CiftiConnect(min_coverage=min_coverage, correlate=False),
            mem_gb=mem_gb,
            name="parcellate_alff",
            n_procs=omp_nthreads,
            iterfield=["atlas_labels", "atlas_file", "parcellated_atlas"],
        )

        # fmt:off
        workflow.connect([
            (inputnode, parcellate_alff, [
                ("alff", "data_file"),
                ("atlas_files", "atlas_file"),
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
            ("denoised_bold", "in_file"),
            ("atlas_names", "atlas_names"),
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
