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
def init_functional_connectivity_nifti_wf(
    output_dir,
    min_coverage,
    mem_gb,
    omp_nthreads,
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
                min_coverage=0.5,
                mem_gb=0.1,
                omp_nthreads=1,
                name="connectivity_wf",
            )

    Parameters
    ----------
    %(output_dir)s
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
    %(template_to_anat_xfm)s
    %(anat_to_native_xfm)s

    Outputs
    -------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    %(timeseries)s
    %(correlations)s
    %(coverage)s
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
                "template_to_anat_xfm",
                "anat_to_native_xfm",
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
            ],
        ),
        name="outputnode",
    )

    atlas_name_grabber = pe.Node(
        Function(output_names=["atlas_names"], function=get_atlas_names),
        name="atlas_name_grabber",
    )

    # fmt:off
    workflow.connect([
        (atlas_name_grabber, outputnode, [("atlas_names", "atlas_names")]),
    ])
    # fmt:on

    # get atlases via pkgrf
    atlas_file_grabber = pe.MapNode(
        Function(
            input_names=["atlas_name"],
            output_names=["atlas_file", "atlas_labels_file"],
            function=get_atlas_nifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas_name"],
    )

    # fmt:off
    workflow.connect([
        (atlas_name_grabber, atlas_file_grabber, [("atlas_names", "atlas_name")]),
    ])
    # fmt:on

    get_transforms_to_bold_space = pe.Node(
        Function(
            input_names=["bold_file", "template_to_anat_xfm", "anat_to_native_xfm"],
            output_names=["transformfile"],
            function=get_std2bold_xfms,
        ),
        name="get_transforms_to_bold_space",
    )

    # fmt:off
    workflow.connect([
        (inputnode, get_transforms_to_bold_space, [
            ("name_source", "bold_file"),
            ("template_to_anat_xfm", "template_to_anat_xfm"),
            ("anat_to_native_xfm", "anat_to_native_xfm"),
        ]),
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
        (inputnode, warp_atlases_to_bold_space, [
            ("boldref", "reference_image"),
        ]),
        (atlas_file_grabber, warp_atlases_to_bold_space, [
            ("atlas_file", "input_image"),
        ]),
        (get_transforms_to_bold_space, warp_atlases_to_bold_space, [
            ("transformfile", "transforms"),
        ]),
    ])
    # fmt:on

    nifti_connect = pe.MapNode(
        NiftiConnect(min_coverage=min_coverage),
        name="nifti_connect",
        iterfield=["atlas", "atlas_labels"],
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, nifti_connect, [
            ("denoised_bold", "filtered_file"),
            ("bold_mask", "mask"),
        ]),
        (atlas_file_grabber, nifti_connect, [
            ("atlas_labels_file", "atlas_labels"),
        ]),
        (warp_atlases_to_bold_space, nifti_connect, [
            ("output_image", "atlas"),
        ]),
        (nifti_connect, outputnode, [
            ("timeseries", "timeseries"),
            ("correlations", "correlations"),
            ("coverage", "coverage"),
        ]),
    ])
    # fmt:on

    # Create a node to plot the matrixes
    matrix_plot = pe.Node(
        ConnectPlot(),
        name="matrix_plot",
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, matrix_plot, [("denoised_bold", "in_file")]),
        (atlas_name_grabber, matrix_plot, [("atlas_names", "atlas_names")]),
        (nifti_connect, matrix_plot, [("correlations", "correlations_tsv")]),
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
        (warp_atlases_to_bold_space, ds_atlas, [("output_image", "in_file")]),
    ])
    # fmt:on

    ds_report_connectivity = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="connectivityplot",
            datatype="figures",
        ),
        name="ds_report_connectivity",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_report_connectivity, [("name_source", "source_file")]),
        (matrix_plot, ds_report_connectivity, [("connectplot", "in_file")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_functional_connectivity_cifti_wf(
    output_dir,
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
                min_coverage=0.5,
                mem_gb=0.1,
                omp_nthreads=1,
                name="connectivity_wf",
            )

    Parameters
    ----------
    %(output_dir)s
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
        niu.IdentityInterface(fields=["name_source", "denoised_bold"]),
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
            ],
        ),
        name="outputnode",
    )

    # get atlases via pkgrf
    atlas_name_grabber = pe.Node(
        Function(output_names=["atlas_names"], function=get_atlas_names),
        name="atlas_name_grabber",
    )

    # fmt:off
    workflow.connect([
        (atlas_name_grabber, outputnode, [("atlas_names", "atlas_names")]),
    ])
    # fmt:on

    atlas_file_grabber = pe.MapNode(
        Function(
            input_names=["atlas_name"],
            output_names=["atlas_file", "atlas_labels_file"],
            function=get_atlas_cifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas_name"],
    )

    # fmt:off
    workflow.connect([
        (atlas_name_grabber, atlas_file_grabber, [("atlas_names", "atlas_name")]),
    ])
    # fmt:on

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
    ])
    # fmt:on

    cifti_connect = pe.MapNode(
        CiftiConnect(min_coverage=min_coverage),
        mem_gb=mem_gb,
        name="cifti_connect",
        n_procs=omp_nthreads,
        iterfield=["atlas_labels", "atlas_file", "parcellated_atlas"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, cifti_connect, [("denoised_bold", "data_file")]),
        (atlas_file_grabber, cifti_connect, [("atlas_labels_file", "atlas_labels")]),
        (resample_atlas_to_data, cifti_connect, [("cifti_out", "atlas_file")]),
        (parcellate_atlas, cifti_connect, [("out_file", "parcellated_atlas")]),
        (cifti_connect, outputnode, [
            ("coverage_ciftis", "coverage_ciftis"),
            ("timeseries_ciftis", "timeseries_ciftis"),
            ("correlation_ciftis", "correlation_ciftis"),
            ("coverage", "coverage"),
            ("timeseries", "timeseries"),
            ("correlations", "correlations"),
        ]),
    ])
    # fmt:on

    # Create a node to plot the matrixes
    matrix_plot = pe.Node(
        ConnectPlot(),
        name="matrix_plot",
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, matrix_plot, [("denoised_bold", "in_file")]),
        (atlas_name_grabber, matrix_plot, [["atlas_names", "atlas_names"]]),
        (cifti_connect, matrix_plot, [("correlations", "correlations_tsv")]),
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
        (atlas_file_grabber, cast_atlas_to_int16, [("atlas_file", "in_file")]),
    ])
    # fmt:on

    ds_atlas = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
            dismiss_entities=["datatype", "subject", "session", "task", "run", "desc"],
            allowed_entities=["space", "res", "den", "atlas", "desc", "cohort"],
            suffix="dseg",
            extension=".dlabel.nii",
        ),
        name="ds_atlas",
        iterfield=["atlas", "in_file"],
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_atlas, [("name_source", "source_file")]),
        (atlas_name_grabber, ds_atlas, [("atlas_names", "atlas")]),
        (cast_atlas_to_int16, ds_atlas, [("out_file", "in_file")]),
    ])
    # fmt:on

    ds_report_connectivity = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="connectivityplot",
            datatype="figures",
        ),
        name="ds_report_connectivity",
        run_without_submitting=False,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_report_connectivity, [("name_source", "source_file")]),
        (matrix_plot, ds_report_connectivity, [("connectplot", "in_file")]),
    ])
    # fmt:on

    return workflow
