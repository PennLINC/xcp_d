# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for extracting time series and computing functional connectivity."""

import nilearn as nl
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.connectivity import (
    ApplyTransformsx,
    CiftiConnect,
    ConnectPlot,
    NiftiConnect,
)
from xcp_d.interfaces.workbench import CiftiCreateDenseFromTemplate, CiftiParcellate
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_names, get_atlas_nifti
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.modified_data import cast_cifti_to_int16
from xcp_d.utils.utils import get_std2bold_xforms


@fill_doc
def init_nifti_functional_connectivity_wf(
    output_dir,
    mem_gb,
    omp_nthreads,
    name="nifti_fcon_wf",
):
    """Extract BOLD time series and compute functional connectivity.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.connectivity import init_nifti_functional_connectivity_wf
            wf = init_nifti_functional_connectivity_wf(
                output_dir=".",
                mem_gb=0.1,
                omp_nthreads=1,
                name="nifti_fcon_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "nifti_fcon_wf".

    Inputs
    ------
    bold_file
        Used for names.
    ref_file
    clean_bold
        clean bold after filtered out nuisscance and filtering
    %(template_to_t1w)s
    t1w_to_native

    Outputs
    -------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    %(timeseries)s
    %(correlations)s
    coverage : list of str
        Paths to atlas-specific coverage files.
    connectplot : str
        Path to the connectivity plot.
        This figure contains four ROI-to-ROI correlation heat maps from four of the atlases.
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = f"""
Processed functional timeseries were extracted from the residual BOLD signal
with *Nilearn's* [version {nl.__version__}, @nilearn] *NiftiLabelsMasker* for the following
atlases:
the Schaefer 17-network 100, 200, 300, 400, 500, 600, 700, 800, 900, and 1000 parcel
atlas [@Schaefer_2017], the Glasser atlas [@Glasser_2016],
the Gordon atlas [@Gordon_2014], and the Tian subcortical artlas [@tian2020topographic].
Corresponding pair-wise functional connectivity between all regions was computed for each atlas,
which was operationalized as the Pearson's correlation of each parcel's unsmoothed timeseries.
In cases of partial coverage, uncovered voxels (values of all zeros or NaNs) were either ignored,
when the parcel had >50% coverage, or were set to zero, when the parcel had <50% coverage.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "bold_file",
                "bold_mask",
                "ref_file",
                "clean_bold",
                "template_to_t1w",
                "t1w_to_native",
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
                "connectplot",
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
            input_names=["bold_file", "template_to_t1w", "t1w_to_native"],
            output_names=["transformfile"],
            function=get_std2bold_xforms,
        ),
        name="get_transforms_to_bold_space",
    )

    # fmt:off
    workflow.connect([
        (inputnode, get_transforms_to_bold_space, [
            ("bold_file", "bold_file"),
            ("template_to_t1w", "template_to_t1w"),
            ("t1w_to_native", "t1w_to_native"),
        ]),
    ])
    # fmt:on

    # Using the generated transforms, apply them to get everything in the correct MNI form
    warp_atlases_to_bold_space = pe.MapNode(
        ApplyTransformsx(
            interpolation="MultiLabel",
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
            ("ref_file", "reference_image"),
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
        NiftiConnect(),
        name="nifti_connect",
        iterfield=["atlas", "atlas_labels"],
        mem_gb=mem_gb,
    )

    # fmt:off
    workflow.connect([
        (inputnode, nifti_connect, [
            ("clean_bold", "filtered_file"),
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
        (inputnode, matrix_plot, [("clean_bold", "in_file")]),
        (atlas_name_grabber, matrix_plot, [("atlas_names", "atlas_names")]),
        (nifti_connect, matrix_plot, [("correlations", "correlations_tsv")]),
        (matrix_plot, outputnode, [("connectplot", "connectplot")]),
    ])
    # fmt:on

    ds_atlas = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            dismiss_entities=["datatype", "subject", "session", "task", "run", "desc"],
            allowed_entities=["space", "res", "den", "atlas", "desc", "cohort"],
            suffix="dseg",
            extension=".nii.gz",
        ),
        name="ds_atlas",
        iterfield=["atlas", "in_file"],
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (inputnode, ds_atlas, [("bold_file", "source_file")]),
        (atlas_name_grabber, ds_atlas, [("atlas_names", "atlas")]),
        (warp_atlases_to_bold_space, ds_atlas, [("output_image", "in_file")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_cifti_functional_connectivity_wf(
    output_dir,
    mem_gb,
    omp_nthreads,
    name="cifti_fcon_wf",
):
    """Extract CIFTI time series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.connectivity import init_cifti_functional_connectivity_wf
            wf = init_cifti_functional_connectivity_wf(
                output_dir=".",
                mem_gb=0.1,
                omp_nthreads=1,
                name="cifti_fcon_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "cifti_fcon_wf".

    Inputs
    ------
    clean_bold
        Clean CIFTI after filtering and nuisance regression.
        The CIFTI file is in the same standard space as the atlases,
        so no transformations will be applied to the data before parcellation.
    %(atlas_names)s
        Defined in the function.

    Outputs
    -------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    %(timeseries)s
    ptseries
        Paths to CIFTI-format timeseries files.
    %(correlations)s
    pconn
        Paths to CIFTI-format correlation matrices.
    coverage : list of str
        Paths to atlas-specific coverage files.
    connectplot : str
        Path to the connectivity plot.
        This figure contains four ROI-to-ROI correlation heat maps from four of the atlases.
    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """
Processed functional timeseries were extracted from residual BOLD using
Connectome Workbench [@hcppipelines] for the following atlases:
the Schaefer 17-network 100, 200, 300, 400, 500, 600, 700, 800, 900, and 1000 parcel
atlas [@Schaefer_2017], the Glasser atlas [@Glasser_2016],
the Gordon atlas [@Gordon_2014], and the Tian subcortical artlas [@tian2020topographic].
Corresponding pair-wise functional connectivity between all regions was computed for each atlas,
which was operationalized as the Pearson's correlation of each parcel's unsmoothed timeseries with
the Connectome Workbench.
In cases of partial coverage, uncovered vertices (values of all zeros or NaNs) were either ignored,
when the parcel had >50% coverage, or were set to zero, when the parcel had <50% coverage.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["clean_bold", "bold_file"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "atlas_names",
                "coverage_pscalar",
                "ptseries",
                "pconn",
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
        (inputnode, resample_atlas_to_data, [("clean_bold", "template_cifti")]),
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
        CiftiConnect(),
        mem_gb=mem_gb,
        name="cifti_connect",
        n_procs=omp_nthreads,
        iterfield=["atlas_labels", "atlas_file", "parcellated_atlas"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, cifti_connect, [("clean_bold", "data_file")]),
        (atlas_file_grabber, cifti_connect, [("atlas_labels_file", "atlas_labels")]),
        (resample_atlas_to_data, cifti_connect, [("cifti_out", "atlas_file")]),
        (parcellate_atlas, cifti_connect, [("out_file", "parcellated_atlas")]),
        (cifti_connect, outputnode, [
            ("coverage_pscalar", "coverage_pscalar"),
            ("ptseries", "ptseries"),
            ("pconn", "pconn"),
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
        (inputnode, matrix_plot, [("clean_bold", "in_file")]),
        (atlas_name_grabber, matrix_plot, [["atlas_names", "atlas_names"]]),
        (cifti_connect, matrix_plot, [("correlations", "correlations_tsv")]),
        (matrix_plot, outputnode, [("connectplot", "connectplot")]),
    ])
    # fmt:on

    # Coerce the bold_file to int16 before feeding it in as source_file,
    # as niworkflows 1.7.1's DerivativesDataSink tries to change the datatype of dseg files,
    # but treats them as niftis, which fails.
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
        (inputnode, ds_atlas, [("bold_file", "source_file")]),
        (atlas_name_grabber, ds_atlas, [("atlas_names", "atlas")]),
        (cast_atlas_to_int16, ds_atlas, [("out_file", "in_file")]),
    ])
    # fmt:on

    return workflow
