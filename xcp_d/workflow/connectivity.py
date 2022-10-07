# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for extracting time series and computing functional connectivity."""

import nilearn as nl
from nipype import Function
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d.interfaces.connectivity import ApplyTransformsx, ConnectPlot
from xcp_d.interfaces.workbench import CiftiParcellate
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_names, get_atlas_nifti
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.fcon import compute_functional_connectivity, extract_timeseries_funct
from xcp_d.utils.utils import extract_ptseries, get_transformfile


@fill_doc
def init_nifti_functional_connectivity_wf(
    mem_gb,
    t1w_to_native,
    mni_to_t1w,
    omp_nthreads,
    name="nifti_fcon_wf",
):
    """Extract BOLD time series and compute functional connectivity.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.connectivity import init_nifti_functional_connectivity_wf
            wf = init_nifti_functional_connectivity_wf(
                mem_gb=0.1,
                t1w_to_native="identity",
                mni_to_t1w="identity",
                omp_nthreads=1,
                name="nifti_fcon_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    t1w_to_native: str
        transformation files from tw1 to native space ( from fmriprep)
    %(mni_to_t1w)s
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

    Outputs
    -------
    %(atlas_names)s
        Used for indexing ``timeseries`` and ``correlations``.
    %(timeseries)s
    %(correlations)s
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
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["bold_file", "ref_file", "clean_bold"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["atlas_names", "timeseries", "correlations", "connectplot"]),
        name="outputnode",
    )

    # get atlases via pkgrf
    atlas_file_grabber = pe.MapNode(
        Function(
            input_names=["atlas_name"],
            output_names=["atlas_file"],
            function=get_atlas_nifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas_name"],
    )

    atlas_name_grabber = pe.Node(
        Function(output_names=["atlas_names"], function=get_atlas_names),
        name="atlas_name_grabber",
    )

    get_transformfile_node = pe.Node(
        Function(
            input_names=["bold_file", "mni_to_t1w", "t1w_to_native"],
            output_names=["transformfile"],
            function=get_transformfile,
        ),
        name="get_transformfile_node",
    )
    get_transformfile_node.inputs.mni_to_t1w = mni_to_t1w
    get_transformfile_node.inputs.t1w_to_native = t1w_to_native

    # Using the generated transforms, apply them to get the atlases in the correct MNI form
    atlas_transform = pe.MapNode(
        ApplyTransformsx(
            interpolation="MultiLabel",
            input_image_type=3,
            dimension=3,
        ),
        name="atlas_mni_to_native",
        iterfield=["input_image"],
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    extract_parcel_timeseries = pe.MapNode(
        Function(
            input_names=["in_file", "atlas"],
            output_names=["timeseries"],
            function=extract_timeseries_funct,
        ),
        name="extract_parcel_timeseries",
        iterfield=["atlas"],
        mem_gb=mem_gb,
    )

    correlate_timeseries = pe.MapNode(
        Function(
            input_names=["in_file"],
            output_names=["correlations_file"],
            function=compute_functional_connectivity,
        ),
        name="correlate_timeseries",
        iterfield=["in_file"],
        mem_gb=mem_gb,
    )

    # Create a node to plot the matrixes
    matrix_plot = pe.Node(
        ConnectPlot(),
        name="matrix_plot",
        mem_gb=mem_gb,
    )

    workflow.connect([
        # Transform Atlas to correct MNI2009 space
        (inputnode, get_transformfile_node, [("bold_file", "bold_file")]),
        (inputnode, atlas_transform, [("ref_file", "reference_image")]),
        (inputnode, extract_parcel_timeseries, [("clean_bold", "in_file")]),
        (inputnode, matrix_plot, [("clean_bold", "in_file")]),
        (atlas_name_grabber, outputnode, [("atlas_names", "atlas_names")]),
        (atlas_name_grabber, atlas_file_grabber, [("atlas_names", "atlas_name")]),
        (atlas_name_grabber, matrix_plot, [["atlas_names", "atlas_names"]]),
        (atlas_file_grabber, atlas_transform, [("atlas_file", "input_image")]),
        (get_transformfile_node, atlas_transform, [("transformfile", "transforms")]),
        (atlas_transform, extract_parcel_timeseries, [("output_image", "atlas")]),
        (extract_parcel_timeseries, outputnode, [("timeseries", "timeseries")]),
        (extract_parcel_timeseries, correlate_timeseries, [("timeseries", "in_file")]),
        (correlate_timeseries, outputnode, [("correlations_file", "correlations")]),
        (correlate_timeseries, matrix_plot, [("correlations_file", "correlation_tsvs")]),
        (matrix_plot, outputnode, [("connectplot", "connectplot")]),
    ])

    return workflow


@fill_doc
def init_cifti_functional_connectivity_wf(
    mem_gb,
    omp_nthreads,
    name="cifti_fcon_wf",
):
    """Extract CIFTI time series.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.connectivity import init_cifti_functional_connectivity_wf
            wf = init_cifti_functional_connectivity_wf(
                mem_gb=0.1,
                omp_nthreads=1,
                name="cifti_fcon_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    %(omp_nthreads)s

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
    %(correlations)s
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
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["clean_bold"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["atlas_names", "timeseries", "correlations", "connectplot"]),
        name="outputnode",
    )

    # get atlases via pkgrf
    atlas_name_grabber = pe.Node(
        Function(output_names=["atlas_names"], function=get_atlas_names),
        name="atlas_name_grabber",
    )

    atlas_file_grabber = pe.MapNode(
        Function(
            input_names=["atlas_name"],
            output_names=["atlas_file"],
            function=get_atlas_cifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas_name"],
    )

    parcellate_data = pe.MapNode(
        CiftiParcellate(direction="COLUMN"),
        mem_gb=mem_gb,
        name="parcellate_data",
        n_procs=omp_nthreads,
        iterfield=["atlas_label"],
    )

    extract_parcel_timeseries = pe.MapNode(
        Function(
            input_names=["in_file"],
            output_names=["timeseries_file", "correlations_file"],
            function=extract_ptseries,
        ),
        name="extract_timeseries",
        iterfield=["in_file"],
        mem_gb=mem_gb,
    )

    correlate_timeseries = pe.MapNode(
        Function(
            input_names=["in_file"],
            output_names=["correlations_file"],
            function=compute_functional_connectivity,
        ),
        name="correlate_timeseries",
        iterfield=["in_file"],
        mem_gb=mem_gb,
    )

    # Create a node to plot the matrixes
    matrix_plot = pe.Node(
        ConnectPlot(),
        name="matrix_plot",
        mem_gb=mem_gb,
    )

    workflow.connect([
        (inputnode, parcellate_data, [("clean_bold", "in_file")]),
        (inputnode, matrix_plot, [("clean_bold", "in_file")]),
        (atlas_name_grabber, outputnode, [("atlas_names", "atlas_names")]),
        (atlas_name_grabber, atlas_file_grabber, [("atlas_names", "atlas_name")]),
        (atlas_name_grabber, matrix_plot, [["atlas_names", "atlas_names"]]),
        (atlas_file_grabber, parcellate_data, [("atlas_file", "atlas_label")]),
        (parcellate_data, extract_parcel_timeseries, [("out_file", "in_file")]),
        (extract_parcel_timeseries, outputnode, [("timeseries_file", "timeseries")]),
        (extract_parcel_timeseries, correlate_timeseries, [("timeseries_file", "in_file")]),
        (correlate_timeseries, matrix_plot, [("correlations_file", "correlation_tsvs")]),
        (correlate_timeseries, outputnode, [("correlations_file", "correlations")]),
        (matrix_plot, outputnode, [("connectplot", "connectplot")]),
    ])

    return workflow
