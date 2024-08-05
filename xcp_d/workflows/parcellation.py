# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for parcellating imaging data."""

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.ants import ApplyTransforms
from xcp_d.interfaces.nilearn import IndexImage
from xcp_d.utils.atlas import get_atlas_cifti, get_atlas_nifti
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
    file_format = config.workflow.file_format
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
            function=get_atlas_cifti if file_format == "cifti" else get_atlas_nifti,
        ),
        name="atlas_file_grabber",
        iterfield=["atlas"],
    )
    atlas_file_grabber.inputs.atlas = atlases

    atlas_buffer = pe.Node(niu.IdentityInterface(fields=["atlas_file"]), name="atlas_buffer")

    if file_format == "nifti":
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


def init_parcellate_cifti_wf(
    mem_gb,
    compute_mask=True,
    name="parcellate_cifti_wf",
):
    """Parcellate a CIFTI file using a set of atlases.

    Part of the parcellation includes applying vertex-wise and node-wise masks.

    Vertex-wise masks are typically calculated from the full BOLD run,
    wherein any vertex that has a time series of all zeros or NaNs is excluded.
    Additionally, if *any* volumes in a vertex's time series are NaNs,
    that vertex will be excluded.

    The node-wise mask is determined based on the vertex-wise mask and the workflow's
    coverage threshold.
    Any nodes in the atlas with less than the coverage threshold's % of vertices retained by the
    vertex-wise mask will have that node's time series set to NaNs.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.connectivity import init_parcellate_cifti_wf

            with mock_config():
                wf = init_parcellate_cifti_wf(mem_gb={"resampled": 2})

    Parameters
    ----------
    mem_gb : :obj:`dict`
        Dictionary of memory allocations.
    compute_mask : :obj:`bool`
        Whether to compute a vertex-wise mask for the CIFTI file.
        When processing full BOLD runs, this should be True.
        When processing truncated BOLD runs or scalar maps, this should be False,
        and the vertex-wise mask should be provided via the inputnode..
        Default is True.
    name : :obj:`str`
        Workflow name.
        Default is "parcellate_cifti_wf".

    Inputs
    ------
    in_file
        CIFTI file to parcellate.
    atlas_files
        List of CIFTI atlas files.
    atlas_labels_files
        List of TSV atlas labels files.
    vertexwise_coverage
        Vertex-wise coverage mask.
        Only used if `compute_mask` is False.
    coverage_cifti
        Coverage CIFTI files. One for each atlas.
        Only used if `compute_mask` is False.

    Outputs
    -------
    parcellated_cifti
        Parcellated CIFTI files. One for each atlas.
    parcellated_tsv
        Parcellated TSV files. One for each atlas.
    vertexwise_coverage
        Vertex-wise coverage mask. Only output if `compute_mask` is True.
    coverage_cifti
        Coverage CIFTI files. One for each atlas. Only output if `compute_mask` is True.
    coverage_tsv
        Coverage TSV files. One for each atlas. Only output if `compute_mask` is True.
    """
    from xcp_d import config
    from xcp_d.interfaces.connectivity import CiftiMask, CiftiToTSV, CiftiVertexMask
    from xcp_d.interfaces.workbench import CiftiMath, CiftiParcellateWorkbench

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
            CiftiVertexMask(),
            name="vertexwise_coverage",
        )
        workflow.connect([
            (inputnode, vertexwise_coverage, [("in_file", "in_file")]),
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
            iterfield=["atlas_label"],
        )
        workflow.connect([
            (inputnode, parcellate_coverage, [("atlas_files", "atlas_label")]),
            (vertexwise_coverage, parcellate_coverage, [("mask_file", "in_file")]),
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
            out_file=f"parcellated_data.{'ptseries' if compute_mask else 'pscalar'}.nii",
        ),
        name="parcellate_data",
        iterfield=["atlas_label"],
        mem_gb=mem_gb["resampled"],
    )
    workflow.connect([
        (inputnode, parcellate_data, [
            ("in_file", "in_file"),
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
        CiftiMask(),
        name="mask_parcellated_data",
        iterfield=["in_file", "mask"],
        mem_gb=mem_gb["resampled"],
    )
    workflow.connect([
        (parcellate_data, mask_parcellated_data, [("out_file", "in_file")]),
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
