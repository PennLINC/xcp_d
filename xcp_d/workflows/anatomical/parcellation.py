"""Workflows for parcellating anatomical data."""

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.utils.atlas import get_atlas_cifti, select_atlases
from xcp_d.utils.doc import fill_doc
from xcp_d.workflows.parcellation import init_parcellate_cifti_wf

LOGGER = logging.getLogger("nipype.workflow")


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
    from xcp_d.interfaces.workbench import CiftiCreateDenseFromTemplate

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
        resample_atlas_to_surface = pe.MapNode(
            CiftiCreateDenseFromTemplate(out_file="resampled_atlas.dlabel.nii"),
            name=f"resample_atlas_to_{file_to_parcellate}",
            iterfield=["label"],
        )
        workflow.connect([
            (inputnode, resample_atlas_to_surface, [(file_to_parcellate, "template_cifti")]),
            (atlas_file_grabber, resample_atlas_to_surface, [("atlas_file", "label")]),
        ])  # fmt:skip

        parcellate_surface_wf = init_parcellate_cifti_wf(
            mem_gb={"resampled": 2},
            compute_mask=True,
            name=f"parcellate_{file_to_parcellate}_wf",
        )
        workflow.connect([
            (inputnode, parcellate_surface_wf, [(file_to_parcellate, "inputnode.in_file")]),
            (atlas_file_grabber, parcellate_surface_wf, [
                ("atlas_labels_file", "inputnode.atlas_labels_files"),
            ]),
            (resample_atlas_to_surface, parcellate_surface_wf, [
                ("out_file", "inputnode.atlas_files"),
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
