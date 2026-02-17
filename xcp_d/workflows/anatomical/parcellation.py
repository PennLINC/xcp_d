"""Workflows for parcellating anatomical data."""

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.config import dismiss_hash
from xcp_d.interfaces.bids import AddHashToTSV, DerivativesDataSink
from xcp_d.utils.atlas import select_atlases
from xcp_d.utils.doc import fill_doc
from xcp_d.workflows.parcellation import init_parcellate_cifti_wf

LOGGER = logging.getLogger('nipype.workflow')


@fill_doc
def init_parcellate_surfaces_wf(files_to_parcellate, name='parcellate_surfaces_wf'):
    """Parcellate surface files and write them out to the output directory.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.parcellation import init_parcellate_surfaces_wf

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
    from xcp_d.utils.atlas import collect_atlases

    workflow = Workflow(name=name)

    SURF_DESCS = {
        'sulcal_depth': 'sulc',
        'sulcal_curv': 'curv',
        'cortical_thickness': 'thickness',
        'cortical_thickness_corr': 'thicknessCorrected',
        'myelin': 'myelin',
        'myelin_smoothed': 'myelinSmoothed',
    }

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'sulcal_depth',
                'sulcal_curv',
                'cortical_thickness',
                'cortical_thickness_corr',
                'myelin',
                'myelin_smoothed',
                # atlases
                'atlas_names',
                'atlas_datasets',
                'atlas_files',
                'atlas_labels_files',
                'atlas_metadata_files',
            ],
        ),
        name='inputnode',
    )

    builtin_atlases = select_atlases(atlases=config.execution.atlases, subset='all')
    external_atlases = sorted(set(config.execution.atlases) - set(builtin_atlases))
    builtin_cortical_atlases = select_atlases(atlases=builtin_atlases, subset='cortical')
    selected_atlases = builtin_cortical_atlases + external_atlases
    atlases = collect_atlases(
        datasets=config.execution.datasets,
        atlases=selected_atlases,
        file_format='cifti',
        bids_filters=config.execution.bids_filters,
    )

    # Reorganize the atlas file information
    atlas_names, atlas_files, atlas_labels_files, atlas_metadata_files = [], [], [], []
    atlas_datasets = []
    for atlas, atlas_dict in atlases.items():
        config.loggers.workflow.info(f'Loading atlas: {atlas}')
        atlas_names.append(atlas)
        atlas_datasets.append(atlas_dict['dataset'])
        atlas_files.append(atlas_dict['image'])
        atlas_labels_files.append(atlas_dict['labels'])
        atlas_metadata_files.append(atlas_dict['metadata'])

    inputnode.inputs.atlas_names = atlas_names
    inputnode.inputs.atlas_datasets = atlas_datasets
    inputnode.inputs.atlas_files = atlas_files
    inputnode.inputs.atlas_labels_files = atlas_labels_files
    inputnode.inputs.atlas_metadata_files = atlas_metadata_files

    if not atlases:
        LOGGER.warning(
            'No cortical atlases have been selected, so surface metrics will not be parcellated.'
        )
        # If no cortical atlases are selected, inputnode could go unconnected, so add explicitly.
        workflow.add_nodes([inputnode])

        return workflow

    for file_to_parcellate in files_to_parcellate:
        resample_atlas_to_surface = pe.MapNode(
            CiftiCreateDenseFromTemplate(
                out_file='resampled_atlas.dlabel.nii',
                num_threads=config.nipype.omp_nthreads,
            ),
            name=f'resample_atlas_to_{file_to_parcellate}',
            iterfield=['label'],
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([
            (inputnode, resample_atlas_to_surface, [
                ('atlas_files', 'label'),
                (file_to_parcellate, 'template_cifti'),
            ]),
        ])  # fmt:skip

        parcellate_surface_wf = init_parcellate_cifti_wf(
            mem_gb={'bold': 2, 'volume': 1},
            compute_mask=True,
            name=f'parcellate_{file_to_parcellate}_wf',
        )
        workflow.connect([
            (inputnode, parcellate_surface_wf, [
                (file_to_parcellate, 'inputnode.in_file'),
                ('atlas_labels_files', 'inputnode.atlas_labels_files'),
            ]),
            (resample_atlas_to_surface, parcellate_surface_wf, [
                ('out_file', 'inputnode.atlas_files'),
            ]),
        ])  # fmt:skip

        add_hash_parcellated_surface = pe.MapNode(
            AddHashToTSV(
                add_to_columns=True,
                add_to_rows=False,
            ),
            name=f'add_hash_parcellated_{file_to_parcellate}',
            iterfield=['in_file'],
        )
        workflow.connect([
            (parcellate_surface_wf, add_hash_parcellated_surface, [
                ('outputnode.parcellated_tsv', 'in_file'),
            ]),
        ])  # fmt:skip

        # Write out the parcellated files
        ds_parcellated_surface = pe.MapNode(
            DerivativesDataSink(
                dismiss_entities=dismiss_hash(['hemi', 'desc', 'den', 'res']),
                desc=SURF_DESCS[file_to_parcellate],
                statistic='mean',
                suffix='morph',
                extension='.tsv',
            ),
            name=f'ds_parcellated_{file_to_parcellate}',
            run_without_submitting=True,
            mem_gb=1,
            iterfield=['atlas', 'in_file'],
        )
        workflow.connect([
            (inputnode, ds_parcellated_surface, [
                (file_to_parcellate, 'source_file'),
                ('atlas_names', 'atlas'),
            ]),
            (add_hash_parcellated_surface, ds_parcellated_surface, [('out_file', 'in_file')]),
        ])  # fmt:skip

    return workflow
