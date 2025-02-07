# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for generating plots from anatomical data."""

from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import ApplyTransformsToPoints
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.surf import CSVToGifti, GiftiToCSV

from xcp_d import config
from xcp_d.data import load as load_data
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.nilearn import ResampleToImage
from xcp_d.interfaces.workbench import ShowScene
from xcp_d.utils.doc import fill_doc
from xcp_d.utils.execsummary import (
    get_png_image_names,
    make_mosaic,
    modify_pngs_scene_template,
    plot_slice_for_brainsprite,
)
from xcp_d.workflows.plotting import init_plot_overlay_wf

LOGGER = logging.getLogger('nipype.workflow')


@fill_doc
def init_brainsprite_figures_wf(
    t1w_available,
    t2w_available,
    apply_transform,
    name='brainsprite_figures_wf',
):
    """Create mosaic and PNG files for executive summary brainsprite.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.plotting import init_brainsprite_figures_wf

            with mock_config():
                wf = init_brainsprite_figures_wf(
                    t1w_available=True,
                    t2w_available=True,
                    apply_transform=True,
                    name="brainsprite_figures_wf",
                )

    Parameters
    ----------
    t1w_available : bool
        True if a T1w image is available.
    t2w_available : bool
        True if a T2w image is available.
    apply_transform : bool
        Whether to apply the transform to the surfaces.
    %(name)s
        Default is "init_brainsprite_figures_wf".

    Inputs
    ------
    t1w
        Path to T1w image. Optional. Should only be defined if ``t1w_available`` is True.
    t2w
        Path to T2w image. Optional. Should only be defined if ``t2w_available`` is True.
    lh_wm_surf
    rh_wm_surf
    lh_pial_surf
    rh_pial_surf
    template_to_anat_xfm
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1w',
                't2w',
                'lh_wm_surf',
                'rh_wm_surf',
                'lh_pial_surf',
                'rh_pial_surf',
                'template_to_anat_xfm',
            ],
        ),
        name='inputnode',
    )

    # Modify the surfaces to be in the same space as the T1w image
    surface_buffer = pe.Node(
        niu.IdentityInterface(
            fields=['lh_wm_surf', 'rh_wm_surf', 'lh_pial_surf', 'rh_pial_surf'],
        ),
        name='surface_buffer',
    )

    if apply_transform:
        for surface in ['lh_wm_surf', 'rh_wm_surf', 'lh_pial_surf', 'rh_pial_surf']:
            # Warp the surfaces to the template space
            warp_to_template_wf = init_itk_warp_gifti_surface_wf(name=f'{surface}_warp_wf')
            workflow.connect([
                (inputnode, warp_to_template_wf, [
                    (surface, 'inputnode.native_surf_gii'),
                    ('template_to_anat_xfm', 'inputnode.itk_warp_file'),
                ]),
                (warp_to_template_wf, surface_buffer, [('outputnode.warped_surf_gii', surface)]),
            ])  # fmt:skip

    else:
        workflow.connect([
            (inputnode, surface_buffer, [
                ('lh_wm_surf', 'lh_wm_surf'),
                ('rh_wm_surf', 'rh_wm_surf'),
                ('lh_pial_surf', 'lh_pial_surf'),
                ('rh_pial_surf', 'rh_pial_surf'),
            ]),
        ])  # fmt:skip

    # Load template scene file
    pngs_scene_template = str(load_data('executive_summary_scenes/pngs_template.scene.gz'))

    if t1w_available and t2w_available:
        image_types = ['T1', 'T2']
    elif t2w_available:
        image_types = ['T2']
    else:
        image_types = ['T1']

    for image_type in image_types:
        inputnode_anat_name = f'{image_type.lower()}w'

        # Modify template scene file with file paths
        plot_slices = pe.Node(
            Function(
                function=plot_slice_for_brainsprite,
                input_names=[
                    'nifti',
                    'lh_wm',
                    'rh_wm',
                    'lh_pial',
                    'rh_pial',
                ],
                output_names=['out_files'],
            ),
            name=f'plot_slices_{image_type}',
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, plot_slices, [(inputnode_anat_name, 'nifti')]),
            (surface_buffer, plot_slices, [
                ('lh_wm_surf', 'lh_wm'),
                ('rh_wm_surf', 'rh_wm'),
                ('lh_pial_surf', 'lh_pial'),
                ('rh_pial_surf', 'rh_pial'),
            ]),
        ])  # fmt:skip

        # Make mosaic
        make_mosaic_node = pe.Node(
            Function(
                function=make_mosaic,
                input_names=['png_files'],
                output_names=['mosaic_file'],
            ),
            name=f'make_mosaic_{image_type}',
            mem_gb=1,
        )
        workflow.connect([(plot_slices, make_mosaic_node, [('out_files', 'png_files')])])

        ds_report_mosaic_file = pe.Node(
            DerivativesDataSink(
                dismiss_entities=['desc'],
                desc='mosaic',
                suffix=f'{image_type}w',
            ),
            name=f'ds_report_mosaic_file_{image_type}',
            run_without_submitting=False,
        )
        workflow.connect([
            (inputnode, ds_report_mosaic_file, [(inputnode_anat_name, 'source_file')]),
            (make_mosaic_node, ds_report_mosaic_file, [('mosaic_file', 'in_file')]),
        ])  # fmt:skip

        # Start working on the selected PNG images for the button
        modify_pngs_template_scene = pe.Node(
            Function(
                function=modify_pngs_scene_template,
                input_names=[
                    'anat_file',
                    'rh_pial_surf',
                    'lh_pial_surf',
                    'rh_wm_surf',
                    'lh_wm_surf',
                    'scene_template',
                ],
                output_names=['out_file'],
            ),
            name=f'modify_pngs_template_scene_{image_type}',
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        modify_pngs_template_scene.inputs.scene_template = pngs_scene_template
        workflow.connect([
            (inputnode, modify_pngs_template_scene, [(inputnode_anat_name, 'anat_file')]),
            (surface_buffer, modify_pngs_template_scene, [
                ('lh_wm_surf', 'lh_wm_surf'),
                ('rh_wm_surf', 'rh_wm_surf'),
                ('lh_pial_surf', 'lh_pial_surf'),
                ('rh_pial_surf', 'rh_pial_surf'),
            ])
        ])  # fmt:skip

        # Create specific PNGs for button
        get_png_scene_names = pe.Node(
            Function(
                function=get_png_image_names,
                output_names=['scene_index', 'scene_descriptions'],
            ),
            name=f'get_png_scene_names_{image_type}',
        )

        create_scenewise_pngs = pe.MapNode(
            ShowScene(image_width=900, image_height=800, num_threads=config.nipype.omp_nthreads),
            name=f'create_scenewise_pngs_{image_type}',
            iterfield=['scene_name_or_number'],
            mem_gb=1,
            n_procs=config.nipype.omp_nthreads,
        )
        workflow.connect([
            (modify_pngs_template_scene, create_scenewise_pngs, [('out_file', 'scene_file')]),
            (get_png_scene_names, create_scenewise_pngs, [
                ('scene_index', 'scene_name_or_number'),
            ]),
        ])  # fmt:skip

        ds_report_scenewise_pngs = pe.MapNode(
            DerivativesDataSink(
                dismiss_entities=['desc'],
                suffix=f'{image_type}w',
            ),
            name=f'ds_report_scenewise_pngs_{image_type}',
            run_without_submitting=False,
            iterfield=['desc', 'in_file'],
            mem_gb=config.DEFAULT_MEMORY_MIN_GB,
        )
        workflow.connect([
            (inputnode, ds_report_scenewise_pngs, [(inputnode_anat_name, 'source_file')]),
            (get_png_scene_names, ds_report_scenewise_pngs, [('scene_descriptions', 'desc')]),
            (create_scenewise_pngs, ds_report_scenewise_pngs, [('out_file', 'in_file')]),
        ])  # fmt:skip

    return workflow


@fill_doc
def init_itk_warp_gifti_surface_wf(name='itk_warp_gifti_surface_wf'):
    """Apply an arbitrary ITK transform to a Gifti file.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.plotting import init_itk_warp_gifti_surface_wf

            with mock_config():
                wf = init_itk_warp_gifti_surface_wf()

    Parameters
    ----------
    %(name)s

    Inputs
    ------
    native_surf_gii
        T1w image, after warping to standard space.
    itk_warp_file
        T2w image, after warping to standard space.

    Outputs
    -------
    warped_surf_gii
        Gifti file where the transform in ``itk_warp_file`` has been applied
        to the vertices in ``native_surf_gii``.
    """
    from bids.utils import listify

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'native_surf_gii',
                'itk_warp_file',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'warped_surf_gii',
            ],
        ),
        name='outputnode',
    )

    listify_transform = pe.Node(
        niu.Function(
            function=listify,
            input_names=['obj'],
            output_names=['out_list'],
        ),
        name='listify_transform',
    )
    workflow.connect([(inputnode, listify_transform, [('itk_warp_file', 'obj')])])

    convert_to_csv = pe.Node(GiftiToCSV(itk_lps=True), name='convert_to_csv')
    workflow.connect([(inputnode, convert_to_csv, [('native_surf_gii', 'in_file')])])

    transform_vertices = pe.Node(
        ApplyTransformsToPoints(dimension=3),
        name='transform_vertices',
    )
    workflow.connect([
        (listify_transform, transform_vertices, [('out_list', 'transforms')]),
        (convert_to_csv, transform_vertices, [('out_file', 'input_file')]),
    ])  # fmt:skip

    csv_to_gifti = pe.Node(CSVToGifti(itk_lps=True), name='csv_to_gifti')
    workflow.connect([
        (inputnode, csv_to_gifti, [('native_surf_gii', 'gii_file')]),
        (transform_vertices, csv_to_gifti, [('output_file', 'in_file')]),
        (csv_to_gifti, outputnode, [('out_file', 'warped_surf_gii')]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_execsummary_anatomical_plots_wf(
    t1w_available,
    t2w_available,
    name='execsummary_anatomical_plots_wf',
):
    """Generate the anatomical figures for an executive summary.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.plotting import init_execsummary_anatomical_plots_wf

            with mock_config():
                wf = init_execsummary_anatomical_plots_wf(
                    t1w_available=True,
                    t2w_available=True,
                )

    Parameters
    ----------
    t1w_available : bool
        Generally True.
    t2w_available : bool
        Generally False.
    %(name)s

    Inputs
    ------
    t1w
        T1w image, after warping to standard space.
    t2w
        T2w image, after warping to standard space.
    template
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                't1w',
                't2w',
                'template',
            ],
        ),
        name='inputnode',
    )

    # Start plotting the overlay figures
    # Atlas in T1w/T2w, T1w/T2w in Atlas
    anatomicals = (['t1w'] if t1w_available else []) + (['t2w'] if t2w_available else [])
    for anat in anatomicals:
        # Resample anatomical to match resolution of template data
        resample_anat = pe.Node(
            ResampleToImage(),
            name=f'resample_{anat}',
            mem_gb=1,
        )
        workflow.connect([
            (inputnode, resample_anat, [
                (anat, 'in_file'),
                ('template', 'target_file'),
            ]),
        ])  # fmt:skip

        plot_anat_on_atlas_wf = init_plot_overlay_wf(
            desc='AnatOnAtlas',
            name=f'plot_{anat}_on_atlas_wf',
        )
        workflow.connect([
            (inputnode, plot_anat_on_atlas_wf, [
                ('template', 'inputnode.underlay_file'),
                (anat, 'inputnode.name_source'),
            ]),
            (resample_anat, plot_anat_on_atlas_wf, [('out_file', 'inputnode.overlay_file')]),
        ])  # fmt:skip

        plot_atlas_on_anat_wf = init_plot_overlay_wf(
            desc='AtlasOnAnat',
            name=f'plot_atlas_on_{anat}_wf',
        )
        workflow.connect([
            (inputnode, plot_atlas_on_anat_wf, [
                ('template', 'inputnode.overlay_file'),
                (anat, 'inputnode.name_source'),
            ]),
            (resample_anat, plot_atlas_on_anat_wf, [('out_file', 'inputnode.underlay_file')]),
        ])  # fmt:skip

    # TODO: Add subcortical overlay images as well.
    # 1. Binarize atlas.

    return workflow
