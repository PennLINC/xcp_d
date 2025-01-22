# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for processing surface anatomical files."""

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.bids import CollectRegistrationFiles, DerivativesDataSink
from xcp_d.interfaces.workbench import (
    CiftiSurfaceResample,
    SurfaceAverage,
    SurfaceGenerateInflated,
    SurfaceSphereProjectUnproject,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.workflows.anatomical.outputs import init_copy_inputs_to_outputs_wf
from xcp_d.workflows.anatomical.plotting import init_brainsprite_figures_wf

LOGGER = logging.getLogger('nipype.workflow')


@fill_doc
def init_postprocess_surfaces_wf(
    mesh_available,
    standard_space_mesh,
    morphometry_files,
    t1w_available,
    t2w_available,
    software,
    name='postprocess_surfaces_wf',
):
    """Postprocess surfaces.

    If DCAN QC is enabled, this will generate a BrainSprite for the executive summary.
    If process-surfaces is enabled *or* fsLR-space mesh files are available,
    then the BrainSprite will use standard-space mesh files.
    Otherwise, it will use the native-space mesh files.

    If process-surfaces is enabled and mesh files (i.e., white and pial surfaces) are available in
    fsnative space, this workflow will warp them to fsLR space.
    If process-surfaces is enabled and the mesh files are already in fsLR space,
    they will be copied to the output directory.
    These fsLR-space mesh files retain the subject's morphology,
    and are thus useful for visualizing fsLR-space statistical derivatives on the subject's brain.
    The workflow will also rigidly align the meshes to the MNI152NLin6Asym template,
    so that they can be overlaid on top of the template for visualization.

    As long as process-surfaces is enabled and mesh files (in either space) are available,
    HCP-style midthickness, inflated, and very-inflated surfaces will be generated from them.

    If process-surfaces is enabled and morphometry files (e.g., sulcal depth, cortical thickness)
    are available in fsLR space, they will be copied to the output directory.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.surface import init_postprocess_surfaces_wf

            with mock_config():
                wf = init_postprocess_surfaces_wf(
                    mesh_available=True,
                    standard_space_mesh=False,
                    morphometry_files=[],
                    t1w_available=True,
                    t2w_available=True,
                    software="FreeSurfer",
                    name="postprocess_surfaces_wf",
                )

    Parameters
    ----------
    mesh_available : bool
    standard_space_mesh : bool
    morphometry_files : list of str
    t1w_available : bool
        True if a T1w image is available.
    t2w_available : bool
        True if a T2w image is available.
    software : {"MCRIBS", "FreeSurfer"}
        The software used to generate the surfaces.
    %(name)s
        Default is "postprocess_surfaces_wf".

    Inputs
    ------
    t1w
        Preprocessed T1w file. May be in native or standard space.
    t2w
        Preprocessed T2w file. May be in native or standard space.
    %(anat_to_template_xfm)s
    lh_pial_surf, rh_pial_surf
    lh_wm_surf, rh_wm_surf
    lh_subject_sphere, rh_subject_sphere
    sulcal_depth
    sulcal_curv
    cortical_thickness
    cortical_thickness_corr
    myelin
    myelin_smoothed
    """
    workflow = Workflow(name=name)

    abcc_qc = config.workflow.abcc_qc
    process_surfaces = config.workflow.process_surfaces
    omp_nthreads = config.nipype.omp_nthreads

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Volumetric anatomical images to use for brainsprite
                't1w',
                't2w',
                # Transform to allow the surfaces to be overlaid on the MNI152NLin6Asym template
                'anat_to_template_xfm',
                # Spheres to use for warping mesh files to fsLR space
                'lh_subject_sphere',
                'rh_subject_sphere',
                # Mesh files, either in fsnative or fsLR space
                'lh_pial_surf',
                'rh_pial_surf',
                'lh_wm_surf',
                'rh_wm_surf',
                # fsLR-space morphometry files to copy to the output directory and parcellate
                'sulcal_depth',
                'sulcal_curv',
                'cortical_thickness',
                'cortical_thickness_corr',
                'myelin',
                'myelin_smoothed',
            ],
        ),
        name='inputnode',
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # Surface files to use as underlays in surface plots
                'lh_midthickness',
                'rh_midthickness',
            ],
        ),
        name='outputnode',
    )
    workflow.add_nodes([outputnode])  # outputnode may not be used

    workflow.__desc__ = ''

    if abcc_qc and mesh_available:
        # Plot the white and pial surfaces on the brain in a brainsprite figure.
        if (not process_surfaces) or (mesh_available and standard_space_mesh):
            # Use original surfaces for brainsprite.
            # For fMRIPrep derivatives, this will be the native-space surfaces.
            # For DCAN/HCP derivatives, it will be standard-space surfaces.
            brainsprite_wf = init_brainsprite_figures_wf(
                t1w_available=t1w_available,
                t2w_available=t2w_available,
                apply_transform=False,
                name='brainsprite_wf',
            )

            workflow.connect([
                (inputnode, brainsprite_wf, [
                    ('t1w', 'inputnode.t1w'),
                    ('t2w', 'inputnode.t2w'),
                    ('lh_pial_surf', 'inputnode.lh_pial_surf'),
                    ('rh_pial_surf', 'inputnode.rh_pial_surf'),
                    ('lh_wm_surf', 'inputnode.lh_wm_surf'),
                    ('rh_wm_surf', 'inputnode.rh_wm_surf'),
                ]),
            ])  # fmt:skip

    if not process_surfaces:
        # Return early, as all other steps require process_surfaces.
        return workflow

    if morphometry_files or (mesh_available and standard_space_mesh):
        # At least some surfaces are already in fsLR space and must be copied,
        # without modification, to the output directory.
        copy_morphs_wf = init_copy_inputs_to_outputs_wf(name='copy_morphs_wf')

    if morphometry_files:
        workflow.__desc__ += (
            ' fsLR-space morphometry surfaces were copied from the preprocessing derivatives to '
            'the XCP-D derivatives.'
        )
        for morphometry_file in morphometry_files:
            workflow.connect([
                (inputnode, copy_morphs_wf, [(morphometry_file, f'inputnode.{morphometry_file}')]),
            ])  # fmt:skip

    if mesh_available:
        workflow.__desc__ += (
            ' HCP-style midthickness, inflated, and very-inflated surfaces were generated from '
            'the white-matter and pial surface meshes.'
        )
        # Generate and output HCP-style surface files.
        hcp_surface_wfs = {
            hemi: init_generate_hcp_surfaces_wf(name=f'{hemi}_generate_hcp_surfaces_wf')
            for hemi in ['lh', 'rh']
        }
        workflow.connect([
            (inputnode, hcp_surface_wfs['lh'], [('lh_pial_surf', 'inputnode.name_source')]),
            (inputnode, hcp_surface_wfs['rh'], [('rh_pial_surf', 'inputnode.name_source')]),
            (hcp_surface_wfs['lh'], outputnode, [('outputnode.midthickness', 'lh_midthickness')]),
            (hcp_surface_wfs['rh'], outputnode, [('outputnode.midthickness', 'rh_midthickness')]),
        ])  # fmt:skip

    if mesh_available and standard_space_mesh:
        workflow.__desc__ += (
            ' All surface files were already in fsLR space, and were copied to the output '
            'directory.'
        )
        # Mesh files are already in fsLR.
        workflow.connect([
            (inputnode, copy_morphs_wf, [
                ('lh_pial_surf', 'inputnode.lh_pial_surf'),
                ('rh_pial_surf', 'inputnode.rh_pial_surf'),
                ('lh_wm_surf', 'inputnode.lh_wm_surf'),
                ('rh_wm_surf', 'inputnode.rh_wm_surf'),
            ]),
            (inputnode, hcp_surface_wfs['lh'], [
                ('lh_pial_surf', 'inputnode.pial_surf'),
                ('lh_wm_surf', 'inputnode.wm_surf'),
            ]),
            (inputnode, hcp_surface_wfs['rh'], [
                ('rh_pial_surf', 'inputnode.pial_surf'),
                ('rh_wm_surf', 'inputnode.wm_surf'),
            ]),
        ])  # fmt:skip

    elif mesh_available:
        workflow.__desc__ += ' fsnative-space surfaces were then warped to fsLR space.'
        # Mesh files are in fsnative and must be warped to fsLR.
        fsnative_to_fsLR_wf = init_fsnative_to_fsLR_wf(
            software=software,
            omp_nthreads=omp_nthreads,
            name='fsnative_to_fsLR_wf',
        )
        workflow.connect([
            (inputnode, fsnative_to_fsLR_wf, [
                ('lh_subject_sphere', 'inputnode.lh_subject_sphere'),
                ('rh_subject_sphere', 'inputnode.rh_subject_sphere'),
                ('lh_pial_surf', 'inputnode.lh_pial_surf'),
                ('rh_pial_surf', 'inputnode.rh_pial_surf'),
                ('lh_wm_surf', 'inputnode.lh_wm_surf'),
                ('rh_wm_surf', 'inputnode.rh_wm_surf'),
            ]),
            (fsnative_to_fsLR_wf, hcp_surface_wfs['lh'], [
                ('outputnode.lh_pial_surf', 'inputnode.pial_surf'),
                ('outputnode.lh_wm_surf', 'inputnode.wm_surf'),
            ]),
            (fsnative_to_fsLR_wf, hcp_surface_wfs['rh'], [
                ('outputnode.rh_pial_surf', 'inputnode.pial_surf'),
                ('outputnode.rh_wm_surf', 'inputnode.wm_surf'),
            ]),
        ])  # fmt:skip

        if abcc_qc:
            # Use standard-space T1w and surfaces for brainsprite.
            brainsprite_wf = init_brainsprite_figures_wf(
                t1w_available=t1w_available,
                t2w_available=t2w_available,
                apply_transform=True,
                name='brainsprite_wf',
            )
            workflow.connect([
                (inputnode, brainsprite_wf, [
                    ('t1w', 'inputnode.t1w'),
                    ('t2w', 'inputnode.t2w'),
                    ('anat_to_template_xfm', 'inputnode.anat_to_template_xfm'),
                    ('lh_pial_surf', 'inputnode.lh_pial_surf'),
                    ('rh_pial_surf', 'inputnode.rh_pial_surf'),
                    ('lh_wm_surf', 'inputnode.lh_wm_surf'),
                    ('rh_wm_surf', 'inputnode.rh_wm_surf'),
                ]),
            ])  # fmt:skip

    elif not morphometry_files:
        raise ValueError(
            'No surfaces found. Surfaces are required if `--warp-surfaces-native2std` is enabled.'
        )

    return workflow


@fill_doc
def init_fsnative_to_fsLR_wf(
    software,
    omp_nthreads,
    name='fsnative_to_fsLR_wf',
):
    """Transform surfaces from native to standard fsLR-32k space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.anatomical.surface import init_fsnative_to_fsLR_wf

            wf = init_fsnative_to_fsLR_wf(
                software="FreeSurfer",
                omp_nthreads=1,
                name="fsnative_to_fsLR_wf",
            )

    Parameters
    ----------
    software : {"MCRIBS", "FreeSurfer"}
        The software used to generate the surfaces.
    %(omp_nthreads)s
    %(name)s
        Default is "fsnative_to_fsLR_wf".

    Inputs
    ------
    lh_subject_sphere, rh_subject_sphere : :obj:`str`
        Left- and right-hemisphere sphere registration files.
    lh_pial_surf, rh_pial_surf : :obj:`str`
        Left- and right-hemisphere pial surface files in fsnative space.
    lh_wm_surf, rh_wm_surf : :obj:`str`
        Left- and right-hemisphere smoothed white matter surface files in fsnative space.

    Outputs
    -------
    lh_pial_surf, rh_pial_surf : :obj:`str`
        Left- and right-hemisphere pial surface files, in fsLR space.
    lh_wm_surf, rh_wm_surf : :obj:`str`
        Left- and right-hemisphere smoothed white matter surface files, in fsLR space.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # spheres to use for warping mesh files to fsLR space
                'lh_subject_sphere',
                'rh_subject_sphere',
                # fsnative mesh files to warp
                'lh_pial_surf',
                'rh_pial_surf',
                'lh_wm_surf',
                'rh_wm_surf',
            ],
        ),
        name='inputnode',
    )
    # Feed the standard-space pial and white matter surfaces to the outputnode for the brainsprite
    # and the HCP-surface generation workflow.
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'lh_pial_surf',
                'rh_pial_surf',
                'lh_wm_surf',
                'rh_wm_surf',
            ],
        ),
        name='outputnode',
    )

    # Warp the surfaces to space-fsLR, den-32k.
    # TODO: It would be nice to replace this for loop with MapNodes or iterables some day.
    for hemi in ['L', 'R']:
        hemi_label = f'{hemi.lower()}h'

        # Warp fsnative to fsLR
        collect_spheres = pe.Node(
            CollectRegistrationFiles(hemisphere=hemi, software=software),
            name=f'collect_registration_files_{hemi}',
            mem_gb=0.1,
            n_procs=1,
        )

        # Project the subject's sphere (fsnative) to the source-sphere (fsaverage) using the
        # fsLR/dhcpAsym-in-fsaverage
        # (fsLR or dhcpAsym vertices with coordinates on the fsaverage sphere) sphere?
        # So what's the result? The fsLR or dhcpAsym vertices with coordinates on the fsnative
        # sphere?
        project_unproject = pe.Node(
            SurfaceSphereProjectUnproject(num_threads=omp_nthreads),
            name=f'surface_sphere_project_unproject_{hemi}',
            n_procs=omp_nthreads,
        )
        workflow.connect([
            (inputnode, project_unproject, [('subject_sphere', 'in_file')]),
            (collect_spheres, project_unproject, [
                ('source_sphere', 'sphere_project_to'),
                ('sphere_to_sphere', 'sphere_unproject_from'),
            ]),
        ])  # fmt:skip

        # Resample the pial and white matter surfaces from fsnative to fsLR-32k or dhcpAsym-32k
        for surf_type in ['pial', 'wm']:
            surf_label = f'{hemi_label}_{surf_type}_surf'

            resample_to_fsLR32k = pe.Node(
                CiftiSurfaceResample(method='BARYCENTRIC', num_threads=omp_nthreads),
                name=f'resample_{surf_label}_to_fsLR32k',
                mem_gb=2,  # TODO: fix
                n_procs=omp_nthreads,
            )
            workflow.connect([
                (inputnode, resample_to_fsLR32k, [(surf_label, 'in_file')]),
                (collect_spheres, resample_to_fsLR32k, [('target_sphere', 'new_sphere')]),
                (project_unproject, resample_to_fsLR32k, [('out_file', 'current_sphere')]),
            ])  # fmt:skip

            ds_fsLR_surf = pe.Node(
                DerivativesDataSink(
                    space='fsLR',
                    den='32k',
                    extension='.surf.gii',  # the extension is taken from the in_file by default
                ),
                name=f'ds_fsLR_surf_{surf_label}',
                run_without_submitting=True,
                mem_gb=1,
            )
            workflow.connect([
                (inputnode, ds_fsLR_surf, [(surf_label, 'source_file')]),
                (resample_to_fsLR32k, ds_fsLR_surf, [(surf_label, 'in_file')]),
                (ds_fsLR_surf, outputnode, [('out_file', surf_label)]),
            ])  # fmt:skip

    return workflow


@fill_doc
def init_generate_hcp_surfaces_wf(name='generate_hcp_surfaces_wf'):
    """Generate midthickness, inflated, and very-inflated HCP-style surfaces.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.tests.tests import mock_config
            from xcp_d import config
            from xcp_d.workflows.anatomical.surface import init_generate_hcp_surfaces_wf

            with mock_config():
                wf = init_generate_hcp_surfaces_wf(name="generate_hcp_surfaces_wf")

    Parameters
    ----------
    %(name)s
        Default is "generate_hcp_surfaces_wf".

    Inputs
    ------
    name_source : :obj:`str`
        Path to the file that will be used as the source_file for datasinks.
    pial_surf : :obj:`str`
        The surface file to inflate.
    wm_surf : :obj:`str`
        The surface file to inflate.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'name_source',
                'pial_surf',
                'wm_surf',
            ],
        ),
        name='inputnode',
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=['midthickness']),
        name='outputnode',
    )

    generate_midthickness = pe.Node(
        SurfaceAverage(num_threads=config.nipype.omp_nthreads),
        name='generate_midthickness',
        mem_gb=2,
        n_procs=config.nipype.omp_nthreads,
    )
    workflow.connect([
        (inputnode, generate_midthickness, [
            ('pial_surf', 'surface_in1'),
            ('wm_surf', 'surface_in2'),
        ]),
        (generate_midthickness, outputnode, [('out_file', 'midthickness')]),
    ])  # fmt:skip

    ds_midthickness = pe.Node(
        DerivativesDataSink(
            check_hdr=False,
            space='fsLR',
            den='32k',
            desc='hcp',
            suffix='midthickness',
            extension='.surf.gii',
        ),
        name='ds_midthickness',
        run_without_submitting=False,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_midthickness, [('name_source', 'source_file')]),
        (generate_midthickness, ds_midthickness, [('out_file', 'in_file')]),
    ])  # fmt:skip

    # Generate (very-)inflated surface from standard-space midthickness surface.
    inflate_surface = pe.Node(
        SurfaceGenerateInflated(
            iterations_scale_value=0.75,
            num_threads=config.nipype.omp_nthreads,
        ),
        mem_gb=2,
        n_procs=config.nipype.omp_nthreads,
        name='inflate_surface',
    )
    workflow.connect([
        (generate_midthickness, inflate_surface, [('out_file', 'anatomical_surface_in')]),
    ])  # fmt:skip

    ds_inflated = pe.Node(
        DerivativesDataSink(
            check_hdr=False,
            space='fsLR',
            den='32k',
            desc='hcp',
            suffix='inflated',
            extension='.surf.gii',
        ),
        name='ds_inflated',
        run_without_submitting=False,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_inflated, [('name_source', 'source_file')]),
        (inflate_surface, ds_inflated, [('inflated_out_file', 'in_file')]),
    ])  # fmt:skip

    ds_vinflated = pe.Node(
        DerivativesDataSink(
            check_hdr=False,
            space='fsLR',
            den='32k',
            desc='hcp',
            suffix='vinflated',
            extension='.surf.gii',
        ),
        name='ds_vinflated',
        run_without_submitting=False,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_vinflated, [('name_source', 'source_file')]),
        (inflate_surface, ds_vinflated, [('very_inflated_out_file', 'in_file')]),
    ])  # fmt:skip

    return workflow
