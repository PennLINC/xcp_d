# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Workflows for processing surface anatomical files."""

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import CompositeTransformUtil  # MB
from nipype.interfaces.freesurfer import MRIsConvert
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from xcp_d import config
from xcp_d.interfaces.ants import CompositeInvTransformUtil, ConvertTransformFile
from xcp_d.interfaces.bids import CollectRegistrationFiles, DerivativesDataSink
from xcp_d.interfaces.c3 import C3d  # TM
from xcp_d.interfaces.nilearn import BinaryMath, Merge
from xcp_d.interfaces.workbench import (  # MB,TM
    ApplyAffine,
    ApplyWarpfield,
    ChangeXfmType,
    CiftiSurfaceResample,
    ConvertAffine,
    SurfaceAverage,
    SurfaceGenerateInflated,
    SurfaceSphereProjectUnproject,
)
from xcp_d.utils.doc import fill_doc
from xcp_d.workflows.anatomical.outputs import init_copy_inputs_to_outputs_wf
from xcp_d.workflows.anatomical.plotting import init_brainsprite_figures_wf

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_postprocess_surfaces_wf(
    mesh_available,
    standard_space_mesh,
    morphometry_files,
    t1w_available,
    t2w_available,
    software,
    name="postprocess_surfaces_wf",
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
    %(template_to_anat_xfm)s
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
    output_dir = config.execution.xcp_d_dir
    omp_nthreads = config.nipype.omp_nthreads

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t2w",
                "anat_to_template_xfm",
                "template_to_anat_xfm",
                "lh_subject_sphere",
                "rh_subject_sphere",
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
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
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "lh_midthickness",
                "rh_midthickness",
            ],
        ),
        name="outputnode",
    )
    workflow.add_nodes([outputnode])  # outputnode may not be used

    workflow.__desc__ = ""

    if abcc_qc and mesh_available:
        # Plot the white and pial surfaces on the brain in a brainsprite figure.
        brainsprite_wf = init_brainsprite_figures_wf(
            t1w_available=t1w_available,
            t2w_available=t2w_available,
        )
        workflow.connect([
            (inputnode, brainsprite_wf, [
                ("t1w", "inputnode.t1w"),
                ("t2w", "inputnode.t2w"),
            ]),
        ])  # fmt:skip

        if (not process_surfaces) or (mesh_available and standard_space_mesh):
            # Use original surfaces for brainsprite.
            # For fMRIPrep derivatives, this will be the native-space surfaces.
            # For DCAN/HCP derivatives, it will be standard-space surfaces.
            workflow.connect([
                (inputnode, brainsprite_wf, [
                    ("lh_pial_surf", "inputnode.lh_pial_surf"),
                    ("rh_pial_surf", "inputnode.rh_pial_surf"),
                    ("lh_wm_surf", "inputnode.lh_wm_surf"),
                    ("rh_wm_surf", "inputnode.rh_wm_surf"),
                ]),
            ])  # fmt:skip

    if not process_surfaces:
        # Return early, as all other steps require process_surfaces.
        return workflow

    if morphometry_files or (mesh_available and standard_space_mesh):
        # At least some surfaces are already in fsLR space and must be copied,
        # without modification, to the output directory.
        copy_std_surfaces_to_datasink = init_copy_inputs_to_outputs_wf(
            name="copy_std_surfaces_to_datasink",
        )

    if morphometry_files:
        workflow.__desc__ += (
            " fsLR-space morphometry surfaces were copied from the preprocessing derivatives to "
            "the XCP-D derivatives."
        )
        for morphometry_file in morphometry_files:
            workflow.connect([
                (inputnode, copy_std_surfaces_to_datasink, [
                    (morphometry_file, f"inputnode.{morphometry_file}"),
                ]),
            ])  # fmt:skip

    if mesh_available:
        workflow.__desc__ += (
            " HCP-style midthickness, inflated, and very-inflated surfaces were generated from "
            "the white-matter and pial surface meshes."
        )
        # Generate and output HCP-style surface files.
        hcp_surface_wfs = {
            hemi: init_generate_hcp_surfaces_wf(name=f"{hemi}_generate_hcp_surfaces_wf")
            for hemi in ["lh", "rh"]
        }
        workflow.connect([
            (inputnode, hcp_surface_wfs["lh"], [("lh_pial_surf", "inputnode.name_source")]),
            (inputnode, hcp_surface_wfs["rh"], [("rh_pial_surf", "inputnode.name_source")]),
            (hcp_surface_wfs["lh"], outputnode, [("outputnode.midthickness", "lh_midthickness")]),
            (hcp_surface_wfs["rh"], outputnode, [("outputnode.midthickness", "rh_midthickness")]),
        ])  # fmt:skip

    if mesh_available and standard_space_mesh:
        workflow.__desc__ += (
            " All surface files were already in fsLR space, and were copied to the output "
            "directory."
        )
        # Mesh files are already in fsLR.
        workflow.connect([
            (inputnode, copy_std_surfaces_to_datasink, [
                ("lh_pial_surf", "inputnode.lh_pial_surf"),
                ("rh_pial_surf", "inputnode.rh_pial_surf"),
                ("lh_wm_surf", "inputnode.lh_wm_surf"),
                ("rh_wm_surf", "inputnode.rh_wm_surf"),
            ]),
            (inputnode, hcp_surface_wfs["lh"], [
                ("lh_pial_surf", "inputnode.pial_surf"),
                ("lh_wm_surf", "inputnode.wm_surf"),
            ]),
            (inputnode, hcp_surface_wfs["rh"], [
                ("rh_pial_surf", "inputnode.pial_surf"),
                ("rh_wm_surf", "inputnode.wm_surf"),
            ]),
        ])  # fmt:skip

    elif mesh_available:
        workflow.__desc__ += " fsnative-space surfaces were then warped to fsLR space."
        # Mesh files are in fsnative and must be warped to fsLR.
        warp_surfaces_to_template_wf = init_warp_surfaces_to_template_wf(
            output_dir=output_dir,
            software=software,
            omp_nthreads=omp_nthreads,
            name="warp_surfaces_to_template_wf",
        )
        workflow.connect([
            (inputnode, warp_surfaces_to_template_wf, [
                ("lh_subject_sphere", "inputnode.lh_subject_sphere"),
                ("rh_subject_sphere", "inputnode.rh_subject_sphere"),
                ("lh_pial_surf", "inputnode.lh_pial_surf"),
                ("rh_pial_surf", "inputnode.rh_pial_surf"),
                ("lh_wm_surf", "inputnode.lh_wm_surf"),
                ("rh_wm_surf", "inputnode.rh_wm_surf"),
                ("anat_to_template_xfm", "inputnode.anat_to_template_xfm"),
                ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
            ]),
            (warp_surfaces_to_template_wf, hcp_surface_wfs["lh"], [
                ("outputnode.lh_pial_surf", "inputnode.pial_surf"),
                ("outputnode.lh_wm_surf", "inputnode.wm_surf"),
            ]),
            (warp_surfaces_to_template_wf, hcp_surface_wfs["rh"], [
                ("outputnode.rh_pial_surf", "inputnode.pial_surf"),
                ("outputnode.rh_wm_surf", "inputnode.wm_surf"),
            ]),
        ])  # fmt:skip

        if abcc_qc:
            # Use standard-space T1w and surfaces for brainsprite.
            workflow.connect([
                (warp_surfaces_to_template_wf, brainsprite_wf, [
                    ("outputnode.lh_pial_surf", "inputnode.lh_pial_surf"),
                    ("outputnode.rh_pial_surf", "inputnode.rh_pial_surf"),
                    ("outputnode.lh_wm_surf", "inputnode.lh_wm_surf"),
                    ("outputnode.rh_wm_surf", "inputnode.rh_wm_surf"),
                ]),
            ])  # fmt:skip

    elif not morphometry_files:
        raise ValueError(
            "No surfaces found. Surfaces are required if `--warp-surfaces-native2std` is enabled."
        )

    return workflow


@fill_doc
def init_warp_surfaces_to_template_wf(
    output_dir,
    software,
    omp_nthreads,
    name="warp_surfaces_to_template_wf",
):
    """Transform surfaces from native to standard fsLR-32k space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.anatomical.surface import init_warp_surfaces_to_template_wf

            wf = init_warp_surfaces_to_template_wf(
                output_dir=".",
                software="FreeSurfer",
                omp_nthreads=1,
                name="warp_surfaces_to_template_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    software : {"MCRIBS", "FreeSurfer"}
        The software used to generate the surfaces.
    %(omp_nthreads)s
    %(name)s
        Default is "warp_surfaces_to_template_wf".

    Inputs
    ------
    %(anat_to_template_xfm)s
        The template in question should match the volumetric space of the BOLD CIFTI files
        being processed by the main xcpd workflow.
        For example, MNI152NLin6Asym for fsLR-space CIFTIs.
    %(template_to_anat_xfm)s
        The template in question should match the volumetric space of the BOLD CIFTI files
        being processed by the main xcpd workflow.
        For example, MNI152NLin6Asym for fsLR-space CIFTIs.
    lh_subject_sphere, rh_subject_sphere : :obj:`str`
        Left- and right-hemisphere sphere registration files.
    lh_pial_surf, rh_pial_surf : :obj:`str`
        Left- and right-hemisphere pial surface files in fsnative space.
    lh_wm_surf, rh_wm_surf : :obj:`str`
        Left- and right-hemisphere smoothed white matter surface files in fsnative space.

    Outputs
    -------
    lh_pial_surf, rh_pial_surf : :obj:`str`
        Left- and right-hemisphere pial surface files, in standard space.
    lh_wm_surf, rh_wm_surf : :obj:`str`
        Left- and right-hemisphere smoothed white matter surface files, in standard space.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # transforms
                "anat_to_template_xfm",
                "template_to_anat_xfm",
                # surfaces
                "lh_subject_sphere",
                "rh_subject_sphere",
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
            ],
        ),
        name="inputnode",
    )
    # Feed the standard-space pial and white matter surfaces to the outputnode for the brainsprite
    # and the HCP-surface generation workflow.
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
            ],
        ),
        name="outputnode",
    )

    # Warp the surfaces to space-fsLR, den-32k.
    # First, we create the Connectome WorkBench-compatible transform files.
    update_xfm_wf = init_ants_xfm_to_fsl_wf(
        mem_gb=1,
        omp_nthreads=omp_nthreads,
        name="update_xfm_wf",
    )
    workflow.connect([
        (inputnode, update_xfm_wf, [
            ("anat_to_template_xfm", "inputnode.anat_to_template_xfm"),
            ("template_to_anat_xfm", "inputnode.template_to_anat_xfm"),
        ]),
    ])  # fmt:skip

    # TODO: It would be nice to replace this for loop with MapNodes or iterables some day.
    for hemi in ["L", "R"]:
        hemi_label = f"{hemi.lower()}h"

        # Place the surfaces in a single node.
        collect_surfaces = pe.Node(
            niu.Merge(2),
            name=f"collect_surfaces_{hemi_label}",
        )
        # NOTE: Must match order of split_up_surfaces_fsLR_32k.
        workflow.connect([
            (inputnode, collect_surfaces, [
                (f"{hemi_label}_pial_surf", "in1"),
                (f"{hemi_label}_wm_surf", "in2"),
            ]),
        ])  # fmt:skip

        apply_transforms_wf = init_warp_one_hemisphere_wf(
            hemisphere=hemi,
            software=software,
            mem_gb=2,
            omp_nthreads=omp_nthreads,
            name=f"{hemi_label}_apply_transforms_wf",
        )
        workflow.connect([
            (inputnode, apply_transforms_wf, [
                (f"{hemi_label}_subject_sphere", "inputnode.subject_sphere"),
            ]),
            (update_xfm_wf, apply_transforms_wf, [
                ("outputnode.merged_warpfield", "inputnode.merged_warpfield"),
                ("outputnode.merged_inv_warpfield", "inputnode.merged_inv_warpfield"),
                ("outputnode.world_xfm", "inputnode.world_xfm"),
            ]),
            (collect_surfaces, apply_transforms_wf, [("out", "inputnode.hemi_files")]),
        ])  # fmt:skip

        # Split up the surfaces
        # NOTE: Must match order of collect_surfaces
        split_up_surfaces_fsLR_32k = pe.Node(
            niu.Split(
                splits=[
                    1,  # pial
                    1,  # wm
                ],
                squeeze=True,
            ),
            name=f"split_up_surfaces_fsLR_32k_{hemi_label}",
        )
        workflow.connect([
            (apply_transforms_wf, split_up_surfaces_fsLR_32k, [
                ("outputnode.warped_hemi_files", "inlist"),
            ]),
            (split_up_surfaces_fsLR_32k, outputnode, [
                ("out1", f"{hemi_label}_pial_surf"),
                ("out2", f"{hemi_label}_wm_surf"),
            ]),
        ])  # fmt:skip

        ds_standard_space_surfaces = pe.MapNode(
            DerivativesDataSink(
                base_directory=output_dir,
                space="fsLR",
                den="32k",
                extension=".surf.gii",  # the extension is taken from the in_file by default
            ),
            name=f"ds_standard_space_surfaces_{hemi_label}",
            run_without_submitting=True,
            mem_gb=1,
            iterfield=["in_file", "source_file"],
        )
        workflow.connect([
            (collect_surfaces, ds_standard_space_surfaces, [("out", "source_file")]),
            (apply_transforms_wf, ds_standard_space_surfaces, [
                ("outputnode.warped_hemi_files", "in_file"),
            ]),
        ])  # fmt:skip

    return workflow


@fill_doc
def init_generate_hcp_surfaces_wf(name="generate_hcp_surfaces_wf"):
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

    output_dir = config.execution.xcp_d_dir
    omp_nthreads = config.nipype.omp_nthreads

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "name_source",
                "pial_surf",
                "wm_surf",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["midthickness"]),
        name="outputnode",
    )

    generate_midthickness = pe.Node(
        SurfaceAverage(),
        name="generate_midthickness",
        mem_gb=2,
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (inputnode, generate_midthickness, [
            ("pial_surf", "surface_in1"),
            ("wm_surf", "surface_in2"),
        ]),
        (generate_midthickness, outputnode, [("out_file", "midthickness")]),
    ])  # fmt:skip

    ds_midthickness = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
            space="fsLR",
            den="32k",
            desc="hcp",
            suffix="midthickness",
            extension=".surf.gii",
        ),
        name="ds_midthickness",
        run_without_submitting=False,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_midthickness, [("name_source", "source_file")]),
        (generate_midthickness, ds_midthickness, [("out_file", "in_file")]),
    ])  # fmt:skip

    # Generate (very-)inflated surface from standard-space midthickness surface.
    inflate_surface = pe.Node(
        SurfaceGenerateInflated(iterations_scale_value=0.75),
        mem_gb=2,
        omp_nthreads=omp_nthreads,
        name="inflate_surface",
    )
    workflow.connect([
        (generate_midthickness, inflate_surface, [("out_file", "anatomical_surface_in")]),
    ])  # fmt:skip

    ds_inflated = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
            space="fsLR",
            den="32k",
            desc="hcp",
            suffix="inflated",
            extension=".surf.gii",
        ),
        name="ds_inflated",
        run_without_submitting=False,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_inflated, [("name_source", "source_file")]),
        (inflate_surface, ds_inflated, [("inflated_out_file", "in_file")]),
    ])  # fmt:skip

    ds_vinflated = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            check_hdr=False,
            space="fsLR",
            den="32k",
            desc="hcp",
            suffix="vinflated",
            extension=".surf.gii",
        ),
        name="ds_vinflated",
        run_without_submitting=False,
        mem_gb=2,
    )
    workflow.connect([
        (inputnode, ds_vinflated, [("name_source", "source_file")]),
        (inflate_surface, ds_vinflated, [("very_inflated_out_file", "in_file")]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_ants_xfm_to_fsl_wf(mem_gb, omp_nthreads, name="ants_xfm_to_fsl_wf"):
    """Modify ANTS-style fMRIPrep transforms to work with Connectome Workbench/FSL FNIRT.

    XXX: Does this only work if the template is MNI152NLin6Asym?

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.anatomical.surface import init_ants_xfm_to_fsl_wf

            wf = init_ants_xfm_to_fsl_wf(
                mem_gb=0.1,
                omp_nthreads=1,
                name="ants_xfm_to_fsl_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "ants_xfm_to_fsl_wf".

    Inputs
    ------
    anat_to_template_xfm
        ANTS/fMRIPrep-style H5 transform from anatomical image to template.
    template_to_anat_xfm
        ANTS/fMRIPrep-style H5 transform from template to anatomical image.

    Outputs
    -------
    world_xfm
        The affine portion of the volumetric anatomical-to-template transform,
        in NIfTI (world) format.
    merged_warpfield
        The warpfield portion of the volumetric anatomical-to-template transform,
        in FSL (FNIRT) format.
    merged_inv_warpfield
        The warpfield portion of the volumetric template-to-anatomical transform,
        in FSL (FNIRT) format.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["anat_to_template_xfm", "template_to_anat_xfm"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["world_xfm", "merged_warpfield", "merged_inv_warpfield"]),
        name="outputnode",
    )

    # Now we can start the actual workflow.
    # Use ANTs CompositeTransformUtil to separate the .h5 into affine and warpfield xfms.
    disassemble_h5 = pe.Node(
        CompositeTransformUtil(
            process="disassemble",
            output_prefix="T1w_to_MNI152NLin6Asym",
        ),
        name="disassemble_h5",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([(inputnode, disassemble_h5, [("anat_to_template_xfm", "in_file")])])

    # Nipype's CompositeTransformUtil assumes a certain file naming and
    # concatenation order of xfms which does not work for the inverse .h5,
    # so we use our modified class, "CompositeInvTransformUtil"
    disassemble_h5_inv = pe.Node(
        CompositeInvTransformUtil(
            process="disassemble",
            output_prefix="MNI152NLin6Asym_to_T1w",
        ),
        name="disassemble_h5_inv",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([(inputnode, disassemble_h5_inv, [("template_to_anat_xfm", "in_file")])])

    # Convert anat-to-template affine from ITK binary to txt
    convert_ants_xfm = pe.Node(
        ConvertTransformFile(dimension=3),
        name="convert_ants_xfm",
    )
    workflow.connect([(disassemble_h5, convert_ants_xfm, [("affine_transform", "in_transform")])])

    # Change xfm type from "AffineTransform" to "MatrixOffsetTransformBase"
    # since wb_command doesn't recognize "AffineTransform"
    # (AffineTransform is a subclass of MatrixOffsetTransformBase which prob makes this okay to do)
    change_xfm_type = pe.Node(ChangeXfmType(), name="change_xfm_type")
    workflow.connect([(convert_ants_xfm, change_xfm_type, [("out_transform", "in_transform")])])

    # Convert affine xfm to "world" so it works with -surface-apply-affine
    convert_xfm2world = pe.Node(
        ConvertAffine(fromwhat="itk", towhat="world"),
        name="convert_xfm2world",
    )
    workflow.connect([(change_xfm_type, convert_xfm2world, [("out_transform", "in_file")])])

    # Use C3d to separate the combined warpfield xfm into x, y, and z components
    get_xyz_components = pe.Node(
        C3d(
            is_4d=True,
            multicomp_split=True,
            out_files=["e1.nii.gz", "e2.nii.gz", "e3.nii.gz"],
        ),
        name="get_xyz_components",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    get_inv_xyz_components = pe.Node(
        C3d(
            is_4d=True,
            multicomp_split=True,
            out_files=["e1inv.nii.gz", "e2inv.nii.gz", "e3inv.nii.gz"],
        ),
        name="get_inv_xyz_components",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (disassemble_h5, get_xyz_components, [("displacement_field", "in_file")]),
        (disassemble_h5_inv, get_inv_xyz_components, [("displacement_field", "in_file")]),
    ])  # fmt:skip

    # Select x-component after separating warpfield above
    select_x_component = pe.Node(
        niu.Select(index=[0]),
        name="select_x_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    select_inv_x_component = pe.Node(
        niu.Select(index=[0]),
        name="select_inv_x_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # Select y-component
    select_y_component = pe.Node(
        niu.Select(index=[1]),
        name="select_y_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    select_inv_y_component = pe.Node(
        niu.Select(index=[1]),
        name="select_inv_y_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # Select z-component
    select_z_component = pe.Node(
        niu.Select(index=[2]),
        name="select_z_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    select_inv_z_component = pe.Node(
        niu.Select(index=[2]),
        name="select_inv_z_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (get_xyz_components, select_x_component, [("out_files", "inlist")]),
        (get_xyz_components, select_y_component, [("out_files", "inlist")]),
        (get_xyz_components, select_z_component, [("out_files", "inlist")]),
        (get_inv_xyz_components, select_inv_x_component, [("out_files", "inlist")]),
        (get_inv_xyz_components, select_inv_y_component, [("out_files", "inlist")]),
        (get_inv_xyz_components, select_inv_z_component, [("out_files", "inlist")]),
    ])  # fmt:skip

    # Reverse y-component of the warpfield
    # (need to do this when converting a warpfield from ANTs to FNIRT format
    # for use with wb_command -surface-apply-warpfield)
    reverse_y_component = pe.Node(
        BinaryMath(expression="img * -1"),
        name="reverse_y_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    reverse_inv_y_component = pe.Node(
        BinaryMath(expression="img * -1"),
        name="reverse_inv_y_component",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (select_y_component, reverse_y_component, [("out", "in_file")]),
        (select_inv_y_component, reverse_inv_y_component, [("out", "in_file")]),
    ])  # fmt:skip

    # Collect new warpfield components in individual nodes
    collect_new_components = pe.Node(
        niu.Merge(3),
        name="collect_new_components",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    collect_new_inv_components = pe.Node(
        niu.Merge(3),
        name="collect_new_inv_components",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (select_x_component, collect_new_components, [("out", "in1")]),
        (reverse_y_component, collect_new_components, [("out_file", "in2")]),
        (select_z_component, collect_new_components, [("out", "in3")]),
        (select_inv_x_component, collect_new_inv_components, [("out", "in1")]),
        (reverse_inv_y_component, collect_new_inv_components, [("out_file", "in2")]),
        (select_inv_z_component, collect_new_inv_components, [("out", "in3")]),
    ])  # fmt:skip

    # Merge warpfield components in FSL FNIRT format, with the reversed y-component from above
    remerge_warpfield = pe.Node(
        Merge(),
        name="remerge_warpfield",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    remerge_inv_warpfield = pe.Node(
        Merge(),
        name="remerge_inv_warpfield",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([
        (collect_new_components, remerge_warpfield, [("out", "in_files")]),
        (collect_new_inv_components, remerge_inv_warpfield, [("out", "in_files")]),
        (convert_xfm2world, outputnode, [("out_file", "world_xfm")]),
        (remerge_warpfield, outputnode, [("out_file", "merged_warpfield")]),
        (remerge_inv_warpfield, outputnode, [("out_file", "merged_inv_warpfield")]),
    ])  # fmt:skip

    return workflow


@fill_doc
def init_warp_one_hemisphere_wf(
    hemisphere,
    software,
    mem_gb,
    omp_nthreads,
    name="warp_one_hemisphere_wf",
):
    """Apply transforms to warp one hemisphere's surface files into standard space.

    Basically, the resulting surface files will have the same vertices as the standard-space
    surfaces, but the coordinates/mesh of those vertices will be the subject's native-space
    coordinates/mesh.
    This way we can visualize surface statistical maps on the subject's unique morphology
    (sulci, gyri, etc.).

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.anatomical.surface import init_warp_one_hemisphere_wf

            wf = init_warp_one_hemisphere_wf(
                hemisphere="L",
                software="FreeSurfer",
                mem_gb=0.1,
                omp_nthreads=1,
                name="warp_one_hemisphere_wf",
            )

    Parameters
    ----------
    hemisphere : {"L", "R"}
    software : {"MCRIBS", "FreeSurfer"}
        The software used for the segmentation.
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "warp_one_hemisphere_wf".

    Inputs
    ------
    hemi_files : list of str
        A list of surface files (i.e., pial and white matter) for the requested hemisphere,
        in fsnative space.
    world_xfm
        The affine portion of the volumetric anatomical-to-template transform,
        in NIfTI (world) format.
    merged_warpfield
        The warpfield portion of the volumetric anatomical-to-template transform,
        in FSL (FNIRT) format.
    merged_inv_warpfield
        The warpfield portion of the volumetric template-to-anatomical transform,
        in FSL (FNIRT) format.
    subject_sphere
        The subject's fsnative sphere registration file to fsaverage
        (sphere.reg in FreeSurfer parlance).
        The file contains the vertices from the subject's fsnative sphere,
        with coordinates that are aligned to the fsaverage sphere.

    Outputs
    -------
    warped_hemi_files : list of str
        The ``hemi_files`` warped from fsnative space to standard space.

    Notes
    -----
    Steps:

    1. Collect the registration files needed for the warp.
    2. Convert the subject's sphere to a GIFTI file.
        - This step is unnecessary since fMRIPrep and Nibabies already write out a GIFTI file.
    3. Project the subject's fsnative-in-fsaverage sphere to a high-resolution
       target-sphere-in-fsaverage-space.
       This retains the subject's fsnative sphere's resolution and vertices
       (e.g., 120079 vertices), but the coordinates are now aligned to the target sphere's space.
        - For Freesurfer, this is the fsLR-164k-in-fsaverage sphere.
        - For MCRIBS, this is the dhcpAsym-41k-in-fsaverage sphere.
        - Nibabies and fMRIPrep do this already to produce the
          space-<fsLR|dhcpAsym>_desc-reg_sphere files, so XCP-D could directly use those and skip
          this step.
    4. Apply the warped sphere from the previous step to warp the pial and white matter surfaces
       to the target space. This includes downsampling to 32k.
        - For Freesurfer, this means the coordinates for these files are fsLR-32k.
        - For MCRIBS, this means the coordinates for these files are dhcpAsym-32k.
    5. Apply the anatomical-to-template affine transform to the 32k surfaces.
    6. Apply the anatomical-to-template warpfield to the 32k surfaces.
       This and the previous step make it so you can overlay the pial and white matter surfaces
       on the associated volumetric template (e.g., for XCP-D's brainsprite).
        - This important thing is that the volumetric template must match the template space
          used here.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "hemi_files",
                "world_xfm",
                "merged_warpfield",
                "merged_inv_warpfield",
                "subject_sphere",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["warped_hemi_files"]),
        name="outputnode",
    )

    collect_registration_files = pe.Node(
        CollectRegistrationFiles(hemisphere=hemisphere, software=software),
        name="collect_registration_files",
        mem_gb=0.1,
        n_procs=1,
    )

    # XXX: Given that fMRIPrep and Nibabies write out the subject spheres as surf.gii,
    # I think this is unnecessary.
    sphere_to_surf_gii = pe.Node(
        MRIsConvert(out_datatype="gii"),
        name="sphere_to_surf_gii",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    workflow.connect([(inputnode, sphere_to_surf_gii, [("subject_sphere", "in_file")])])

    # NOTE: What does this step do?
    # Project the subject's sphere (fsnative) to the source-sphere (fsaverage) using the
    # fsLR/dhcpAsym-in-fsaverage
    # (fsLR or dhcpAsym vertices with coordinates on the fsaverage sphere) sphere?
    # So what's the result? The fsLR or dhcpAsym vertices with coordinates on the fsnative sphere?
    surface_sphere_project_unproject = pe.Node(
        SurfaceSphereProjectUnproject(),
        name="surface_sphere_project_unproject",
    )
    workflow.connect([
        (collect_registration_files, surface_sphere_project_unproject, [
            ("source_sphere", "sphere_project_to"),
            ("sphere_to_sphere", "sphere_unproject_from"),
        ]),
        (sphere_to_surf_gii, surface_sphere_project_unproject, [("converted", "in_file")]),
    ])  # fmt:skip

    # Resample the pial and white matter surfaces from fsnative to fsLR-32k or dhcpAsym-32k
    resample_to_fsLR32k = pe.MapNode(
        CiftiSurfaceResample(method="BARYCENTRIC"),
        name="resample_to_fsLR32k",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
        iterfield=["in_file"],
    )
    workflow.connect([
        (inputnode, resample_to_fsLR32k, [("hemi_files", "in_file")]),
        (collect_registration_files, resample_to_fsLR32k, [("target_sphere", "new_sphere")]),
        (surface_sphere_project_unproject, resample_to_fsLR32k, [("out_file", "current_sphere")]),
    ])  # fmt:skip

    # Apply FLIRT-format anatomical-to-template affine transform to 32k surfs
    # NOTE: What does this step do? Aren't the data in fsLR/dhcpAsym-32k from resample_to_fsLR32k?
    apply_affine_to_fsLR32k = pe.MapNode(
        ApplyAffine(),
        name="apply_affine_to_fsLR32k",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
        iterfield=["in_file"],
    )
    workflow.connect([
        (inputnode, apply_affine_to_fsLR32k, [("world_xfm", "affine")]),
        (resample_to_fsLR32k, apply_affine_to_fsLR32k, [("out_file", "in_file")]),
    ])  # fmt:skip

    # Apply FNIRT-format (forward) anatomical-to-template warpfield
    # NOTE: What does this step do?
    apply_warpfield_to_fsLR32k = pe.MapNode(
        ApplyWarpfield(),
        name="apply_warpfield_to_fsLR32k",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
        iterfield=["in_file"],
    )
    workflow.connect([
        (inputnode, apply_warpfield_to_fsLR32k, [
            ("merged_warpfield", "forward_warp"),
            ("merged_inv_warpfield", "warpfield"),
        ]),
        (apply_affine_to_fsLR32k, apply_warpfield_to_fsLR32k, [("out_file", "in_file")]),
        (apply_warpfield_to_fsLR32k, outputnode, [("out_file", "warped_hemi_files")]),
    ])  # fmt:skip

    return workflow
