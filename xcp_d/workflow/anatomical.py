# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Anatomical post-processing workflows."""
from nipype import Function, logging
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import CompositeTransformUtil  # MB
from nipype.interfaces.freesurfer import MRIsConvert
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from pkg_resources import resource_filename as pkgrf
from templateflow.api import get as get_template

from xcp_d.interfaces.ants import CompositeInvTransformUtil, ConvertTransformFile
from xcp_d.interfaces.bids import DerivativesDataSink
from xcp_d.interfaces.c3 import C3d  # TM
from xcp_d.interfaces.connectivity import ApplyTransformsx
from xcp_d.interfaces.nilearn import BinaryMath, Merge
from xcp_d.interfaces.utils import FilterUndefined
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
from xcp_d.utils.bids import get_freesurfer_dir, get_freesurfer_sphere
from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_warp_anats_to_template_wf(
    output_dir,
    input_type,
    target_space,
    omp_nthreads,
    mem_gb,
    name="warp_anats_to_template_wf",
):
    """Copy T1w and segmentation to the derivative directory.

    If necessary, this workflow will also warp the images to standard space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_warp_anats_to_template_wf
            wf = init_warp_anats_to_template_wf(
                output_dir=".",
                input_type="fmriprep",
                target_space="MNI152NLin6Asym",
                omp_nthreads=1,
                mem_gb=0.1,
                name="warp_anats_to_template_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    %(input_type)s
    target_space : str
        Target NIFTI template for T1w.
    %(omp_nthreads)s
    %(mem_gb)s
    %(name)s
        Default is "warp_anats_to_template_wf".

    Inputs
    ------
    t1w : str
        Path to the T1w file.
    t1seg : str
        Path to the T1w segmentation file.
    %(t1w_to_template)s
        We need to use MNI152NLin6Asym for the template.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w", "t1seg", "t1w_to_template"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w", "t1seg"]),
        name="outputnode",
    )

    # Split cohort out of the space for MNIInfant templates.
    cohort = None
    if "+" in target_space:
        target_space, cohort = target_space.split("+")

    template_file = str(
        get_template(template=target_space, cohort=cohort, resolution=1, desc=None, suffix="T1w")
    )

    if input_type in ("dcan", "hcp"):
        # Assume that the T1w and T1w segmentation files are in standard space,
        # but don't have the "space" entity, for the "dcan" and "hcp" derivatives.
        # This is a bug, and the converted filenames are inaccurate, so we have this
        # workaround in place.
        ds_t1w = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                extension=".nii.gz",
            ),
            name="ds_t1w",
            run_without_submitting=False,
        )

        ds_t1seg = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                extension=".nii.gz",
            ),
            name="ds_t1seg",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_t1w, [("t1w", "in_file")]),
            (inputnode, ds_t1seg, [("t1seg", "in_file")]),
        ])
        # fmt:on

    else:
        # Warp the native T1w-space T1w and T1w segmentation files to the selected standard space.
        warp_t1w_to_template = pe.Node(
            ApplyTransformsx(
                num_threads=2,
                reference_image=template_file,
                interpolation="LanczosWindowedSinc",
                input_image_type=3,
                dimension=3,
            ),
            name="warp_t1w_to_template",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, warp_t1w_to_template, [
                ("t1w", "input_image"),
                ("t1w_to_template", "transforms"),
            ]),
        ])
        # fmt:on

        warp_t1seg_to_template = pe.Node(
            ApplyTransformsx(
                num_threads=2,
                reference_image=template_file,
                interpolation="MultiLabel",
                input_image_type=3,
                dimension=3,
            ),
            name="warp_t1seg_to_template",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, warp_t1seg_to_template, [
                ("t1seg", "input_image"),
                ("t1w_to_template", "transforms"),
            ]),
        ])
        # fmt:on

        ds_t1w = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=target_space,
                cohort=cohort,
                extension=".nii.gz",
            ),
            name="ds_t1w",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect([(warp_t1w_to_template, ds_t1w, [("output_image", "in_file")])])
        # fmt:on

        ds_t1seg = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space=target_space,
                cohort=cohort,
                extension=".nii.gz",
            ),
            name="ds_t1seg",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect([(warp_t1seg_to_template, ds_t1seg, [("output_image", "in_file")])])
        # fmt:on

    # fmt:off
    workflow.connect([
        (inputnode, ds_t1w, [("t1w", "source_file")]),
        (inputnode, ds_t1seg, [("t1seg", "source_file")]),
        (ds_t1w, outputnode, [("out_file", "t1w")]),
        (ds_t1seg, outputnode, [("out_file", "t1seg")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_warp_surfaces_to_template_wf(
    fmri_dir,
    subject_id,
    output_dir,
    standard_spaces_available,
    surfaces_found,
    omp_nthreads,
    mem_gb,
    name="warp_surfaces_to_template_wf",
):
    """Warp surfaces from fsnative to standard fsLR-32k space.

    For the ``hcp`` and ``dcan`` preprocessing workflows,
    the fsLR-32k-space surfaces already exist and will simply be copied to the output directory.

    For other preprocessing workflows, the native space surfaces are present in the preprocessed
    derivatives directory (if Freesurfer was run), and must be transformed to standard space.
    The FreeSurfer derivatives must be indexed to grab sphere files needed to warp the surfaces.
    If Freesurfer derivatives are not available, then an error will be raised.

    Shapes in standard space, meshes in native space

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_warp_surfaces_to_template_wf

            standard_spaces_available = {
                "mesh": False,
                "morphometry": True,
                "shape": True,
            }
            surfaces_found = {
                "mesh": True,
                "morphometry": True,
                "shape": True,
            }
            wf = init_warp_surfaces_to_template_wf(
                fmri_dir=".",
                subject_id="01",
                output_dir=".",
                standard_spaces_available=standard_spaces_available,
                surfaces_found=surfaces_found,
                omp_nthreads=1,
                mem_gb=0.1,
                name="warp_surfaces_to_template_wf",
            )

    Everything in standard space

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_warp_surfaces_to_template_wf

            standard_spaces_available = {
                "mesh": True,
                "morphometry": True,
                "shape": True,
            }
            surfaces_found = {
                "mesh": True,
                "morphometry": True,
                "shape": True,
            }
            wf = init_warp_surfaces_to_template_wf(
                fmri_dir=".",
                subject_id="01",
                output_dir=".",
                standard_spaces_available=standard_spaces_available,
                surfaces_found=surfaces_found,
                omp_nthreads=1,
                mem_gb=0.1,
                name="warp_surfaces_to_template_wf",
            )

    Native-space meshes

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_warp_surfaces_to_template_wf

            standard_spaces_available = {
                "mesh": False,
                "morphometry": True,
                "shape": True,
            }
            surfaces_found = {
                "mesh": True,
                "morphometry": True,
                "shape": True,
            }
            wf = init_warp_surfaces_to_template_wf(
                fmri_dir=".",
                subject_id="01",
                output_dir=".",
                standard_spaces_available=standard_spaces_available,
                surfaces_found=surfaces_found,
                omp_nthreads=1,
                mem_gb=0.1,
                name="warp_surfaces_to_template_wf",
            )

    Nothing in standard space

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_warp_surfaces_to_template_wf

            standard_spaces_available = {
                "mesh": False,
                "morphometry": False,
                "shape": False,
            }
            surfaces_found = {
                "mesh": True,
                "morphometry": True,
                "shape": True,
            }
            wf = init_warp_surfaces_to_template_wf(
                fmri_dir=".",
                subject_id="01",
                output_dir=".",
                standard_spaces_available=standard_spaces_available,
                surfaces_found=surfaces_found,
                omp_nthreads=1,
                mem_gb=0.1,
                name="warp_surfaces_to_template_wf",
            )

    Meshes in standard space, no shapes

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_warp_surfaces_to_template_wf

            standard_spaces_available = {
                "mesh": True,
                "morphometry": True,
                "shape": False,
            }
            surfaces_found = {
                "mesh": True,
                "morphometry": True,
                "shape": False,
            }
            wf = init_warp_surfaces_to_template_wf(
                fmri_dir=".",
                subject_id="01",
                output_dir=".",
                standard_spaces_available=standard_spaces_available,
                surfaces_found=surfaces_found,
                omp_nthreads=1,
                mem_gb=0.1,
                name="warp_surfaces_to_template_wf",
            )

    Parameters
    ----------
    %(fmri_dir)s
    %(subject_id)s
    %(output_dir)s
    warp_to_standard : :obj:`bool`
        Whether to warp native-space surface files to standard space or not.
        If False, the files are assumed to be in standard space already.
    shapes_available : :obj:`bool`
        True if shape files (sulcal depth, sulcal curvature, and cortical thickness)
        are available. False if not.
    %(omp_nthreads)s
    %(mem_gb)s
    %(name)s
        Default is "warp_surfaces_to_template_wf".

    Inputs
    ------
    t1w_to_template_xform : str
        The transform from T1w space to template space.

        The template in question should match the volumetric space of the BOLD CIFTI files
        being processed by the main xcpd workflow.
        For example, MNI152NLin6Asym for fsLR-space CIFTIs.

        If ``warp_to_standard`` is False, this file is unused.
    template_to_t1w_xform : str
        The transform from template space to T1w space.

        The template in question should match the volumetric space of the BOLD CIFTI files
        being processed by the main xcpd workflow.
        For example, MNI152NLin6Asym for fsLR-space CIFTIs.

        If ``warp_to_standard`` is False, this file is unused.
    lh_pial_surf, rh_pial_surf : str
        Left- and right-hemisphere pial surface files.

        If ``warp_to_standard`` is False, then this file is just written out to the output
        directory and returned via outputnode for use in a brainsprite.

        If ``warp_to_standard`` is True, then it is also warped to standard space and used
        to generate HCP-style midthickness, inflated, and veryinflated surfaces.
    lh_wm_surf, rh_wm_surf : str
        Left- and right-hemisphere smoothed white matter surface files.

        If ``warp_to_standard`` is False, then this file is just written out to the output
        directory and returned via outputnode for use in a brainsprite.

        If ``warp_to_standard`` is True, then it is also warped to standard space and used
        to generate HCP-style midthickness, inflated, and veryinflated surfaces.
    lh_midthickness_surf, rh_midthickness_surf : str or None
        Left- and right-hemisphere midthickness surface files.

        If ``warp_to_standard`` is False, then this file is just written out to the output
        directory.

        If ``warp_to_standard`` is True, then this input is ignored and a replacement file
        are generated from the pial and wm files after they are warped to standard space.
    lh_inflated_surf, rh_inflated_surf : str or None
        Left- and right-hemisphere inflated surface files.

        If ``warp_to_standard`` is False, then this file is just written out to the output
        directory.

        If ``warp_to_standard`` is True, then this input is ignored and a replacement file
        are generated from the pial and wm files after they are warped to standard space.
    lh_vinflated_surf, rh_vinflated_surf : str or None
        Left- and right-hemisphere very-inflated surface files.

        If ``warp_to_standard`` is False, then this file is just written out to the output
        directory.

        If ``warp_to_standard`` is True, then this input is ignored and a replacement file
        are generated from the pial and wm files after they are warped to standard space.
    lh_sulcal_depth, rh_sulcal_depth : str or None
        Should only be a string if ``shapes_available`` is True.
    lh_sulcal_curv, rh_sulcal_curv : str or None
        Should only be a string if ``shapes_available`` is True.
    lh_cortical_thickness, rh_cortical_thickness : str or None
        Should only be a string if ``shapes_available`` is True.

    Notes
    -----
    If "hcp" or "dcan" input type, standard-space surface files will be collected from the
    converted preprocessed derivatives.
    This includes the HCP-style surfaces (midthickness, inflated, and vinflated).

    If "fmriprep" or "nibabies", surface files in fsnative space will be extracted from
    the preprocessed derivatives and will be warped to standard space.
    The HCP-style surfaces will also be generated.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                # surface mesh files
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_wm_surf",
                "rh_wm_surf",
                # optional surface morphometry files
                "lh_midthickness_surf",
                "rh_midthickness_surf",
                "lh_inflated_surf",
                "rh_inflated_surf",
                "lh_vinflated_surf",
                "rh_vinflated_surf",
                # optional surface shape files
                "lh_sulcal_depth",
                "rh_sulcal_depth",
                "lh_sulcal_curv",
                "rh_sulcal_curv",
                "lh_cortical_thickness",
                "rh_cortical_thickness",
                # transforms (only used if warp_to_standard is True)
                "t1w_to_template_xform",
                "template_to_t1w_xform",
            ],
        ),
        name="inputnode",
    )
    # Feed only the standard-space pial and white matter surfaces to the outputnode for the
    # brainsprite.
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

    apply_warps_to = {}
    if standard_spaces_available["mesh"] and not standard_spaces_available["morphometry"]:
        # Run HCP-generation workflow, then pass along to datasink/outputnode.
        apply_warps_to["mesh"] = False
        generate_morphometry = True
    elif standard_spaces_available["mesh"] and standard_spaces_available["morphometry"]:
        apply_warps_to["mesh"] = False
        generate_morphometry = False
    elif surfaces_found["mesh"] and not standard_spaces_available["mesh"]:
        # Run warp workflows, apply to mesh files, and generate HCP morphometry files.
        apply_warps_to["mesh"] = True
        generate_morphometry = True

    if surfaces_found["shape"] and not standard_spaces_available["shape"]:
        # Run warp workflows, apply to shape files.
        apply_warps_to["shape"] = True
    elif standard_spaces_available["shape"] or not surfaces_found["shape"]:
        apply_warps_to["shape"] = False

    if (
        standard_spaces_available["shape"]
        and standard_spaces_available["morphometry"]
        and standard_spaces_available["mesh"]
    ):
        # If everything's available in fsLR, then just filter and pass along to datasink.
        apply_warps_to["mesh"] = False
        apply_warps_to["shape"] = False
        generate_morphometry = False

    # Prepare to warp (at least some) surfaces to standard space
    if apply_warps_to["mesh"] or apply_warps_to["shape"]:
        get_freesurfer_dir_node = pe.Node(
            Function(
                function=get_freesurfer_dir,
                input_names=["fmri_dir"],
                output_names=["freesurfer_path"],
            ),
            name="get_freesurfer_dir_node",
        )
        get_freesurfer_dir_node.inputs.fmri_dir = fmri_dir

        # First, we create the Connectome WorkBench-compatible transform files.
        update_xform_wf = init_ants_xform_to_fsl_wf(
            mem_gb=mem_gb,
            omp_nthreads=omp_nthreads,
            name="update_xform_wf",
        )

        # fmt:off
        workflow.connect([
            (inputnode, update_xform_wf, [
                ("t1w_to_template_xform", "inputnode.t1w_to_template_xform"),
                ("template_to_t1w_xform", "inputnode.template_to_t1w_xform"),
            ]),
        ])
        # fmt:on

    for hemi in ["L", "R"]:
        hemi_label = f"{hemi.lower()}h"

        if generate_morphometry:
            standard_space_meshes = pe.Node(
                niu.IdentityInterface(fields=["pial_surf", "wm_surf"]),
                name=f"{hemi_label}_standard_space_meshes",
            )

            # Generate and output HCP-style surface files
            generate_hcp_surfaces_wf = init_generate_hcp_surfaces_wf(
                output_dir=output_dir,
                mem_gb=mem_gb,
                omp_nthreads=omp_nthreads,
                name=f"{hemi_label}_generate_hcp_surfaces_wf",
            )

            # fmt:off
            workflow.connect([
                (inputnode, generate_hcp_surfaces_wf, [
                    (f"{hemi_label}_pial_surf", "inputnode.name_source"),
                ]),
                (standard_space_meshes, generate_hcp_surfaces_wf, [
                    ("pial_surf", "inputnode.pial_surf"),
                    ("wm_surf", "inputnode.wm_surf"),
                ]),
            ])
            # fmt:on

        if not apply_warps_to["mesh"] or not apply_warps_to["shape"]:
            # If there are any standard-space surface files available from the preprocessing
            # derivatives, we will pass those on to the postprocessing derivatives without
            # modification.
            standard_space_surfaces_holder = pe.Node(
                niu.IdentityInterface(fields=["standard_space_surfaces"]),
                name=f"{hemi_label}_standard_space_surfaces_holder",
            )

            # The DataSink will fail if source_file or in_file is undefined/None,
            # so we must filter out any undefined inputs.
            filter_out_missing_surfaces = pe.Node(
                FilterUndefined(),
                name=f"{hemi_label}_filter_out_missing_surfaces",
            )

            # fmt:off
            workflow.connect([
                (standard_space_surfaces_holder, filter_out_missing_surfaces, [
                    ("standard_space_surfaces", "inlist"),
                ]),
            ])
            # fmt:on

            # Write out standard-space surfaces to output directory
            ds_standard_space_surfaces = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                ),
                name=f"ds_standard_space_surfaces_{hemi_label}",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["in_file", "source_file"],
            )

            # fmt:off
            workflow.connect([
                (filter_out_missing_surfaces, ds_standard_space_surfaces, [
                    ("outlist", "in_file"),
                    ("outlist", "source_file"),
                ])
            ])
            # fmt:on

        if not apply_warps_to["mesh"] and not apply_warps_to["shape"]:
            # All mesh and shape files get passed on to the datasink, unmodified.
            collect_surfaces_already_in_std = pe.Node(
                niu.Merge(5 if generate_morphometry else 8),
                name=f"collect_surfaces_already_in_std_{hemi_label}",
            )

            # fmt:off
            workflow.connect([
                (inputnode, collect_surfaces_already_in_std, [
                    (f"{hemi_label}_pial_surf", "in1"),
                    (f"{hemi_label}_wm_surf", "in2"),
                    (f"{hemi_label}_sulcal_depth", "in3"),
                    (f"{hemi_label}_sulcal_curv", "in4"),
                    (f"{hemi_label}_cortical_thickness", "in5"),
                ]),
                (inputnode, outputnode, [
                    (f"{hemi_label}_pial_surf", f"{hemi_label}_pial_surf"),
                    (f"{hemi_label}_wm_surf", f"{hemi_label}_wm_surf"),
                ]),
                (collect_surfaces_already_in_std, standard_space_surfaces_holder, [
                    ("out", "standard_space_surfaces"),
                ]),
            ])
            # fmt:on

            if generate_morphometry:
                # fmt:off
                workflow.connect([
                    (inputnode, standard_space_meshes, [
                        (f"{hemi_label}_pial_surf", "pial_surf"),
                        (f"{hemi_label}_wm_surf", "wm_surf"),
                    ]),
                ])
                # fmt:on
            else:
                # Just feed the existing files forward to the datasink.
                # fmt:off
                workflow.connect([
                    (inputnode, collect_surfaces_already_in_std, [
                        (f"{hemi_label}_midthickness_surf", "in6"),
                        (f"{hemi_label}_inflated_surf", "in7"),
                        (f"{hemi_label}_vinflated_surf", "in8"),
                    ]),
                ])
                # fmt:on

        else:
            # Some warps must be applied, so prepare the warping workflows.
            apply_transforms_wf = init_warp_one_hemisphere_wf(
                hemisphere=hemi,
                mem_gb=mem_gb,
                omp_nthreads=omp_nthreads,
                name=f"{hemi_label}_apply_transforms_wf",
            )
            apply_transforms_wf.inputs.inputnode.participant_id = subject_id

            # fmt:off
            workflow.connect([
                (get_freesurfer_dir_node, apply_transforms_wf, [
                    ("freesurfer_path", "inputnode.freesurfer_path"),
                ]),
                (update_xform_wf, apply_transforms_wf, [
                    ("outputnode.merged_warpfield", "inputnode.merged_warpfield"),
                    ("outputnode.merged_inv_warpfield", "inputnode.merged_inv_warpfield"),
                    ("outputnode.world_xform", "inputnode.world_xform"),
                ]),
            ])
            # fmt:on

            if apply_warps_to["mesh"] and not apply_warps_to["shape"]:
                # Warp mesh files, generate morphometry files,
                # and pass shape files along to datasink
                n_native_to_warp = 2
                splits = [3]  # shapes
            elif apply_warps_to["shape"] and not apply_warps_to["mesh"]:
                # Warp shape files and pass mesh files along to datasink
                n_native_to_warp = 3
                splits = [1, 1]  # pial, white
            else:
                # Warp both mesh and shape files
                n_native_to_warp = 5
                splits = [1, 1, 3]  # pial, white, shapes

            # Place the surfaces in a single node that will feed into the transform workflow.
            collect_surfaces_to_warp = pe.Node(
                niu.Merge(n_native_to_warp),
                name=f"collect_surfaces_to_warp_{hemi_label}",
            )

            # fmt:off
            workflow.connect([
                (collect_surfaces_to_warp, apply_transforms_wf, [
                    ("out", "inputnode.hemi_files"),
                ]),
            ])
            # fmt:on

            # Split up the surfaces
            # NOTE: Must match order of collect_surfaces
            split_up_warped_surfaces = pe.Node(
                niu.Split(
                    splits=splits,
                    squeeze=True,
                ),
                name=f"split_up_warped_surfaces_{hemi_label}",
            )

            # fmt:off
            workflow.connect([
                (apply_transforms_wf, split_up_warped_surfaces, [
                    ("outputnode.warped_hemi_files", "inlist"),
                ]),
            ])
            # fmt:on

            if apply_warps_to["mesh"] and not apply_warps_to["shape"]:
                # Warp mesh files, generate morphometry files,
                # and pass shape files along to datasink
                # NOTE: Must match order of split_up_surfaces_fsLR_32k.
                # NOTE: generate_morphometry is already True here.
                # fmt:off
                workflow.connect([
                    (inputnode, collect_surfaces_to_warp, [
                        (f"{hemi_label}_pial_surf", "in1"),
                        (f"{hemi_label}_wm_surf", "in2"),
                    ]),
                    (split_up_warped_surfaces, standard_space_meshes, [
                        ("out1", "pial_surf"),
                        ("out2", "wm_surf"),
                    ]),
                ])
                # fmt:on
            elif apply_warps_to["shape"] and not apply_warps_to["mesh"]:
                # Warp shape files and pass mesh files along to datasink
                # fmt:off
                workflow.connect([
                    (inputnode, collect_surfaces_to_warp, [
                        (f"{hemi_label}_sulcal_depth", "in1"),
                        (f"{hemi_label}_sulcal_curv", "in2"),
                        (f"{hemi_label}_cortical_thickness", "in3"),
                    ]),
                    (inputnode, standard_space_meshes, [
                        (f"{hemi_label}_pial_surf", "pial_surf"),
                        (f"{hemi_label}_wm_surf", "wm_surf"),
                    ]),
                ])
                # fmt:on
            else:
                # Warp both mesh and shape files
                # NOTE: generate_morphometry is already True here.
                # fmt:off
                workflow.connect([
                    (inputnode, collect_surfaces_to_warp, [
                        (f"{hemi_label}_pial_surf", "in1"),
                        (f"{hemi_label}_wm_surf", "in2"),
                        (f"{hemi_label}_sulcal_depth", "in3"),
                        (f"{hemi_label}_sulcal_curv", "in4"),
                        (f"{hemi_label}_cortical_thickness", "in5"),
                    ]),
                    (split_up_warped_surfaces, standard_space_meshes, [
                        ("out1", "pial_surf"),
                        ("out2", "wm_surf"),
                    ]),
                ])
                # fmt:on

            ds_warped_standard_space_surfaces = pe.MapNode(
                DerivativesDataSink(
                    base_directory=output_dir,
                    space="fsLR",
                    den="32k",
                ),
                name=f"ds_warped_standard_space_surfaces_{hemi_label}",
                run_without_submitting=True,
                mem_gb=1,
                iterfield=["in_file", "source_file"],
            )

            # fmt:off
            workflow.connect([
                (collect_surfaces_to_warp, ds_warped_standard_space_surfaces, [
                    ("out", "source_file"),
                ]),
                (apply_transforms_wf, ds_warped_standard_space_surfaces, [
                    ("outputnode.warped_hemi_files", "in_file"),
                ]),
            ])
            # fmt:on

    return workflow


@fill_doc
def init_generate_hcp_surfaces_wf(
    output_dir,
    mem_gb,
    omp_nthreads,
    name="generate_hcp_surfaces_wf",
):
    """Generate midthickness, inflated, and very-inflated HCP-style surfaces.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_generate_hcp_surfaces_wf
            wf = init_generate_hcp_surfaces_wf(
                output_dir=".",
                mem_gb=0.1,
                omp_nthreads=1,
                name="generate_hcp_surfaces_wf",
            )

    Parameters
    ----------
    output_dir
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "generate_hcp_surfaces_wf".

    Inputs
    ------
    name_source : str
        Path to the file that will be used as the source_file for datasinks.
    pial_surf : str
        The surface file to inflate.
    wm_surf : str
        The surface file to inflate.
    """
    workflow = Workflow(name=name)

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

    generate_midthickness = pe.Node(
        SurfaceAverage(),
        name="generate_midthickness",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, generate_midthickness, [
            ("pial_surf", "surface_in1"),
            ("wm_surf", "surface_in2"),
        ]),
    ])
    # fmt:on

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

    # fmt:off
    workflow.connect([
        (inputnode, ds_midthickness, [("name_source", "source_file")]),
        (generate_midthickness, ds_midthickness, [("out_file", "in_file")]),
    ])
    # fmt:on

    # Generate (very-)inflated surface from standard-space midthickness surface.
    inflate_surface = pe.Node(
        SurfaceGenerateInflated(iterations_scale_value=0.75),
        mem_gb=mem_gb,
        omp_nthreads=omp_nthreads,
        name="inflate_surface",
    )

    # fmt:off
    workflow.connect([
        (generate_midthickness, inflate_surface, [("out_file", "anatomical_surface_in")]),
    ])
    # fmt:on

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

    # fmt:off
    workflow.connect([
        (inputnode, ds_inflated, [("name_source", "source_file")]),
        (inflate_surface, ds_inflated, [("inflated_out_file", "in_file")]),
    ])
    # fmt:on

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

    # fmt:off
    workflow.connect([
        (inputnode, ds_vinflated, [("name_source", "source_file")]),
        (inflate_surface, ds_vinflated, [("very_inflated_out_file", "in_file")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_ants_xform_to_fsl_wf(mem_gb, omp_nthreads, name="ants_xform_to_fsl_wf"):
    """Modify ANTS-style fMRIPrep transforms to work with Connectome Workbench/FSL FNIRT.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_ants_xform_to_fsl_wf
            wf = init_ants_xform_to_fsl_wf(
                mem_gb=0.1,
                omp_nthreads=1,
                name="ants_xform_to_fsl_wf",
            )

    Parameters
    ----------
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "ants_xform_to_fsl_wf".

    Inputs
    ------
    t1w_to_template_xform
        ANTS/fMRIPrep-style H5 transform from T1w image to template.
    template_to_t1w_xform
        ANTS/fMRIPrep-style H5 transform from template to T1w image.

    Outputs
    -------
    world_xform
        TODO: Add description.
    merged_warpfield
        TODO: Add description.
    merged_inv_warpfield
        TODO: Add description.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w_to_template_xform", "template_to_t1w_xform"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["world_xform", "merged_warpfield", "merged_inv_warpfield"]),
        name="outputnode",
    )

    # Now we can start the actual workflow.
    # use ANTs CompositeTransformUtil to separate the .h5 into affine and warpfield xfms
    disassemble_h5 = pe.Node(
        CompositeTransformUtil(
            process="disassemble",
            output_prefix="T1w_to_MNI152NLin6Asym",
        ),
        name="disassemble_h5",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )  # MB

    # fmt:off
    workflow.connect([
        (inputnode, disassemble_h5, [("t1w_to_template_xform", "in_file")]),
    ])
    # fmt:on

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

    # fmt:off
    workflow.connect([
        (inputnode, disassemble_h5_inv, [("template_to_t1w_xform", "in_file")]),
    ])
    # fmt:on

    # convert affine from ITK binary to txt
    convert_ants_transform = pe.Node(
        ConvertTransformFile(dimension=3),
        name="convert_ants_transform",
    )

    # fmt:off
    workflow.connect([
        (disassemble_h5, convert_ants_transform, [("affine_transform", "in_transform")]),
    ])
    # fmt:on

    # change xfm type from "AffineTransform" to "MatrixOffsetTransformBase"
    # since wb_command doesn't recognize "AffineTransform"
    # (AffineTransform is a subclass of MatrixOffsetTransformBase
    # which makes this okay to do AFAIK)
    change_xfm_type = pe.Node(ChangeXfmType(), name="change_xfm_type")

    # fmt:off
    workflow.connect([
        (convert_ants_transform, change_xfm_type, [("out_transform", "in_transform")]),
    ])
    # fmt:on

    # convert affine xfm to "world" so it works with -surface-apply-affine
    convert_xfm2world = pe.Node(
        ConvertAffine(fromwhat="itk", towhat="world"),
        name="convert_xfm2world",
    )

    # fmt:off
    workflow.connect([
        (change_xfm_type, convert_xfm2world, [("out_transform", "in_file")]),
    ])
    # fmt:on

    # use C3d to separate the combined warpfield xfm into x, y, and z components
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

    # fmt:off
    workflow.connect([
        (disassemble_h5, get_xyz_components, [("displacement_field", "in_file")]),
        (disassemble_h5_inv, get_inv_xyz_components, [("displacement_field", "in_file")]),
    ])
    # fmt:on

    # select x-component after separating warpfield above
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

    # select y-component
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

    # select z-component
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

    # fmt:off
    workflow.connect([
        (get_xyz_components, select_x_component, [("out_files", "inlist")]),
        (get_xyz_components, select_y_component, [("out_files", "inlist")]),
        (get_xyz_components, select_z_component, [("out_files", "inlist")]),
        (get_inv_xyz_components, select_inv_x_component, [("out_files", "inlist")]),
        (get_inv_xyz_components, select_inv_y_component, [("out_files", "inlist")]),
        (get_inv_xyz_components, select_inv_z_component, [("out_files", "inlist")]),
    ])
    # fmt:on

    # reverse y-component of the warpfield
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

    # fmt:off
    workflow.connect([
        (select_y_component, reverse_y_component, [("out", "in_file")]),
        (select_inv_y_component, reverse_inv_y_component, [("out", "in_file")]),
    ])
    # fmt:on

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

    # fmt:off
    workflow.connect([
        (select_x_component, collect_new_components, [("out", "in1")]),
        (reverse_y_component, collect_new_components, [("out_file", "in2")]),
        (select_z_component, collect_new_components, [("out", "in3")]),
        (select_inv_x_component, collect_new_inv_components, [("out", "in1")]),
        (reverse_inv_y_component, collect_new_inv_components, [("out_file", "in2")]),
        (select_inv_z_component, collect_new_inv_components, [("out", "in3")]),
    ])
    # fmt:on

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

    # fmt:off
    workflow.connect([
        (collect_new_components, remerge_warpfield, [("out", "in_files")]),
        (collect_new_inv_components, remerge_inv_warpfield, [("out", "in_files")]),
        (convert_xfm2world, outputnode, [("out_file", "world_xform")]),
        (remerge_warpfield, outputnode, [("out_file", "merged_warpfield")]),
        (remerge_inv_warpfield, outputnode, [("out_file", "merged_inv_warpfield")]),
    ])
    # fmt:on

    return workflow


@fill_doc
def init_warp_one_hemisphere_wf(hemisphere, mem_gb, omp_nthreads, name="warp_one_hemisphere_wf"):
    """Apply transforms to warp one hemisphere's surface files into standard space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_warp_one_hemisphere_wf
            wf = init_warp_one_hemisphere_wf(
                hemisphere="L",
                mem_gb=0.1,
                omp_nthreads=1,
                name="warp_one_hemisphere_wf",
            )

    Parameters
    ----------
    hemisphere : {"L", "R"}
    %(mem_gb)s
    %(omp_nthreads)s
    %(name)s
        Default is "warp_one_hemisphere_wf".

    Inputs
    ------
    hemi_files : list of str
        A list of surface files for the requested hemisphere, in fsnative space.
    world_xform
    merged_warpfield
    merged_inv_warpfield
    freesurfer_path
        Path to FreeSurfer derivatives. Used to load the subject's sphere file.
    participant_id

    Outputs
    -------
    warped_hemi_files
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "hemi_files",
                "world_xform",
                "merged_warpfield",
                "merged_inv_warpfield",
                "freesurfer_path",
                "participant_id",
            ],
        ),
        name="inputnode",
    )

    # Load the fsaverage-164k sphere
    # NOTE: Why do we need the fsaverage mesh?
    fsaverage_mesh = str(
        get_template(
            template="fsaverage",
            space=None,
            hemi=hemisphere,
            density="164k",
            desc=None,
            suffix="sphere",
        )
    )

    # NOTE: Can we upload these to templateflow?
    fs_hemisphere_to_fsLR = pkgrf(
        "xcp_d",
        (
            f"data/standard_mesh_atlases/fs_{hemisphere}/"
            f"fs_{hemisphere}-to-fs_LR_fsaverage.{hemisphere}_LR.spherical_std."
            f"164k_fs_{hemisphere}.surf.gii"
        ),
    )
    get_freesurfer_sphere_node = pe.Node(
        Function(
            function=get_freesurfer_sphere,
            input_names=["freesurfer_path", "subject_id", "hemisphere"],
            output_names=["sphere_raw"],
        ),
        name="get_freesurfer_sphere_node",
    )
    get_freesurfer_sphere_node.inputs.hemisphere = hemisphere

    # fmt:off
    workflow.connect([
        (inputnode, get_freesurfer_sphere_node, [
            ("freesurfer_path", "freesurfer_path"),
            ("participant_id", "subject_id"),
        ])
    ])
    # fmt:on

    # NOTE: What does this step do?
    sphere_to_surf_gii = pe.Node(
        MRIsConvert(out_datatype="gii"),
        name="sphere_to_surf_gii",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (get_freesurfer_sphere_node, sphere_to_surf_gii, [("sphere_raw", "in_file")]),
    ])
    # fmt:on

    # NOTE: What does this step do?
    surface_sphere_project_unproject = pe.Node(
        SurfaceSphereProjectUnproject(
            sphere_project_to=fsaverage_mesh,
            sphere_unproject_from=fs_hemisphere_to_fsLR,
        ),
        name="surface_sphere_project_unproject",
    )

    # fmt:off
    workflow.connect([
        (sphere_to_surf_gii, surface_sphere_project_unproject, [("converted", "in_file")]),
    ])
    # fmt:on

    fsLR_sphere = str(
        get_template(
            template="fsLR",
            space=None,
            hemi=hemisphere,
            density="32k",
            desc=None,
            suffix="sphere",
        )
    )

    # resample the surfaces to fsLR-32k
    # NOTE: Does that mean the data are in fsLR-164k before this?
    resample_to_fsLR32k = pe.MapNode(
        CiftiSurfaceResample(
            new_sphere=fsLR_sphere,
            metric=" BARYCENTRIC ",
        ),
        name="resample_to_fsLR32k",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
        iterfield=["in_file"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, resample_to_fsLR32k, [
            ("hemi_files", "in_file"),
        ]),
        (surface_sphere_project_unproject, resample_to_fsLR32k, [
            ("out_file", "current_sphere"),
        ]),
    ])
    # fmt:on

    # apply affine to 32k surfs
    # NOTE: What does this step do? Aren't the data in fsLR-32k from resample_to_fsLR32k?
    apply_affine_to_fsLR32k = pe.MapNode(
        ApplyAffine(),
        name="apply_affine_to_fsLR32k",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
        iterfield=["in_file"],
    )

    # fmt:off
    workflow.connect([
        (resample_to_fsLR32k, apply_affine_to_fsLR32k, [("out_file", "in_file")]),
        (inputnode, apply_affine_to_fsLR32k, [("world_xform", "affine")]),
    ])
    # fmt:on

    # apply FNIRT-format warpfield
    # NOTE: What does this step do?
    apply_warpfield_to_fsLR32k = pe.MapNode(
        ApplyWarpfield(),
        name="apply_warpfield_to_fsLR32k",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
        iterfield=["in_file"],
    )

    # fmt:off
    workflow.connect([
        (inputnode, apply_warpfield_to_fsLR32k, [
            ("merged_warpfield", "forward_warp"),
            ("merged_inv_warpfield", "warpfield"),
        ]),
        (apply_affine_to_fsLR32k, apply_warpfield_to_fsLR32k, [
            ("out_file", "in_file"),
        ]),
    ])
    # fmt:on

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["warped_hemi_files"]),
        name="outputnode",
    )

    # fmt:off
    workflow.connect([
        (apply_warpfield_to_fsLR32k, outputnode, [("out_file", "warped_hemi_files")]),
    ])
    # fmt:on

    return workflow
