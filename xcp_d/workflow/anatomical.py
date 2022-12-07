# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Anatomical post-processing workflows."""
import glob
import os

from nipype import logging
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import CompositeTransformUtil  # MB
from nipype.interfaces.ants.resampling import ApplyTransforms  # TM
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

LOGGER = logging.getLogger("nipype.workflow")


@fill_doc
def init_t1w_wf(
    output_dir,
    input_type,
    omp_nthreads,
    mem_gb,
    name="t1w_wf",
):
    """Copy T1w and segmentation to the derivative directory.

    If necessary, this workflow will also warp the images to standard space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_t1w_wf
            wf = init_t1w_wf(
                output_dir=".",
                input_type="fmriprep",
                omp_nthreads=1,
                mem_gb=0.1,
                name="t1w_wf",
            )

    Parameters
    ----------
    %(output_dir)s
    %(input_type)s
    %(omp_nthreads)s
    %(mem_gb)s
    %(name)s
        Default is "t1w_wf".

    Inputs
    ------
    t1w : str
        Path to the T1w file.
    t1w_seg : str
        Path to the T1w segmentation file.
    %(t1w_to_template)s
        We need to use MNI152NLin6Asym for the template.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w", "t1w_seg", "t1w_to_template"]),
        name="inputnode",
    )

    # MNI92FSL = pkgrf("xcp_d", "data/transform/FSL2MNI9Composite.h5")
    mnitemplate = str(
        get_template(template="MNI152NLin6Asym", resolution=2, desc=None, suffix="T1w")
    )
    # mnitemplatemask = str(
    #     get_template(
    #         template="MNI152NLin6Asym", resolution=2, desc="brain", suffix="mask"
    #     )
    # )

    if input_type in ("dcan", "hcp"):
        ds_t1wmni = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                extension=".nii.gz",
            ),
            name="ds_t1wmni",
            run_without_submitting=False,
        )

        ds_t1wseg = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                extension=".nii.gz",
            ),
            name="ds_t1wseg",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect(
            [
                (inputnode, ds_t1wmni, [("t1w", "in_file")]),
                (inputnode, ds_t1wseg, [("t1w_seg", "in_file")]),
            ]
        )
        # fmt:on
    else:
        # #TM: need to replace MNI92FSL xfm with the correct
        # xfm from the MNI output space of fMRIPrep/NiBabies
        # (MNI2009, MNIInfant, or for cifti output MNI152NLin6Asym)
        # to MNI152NLin6Asym.
        t1w_transform = pe.Node(
            ApplyTransformsx(
                num_threads=2,
                reference_image=mnitemplate,
                interpolation="LanczosWindowedSinc",
                input_image_type=3,
                dimension=3,
            ),
            name="t1w_transform",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        seg_transform = pe.Node(
            ApplyTransformsx(
                num_threads=2,
                reference_image=mnitemplate,
                interpolation="MultiLabel",
                input_image_type=3,
                dimension=3,
            ),
            name="seg_transform",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        ds_t1wmni = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space="MNI152NLin6Asym",
                extension=".nii.gz",
            ),
            name="ds_t1wmni",
            run_without_submitting=False,
        )

        ds_t1wseg = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space="MNI152NLin6Asym",
                extension=".nii.gz",
            ),
            name="ds_t1wseg",
            run_without_submitting=False,
        )

        # fmt:off
        workflow.connect(
            [
                (inputnode, t1w_transform, [("t1w", "input_image"),
                                            ("t1w_to_template", "transforms")]),
                (inputnode, seg_transform, [("t1w_seg", "input_image"),
                                            ("t1w_to_template", "transforms")]),
                (t1w_transform, ds_t1wmni, [("output_image", "in_file")]),
                (seg_transform, ds_t1wseg, [("output_image", "in_file")]),
            ]
        )
        # fmt:on

    # fmt:off
    workflow.connect(
        [
            (inputnode, ds_t1wmni, [("t1w", "source_file")]),
            (inputnode, ds_t1wseg, [("t1w_seg", "source_file")]),
        ]
    )
    # fmt:on

    return workflow


@fill_doc
def init_anatomical_wf(
    layout,
    fmri_dir,
    subject_id,
    output_dir,
    input_type,
    warp_to_standard,
    omp_nthreads,
    mem_gb,
    name="anatomical_wf",
):
    """Transform surfaces from native to standard fsLR-32k space.

    For the ``hcp`` and ``dcan`` preprocessing workflows,
    the fsLR-32k space surfaces already exist, and will simply be copied to the output directory.

    For other preprocessing workflows, the native space surfaces are present in the Freesurfer
    directory (if Freesurfer was run), and must be transformed to standard space.
    If Freesurfer derivatives are not available, then a warning will be raised and
    no output files will be generated.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.anatomical import init_anatomical_wf
            wf = init_anatomical_wf(
                layout=None,
                omp_nthreads=1,
                fmri_dir=".",
                subject_id="01",
                output_dir=".",
                input_type="fmriprep",
                mem_gb=0.1,
                name="anatomical_wf",
            )

    Parameters
    ----------
    %(layout)s
    %(fmri_dir)s
    %(subject_id)s
    %(output_dir)s
    %(input_type)s
    %(omp_nthreads)s
    %(mem_gb)s
    %(name)s
        Default is "anatomical_wf".

    Inputs
    ------
    t1w : str
        Path to the T1w file.

    Notes
    -----
    If "hcp" or "dcan" input type, pre-generated surface files will be collected from the
    converted preprocessed derivatives.
    However, these derivatives do not include HCP-style surfaces.

    If "fmriprep" or "nibabies", surface files in fsnative space will be extracted from the
    associated Freesurfer directory (if available), and warped to fsLR space.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "t1w",
                "t1w_to_template_xform",
                "template_to_t1w_xform",
                "lh_inflated_surf",
                "rh_inflated_surf",
                "lh_midthickness_surf",
                "rh_midthickness_surf",
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_smoothwm_surf",
                "rh_smoothwm_surf",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "lh_inflated_surf",
                "rh_inflated_surf",
                "lh_midthickness_surf",
                "rh_midthickness_surf",
                "lh_pial_surf",
                "rh_pial_surf",
                "lh_smoothwm_surf",
                "rh_smoothwm_surf",
            ],
        ),
        name="outputnode",
    )

    merge_files_to_list = pe.Node(niu.Merge(8))

    if not warp_to_standard:
        derivative_entities = {}

        # fmt:off
        workflow.connect([
            (inputnode, merge_files_to_list, [
                ("lh_inflated_surf", "in1"),
                ("rh_inflated_surf", "in2"),
                ("lh_midthickness_surf", "in3"),
                ("rh_midthickness_surf", "in4"),
                ("lh_pial_surf", "in5"),
                ("rh_pial_surf", "in6"),
                ("lh_smoothwm_surf", "in7"),
                ("rh_smoothwm_surf", "in8"),
            ]),
            (inputnode, outputnode, [
                ("lh_inflated_surf", "lh_inflated_surf"),
                ("rh_inflated_surf", "rh_inflated_surf"),
                ("lh_midthickness_surf", "lh_midthickness_surf"),
                ("rh_midthickness_surf", "rh_midthickness_surf"),
                ("lh_pial_surf", "lh_pial_surf"),
                ("rh_pial_surf", "rh_pial_surf"),
                ("lh_smoothwm_surf", "lh_smoothwm_surf"),
                ("rh_smoothwm_surf", "rh_smoothwm_surf"),
            ]),
        ])
        # fmt:on

    else:
        # Warp the surfaces to space-fsLR, den-32k
        derivative_entities = {"space": "fsLR", "den": "32k"}

        # Load necessary files
        fs_std_mesh_L = str(
            get_template(
                template="fsaverage",
                hemi="L",
                space=None,
                density="164k",
                desc=None,
                suffix="sphere",
            )
        )
        fs_std_mesh_R = str(
            get_template(
                template="fsaverage",
                hemi="R",
                space=None,
                density="164k",
                desc=None,
                suffix="sphere",
            )
        )
        fs_L2fsLR = pkgrf(
            "xcp_d",
            (
                "data/standard_mesh_atlases/fs_L/"
                "fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii"
            ),
        )
        fs_R2fsLR = pkgrf(
            "xcp_d",
            (
                "data/standard_mesh_atlases/fs_R/"
                "fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.surf.gii"
            ),
        )
        lh_sphere_fsLR = str(
            get_template(template="fsLR", hemi="L", density="32k", suffix="sphere")[0]
        )
        rh_sphere_fsLR = str(
            get_template(template="fsLR", hemi="R", density="32k", suffix="sphere")[0]
        )

        # Find freesurfer directory
        freesurfer_paths = glob.glob(os.path.join(fmri_dir, "sourcedata/*freesurfer*"))
        if len(freesurfer_paths) == 0:
            freesurfer_paths = glob.glob(os.path.join(os.path.dirname(fmri_dir), "*freesurfer*"))

        if len(freesurfer_paths) > 0:
            freesurfer_path = freesurfer_paths[0]
        else:
            freesurfer_path = None

        if not freesurfer_path:
            raise ValueError("No FreeSurfer derivatives found.")

        # get sphere surfaces to be converted
        if not subject_id.startswith("sub-"):
            subject_id = "sub-" + subject_id

        lh_sphere_raw = os.path.join(freesurfer_path, subject_id, "surf/lh.sphere.reg")
        rh_sphere_raw = os.path.join(freesurfer_path, subject_id, "surf/rh.sphere.reg")

        update_xform_wf = init_update_xform_wf(name="update_xform_wf")

        # convert spheres (from FreeSurfer surf dir) to gifti
        lh_sphere_raw_mris = pe.Node(
            MRIsConvert(out_datatype="gii", in_file=lh_sphere_raw),
            name="lh_sphere_raw_mris",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # MB
        rh_sphere_raw_mris = pe.Node(
            MRIsConvert(out_datatype="gii", in_file=rh_sphere_raw),
            name="rh_sphere_raw_mris",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # MB

        # apply affine to native surfs
        lh_native_apply_affine = pe.Node(
            ApplyAffine(),
            name="lh_native_apply_affine",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM
        lh_native_apply_affine.iterables = (
            "in_file",
            [L_midthick_surf, L_pial_surf, L_wm_surf],
        )

        # fmt:off
        workflow.connect([
            (update_xform_wf, lh_native_apply_affine, [("outputnode.world_xfm", "affine")]),
        ])
        # fmt:on

        rh_native_apply_affine = pe.Node(
            ApplyAffine(),
            name="rh_native_apply_affine",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM
        rh_native_apply_affine.iterables = (
            "in_file",
            [R_midthick_surf, R_pial_surf, R_wm_surf],
        )

        # fmt:off
        workflow.connect([
            (update_xform_wf, rh_native_apply_affine, [("outputnode.world_xfm", "affine")]),
        ])
        # fmt:on

        # apply FNIRT-format warpfield
        lh_native_apply_warpfield = pe.Node(
            ApplyWarpfield(),
            name="lh_native_apply_warpfield",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM
        rh_native_apply_warpfield = pe.Node(
            ApplyWarpfield(),
            name="rh_native_apply_warpfield",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM

        # fmt:off
        workflow.connect([
            (lh_native_apply_affine, lh_native_apply_warpfield, [("out_file", "in_file")]),
            (update_xform_wf, lh_native_apply_warpfield, [
                ("outputnode.merged_warpfield", "forward_warp"),
            ]),
            (update_xform_wf, lh_native_apply_warpfield, [
                ("outputnode.merged_inv_warpfield", "warpfield"),
            ]),
            (rh_native_apply_affine, rh_native_apply_warpfield, [("out_file", "in_file")]),
            (update_xform_wf, rh_native_apply_warpfield, [
                ("outputnode.merged_warpfield", "forward_warp"),
            ]),
            (update_xform_wf, rh_native_apply_warpfield, [
                ("outputnode.merged_inv_warpfield", "warpfield"),
            ]),
        ])
        # fmt:on

        surface_sphere_project_unproject_lh = pe.Node(
            SurfaceSphereProjectUnproject(
                sphere_project_to=fs_std_mesh_L,
                sphere_unproject_from=fs_L2fsLR,
            ),
            name="surface_sphere_project_unproject_lh",
        )
        surface_sphere_project_unproject_rh = pe.Node(
            SurfaceSphereProjectUnproject(
                sphere_project_to=fs_std_mesh_R,
                sphere_unproject_from=fs_R2fsLR,
            ),
            name="surface_sphere_project_unproject_rh",
        )

        # fmt:off
        workflow.connect([
            (lh_sphere_raw_mris, surface_sphere_project_unproject_lh, [("converted", "in_file")]),
            (rh_sphere_raw_mris, surface_sphere_project_unproject_rh, [("converted", "in_file")]),
        ])
        # fmt:on

        # resample the mid, pial, wm surfs to fsLR32k
        lh_32k_resample_wf = pe.Node(
            CiftiSurfaceResample(
                new_sphere=lh_sphere_fsLR,
                metric=" BARYCENTRIC ",
            ),
            name="lh_32k_resample_wf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        lh_32k_resample_wf.iterables = (
            "in_file",
            [L_midthick_surf, L_pial_surf, L_wm_surf],
        )
        rh_32k_resample_wf = pe.Node(
            CiftiSurfaceResample(
                new_sphere=rh_sphere_fsLR,
                metric=" BARYCENTRIC ",
            ),
            name="rh_32k_resample_wf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        rh_32k_resample_wf.iterables = (
            "in_file",
            [R_midthick_surf, R_pial_surf, R_wm_surf],
        )

        # fmt:off
        workflow.connect([
            (surface_sphere_project_unproject_lh, lh_32k_resample_wf, [
                ("out_file", "current_sphere"),
            ]),
            (surface_sphere_project_unproject_rh, rh_32k_resample_wf, [
                ("out_file", "current_sphere"),
            ]),
        ])
        # fmt:on

        # apply affine to 32k surfs
        lh_32k_apply_affine = pe.Node(
            ApplyAffine(),
            name="lh_32k_apply_affine",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM
        rh_32k_apply_affine = pe.Node(
            ApplyAffine(),
            name="rh_32k_apply_affine",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM

        # fmt:off
        workflow.connect([
            (lh_32k_resample_wf, lh_32k_apply_affine, [("out_file", "in_file")]),
            (update_xform_wf, lh_32k_apply_affine, [("outputnode.world_xfm", "affine")]),
            (rh_32k_resample_wf, rh_32k_apply_affine, [("out_file", "in_file")]),
            (update_xform_wf, rh_32k_apply_affine, [("outputnode.world_xfm", "affine")]),
        ])
        # fmt:on

        # apply FNIRT-format warpfield
        lh_32k_apply_warpfield = pe.Node(
            ApplyWarpfield(),
            name="lh_32k_apply_warpfield",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM
        rh_32k_apply_warpfield = pe.Node(
            ApplyWarpfield(),
            name="rh_32k_apply_warpfield",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM

        # fmt:off
        workflow.connect([
            (update_xform_wf, lh_32k_apply_warpfield, [
                ("outputnode.merged_warpfield", "forward_warp"),
                ("outputnode.merged_inv_warpfield", "warpfield"),
            ]),
            (lh_32k_apply_affine, lh_32k_apply_warpfield, [("out_file", "in_file")]),
            (update_xform_wf, rh_32k_apply_warpfield, [
                ("outputnode.merged_warpfield", "forward_warp"),
                ("outputnode.merged_inv_warpfield", "warpfield"),
            ]),
            (rh_32k_apply_affine, rh_32k_apply_warpfield, [("out_file", "in_file")]),
        ])
        # fmt:on

        join_lh_32k_warped_surfs = pe.JoinNode(
            niu.Merge(1),
            name="join_lh_32k_warped_surfs",
            joinsource="lh_32k_resample_wf",
            joinfield="in1",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        join_rh_32k_warped_surfs = pe.JoinNode(
            niu.Merge(1),
            name="join_rh_32k_warped_surfs",
            joinsource="rh_32k_resample_wf",
            joinfield="in1",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (lh_32k_apply_warpfield, join_lh_32k_warped_surfs, [("out_file", "in1")]),
            (rh_32k_apply_warpfield, join_rh_32k_warped_surfs, [("out_file", "in1")]),
        ])
        # fmt:on

        select_lh_32k_midthick_surf = pe.Node(
            niu.Select(index=[0]),
            name="select_lh_32k_midthick_surf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        select_lh_32k_pial_surf = pe.Node(
            niu.Select(index=[1]),
            name="select_lh_32k_pial_surf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        select_lh_32k_wm_surf = pe.Node(
            niu.Select(index=[2]),
            name="select_lh_32k_wm_surf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        select_rh_32k_midthick_surf = pe.Node(
            niu.Select(index=[0]),
            name="select_rh_32k_midthick_surf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        select_rh_32k_pial_surf = pe.Node(
            niu.Select(index=[1]),
            name="select_rh_32k_pial_surf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )
        select_rh_32k_wm_surf = pe.Node(
            niu.Select(index=[2]),
            name="select_rh_32k_wm_surf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (join_lh_32k_warped_surfs, select_lh_32k_midthick_surf, [("out", "inlist")]),
            (join_lh_32k_warped_surfs, select_lh_32k_pial_surf, [("out", "inlist")]),
            (join_lh_32k_warped_surfs, select_lh_32k_wm_surf, [("out", "inlist")]),
            (join_rh_32k_warped_surfs, select_rh_32k_midthick_surf, [("out", "inlist")]),
            (join_rh_32k_warped_surfs, select_rh_32k_pial_surf, [("out", "inlist")]),
            (join_rh_32k_warped_surfs, select_rh_32k_wm_surf, [("out", "inlist")]),
        ])
        # fmt:on

        # write report node
        ds_wmLsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                space="fsLR",
                den="32k",
                hemi="L",
                suffix="smoothwm",
                extension=".surf.gii",
            ),
            name="ds_wmLsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )
        ds_wmRsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                space="fsLR",
                den="32k",
                hemi="R",
                suffix="smoothwm",
                extension=".surf.gii",
            ),
            name="ds_wmRsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        ds_pialLsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                space="fsLR",
                den="32k",
                hemi="L",
                suffix="pial",
                extension=".surf.gii",
            ),
            name="ds_pialLsurf_wf",
            run_without_submitting=True,
            mem_gb=2,
        )
        ds_pialRsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                space="fsLR",
                den="32k",
                hemi="R",
                suffix="pial",
                extension=".surf.gii",
            ),
            name="ds_pialRsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        ds_midLsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                space="fsLR",
                den="32k",
                suffix="midthickness",
                extension=".surf.gii",
                hemi="L",
            ),
            name="ds_midLsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        ds_midRsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                space="fsLR",
                den="32k",
                hemi="R",
                suffix="midthickness",
                extension=".surf.gii",
            ),
            name="ds_midRsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_midLsurf_wf, [("lh_midthickness_surface", "source_file")]),
            (select_lh_32k_midthick_surf, ds_midLsurf_wf, [("out", "in_file")]),
            (inputnode, ds_pialLsurf_wf, [("lh_pial_surface", "source_file")]),
            (select_lh_32k_pial_surf, ds_pialLsurf_wf, [("out", "in_file")]),
            (inputnode, ds_wmLsurf_wf, [("lh_wm_surface", "source_file")]),
            (select_lh_32k_wm_surf, ds_wmLsurf_wf, [("out", "in_file")]),
        ])
        workflow.connect([
            (inputnode, ds_midRsurf_wf, [("rh_midthickness_surface", "source_file")]),
            (select_rh_32k_midthick_surf, ds_midRsurf_wf, [("out", "in_file")]),
            (inputnode, ds_pialRsurf_wf, [("rh_pial_surface", "source_file")]),
            (select_rh_32k_pial_surf, ds_pialRsurf_wf, [("out", "in_file")]),
            (inputnode, ds_wmRsurf_wf, [("rh_wm_surface", "source_file")]),
            (select_rh_32k_wm_surf, ds_wmRsurf_wf, [("out", "in_file")]),
        ])
        # fmt:on

        # make "HCP-style" native midthickness and inflated
        lh_native_hcpmidthick_wf = pe.Node(
            SurfaceAverage(),
            name="lh_native_hcpmidthick_wf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, lh_native_hcpmidthick_wf, [
                ("lh_pial_surface", "surface_in1"),
                ("lh_wm_surface", "surface_in2"),
            ])
        ])
        # fmt:on

        rh_native_hcpmidthick_wf = pe.Node(
            SurfaceAverage(),
            name="rh_native_hcpmidthick_wf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (inputnode, rh_native_hcpmidthick_wf, [
                ("rh_pial_surface", "surface_in1"),
                ("rh_wm_surface", "surface_in2"),
            ])
        ])
        # fmt:on

        lh_32k_hcpmidthick_resample_wf = pe.Node(
            CiftiSurfaceResample(new_sphere=lh_sphere_fsLR, metric=" BARYCENTRIC "),
            name="lh_32k_hcpmidthick_resample_wf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (surface_sphere_project_unproject_lh, lh_32k_hcpmidthick_resample_wf, [
                ("out_file", "current_sphere"),
            ]),
            (lh_native_hcpmidthick_wf, lh_32k_hcpmidthick_resample_wf, [
                ("out_file", "in_file"),
            ]),
        ])
        # fmt:on

        rh_32k_hcpmidthick_resample_wf = pe.Node(
            CiftiSurfaceResample(new_sphere=rh_sphere_fsLR, metric=" BARYCENTRIC "),
            name="rh_32k_hcpmidthick_resample_wf",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        # fmt:off
        workflow.connect([
            (surface_sphere_project_unproject_rh, rh_32k_hcpmidthick_resample_wf, [
                ("out_file", "current_sphere"),
            ]),
            (rh_native_hcpmidthick_wf, rh_32k_hcpmidthick_resample_wf, [
                ("out_file", "in_file"),
            ]),
        ])
        # fmt:on

        # apply affine to 32k hcpmidthick
        lh_32k_hcpmidthick_apply_affine = pe.Node(
            ApplyAffine(),
            name="lh_32k_hcpmidthick_apply_affine",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM

        # fmt:off
        workflow.connect([
            (lh_32k_hcpmidthick_resample_wf, lh_32k_hcpmidthick_apply_affine, [
                ("out_file", "in_file")
            ]),
            (update_xform_wf, lh_32k_hcpmidthick_apply_affine, [
                ("outputnode.world_xform", "affine"),
            ]),
        ])
        # fmt:on

        rh_32k_hcpmidthick_apply_affine = pe.Node(
            ApplyAffine(),
            name="rh_32k_hcpmidthick_apply_affine",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM

        # fmt:off
        workflow.connect([
            (rh_32k_hcpmidthick_resample_wf, rh_32k_hcpmidthick_apply_affine, [
                ("out_file", "in_file"),
            ]),
            (update_xform_wf, rh_32k_hcpmidthick_apply_affine, [
                ("outputnode.world_xform", "affine"),
            ]),
        ])
        # fmt:on

        # apply FNIRT-format warpfield
        lh_32k_hcpmidthick_apply_warpfield = pe.Node(
            ApplyWarpfield(),
            name="lh_32k_hcpmidthick_apply_warpfield",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM

        # fmt:off
        workflow.connect([
            (update_xform_wf, lh_32k_hcpmidthick_apply_warpfield, [
                ("outputnode.merged_warpfield", "forward_warp"),
            ]),
            (update_xform_wf, lh_32k_hcpmidthick_apply_warpfield, [
                ("outputnode.merged_inv_warpfield", "warpfield"),
            ]),
            (lh_32k_hcpmidthick_apply_affine, lh_32k_hcpmidthick_apply_warpfield, [
                ("out_file", "in_file"),
            ]),
        ])
        # fmt:on

        rh_32k_hcpmidthick_apply_warpfield = pe.Node(
            ApplyWarpfield(),
            name="rh_32k_hcpmidthick_apply_warpfield",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )  # TM

        # fmt:off
        workflow.connect([
            (update_xform_wf, rh_32k_hcpmidthick_apply_warpfield, [
                ("outputnode.merged_warpfield", "forward_warp"),
            ]),
            (update_xform_wf, rh_32k_hcpmidthick_apply_warpfield, [
                ("outputnode.merged_inv_warpfield", "warpfield"),
            ]),
            (rh_32k_hcpmidthick_apply_affine, rh_32k_hcpmidthick_apply_warpfield, [
                ("out_file", "in_file"),
            ]),
        ])
        # fmt:on

        lh_32k_hcpinflated_surf_wf = pe.Node(
            SurfaceGenerateInflated(iterations_scale_value=0.75),
            name="lh_hcpinflated_surf_wf",
        )

        # fmt:off
        workflow.connect([
            (lh_32k_hcpmidthick_apply_warpfield, lh_32k_hcpinflated_surf_wf, [
                ("out_file", "anatomical_surface_in"),
            ]),
        ])
        # fmt:on

        rh_32k_hcpinflated_surf_wf = pe.Node(
            SurfaceGenerateInflated(iterations_scale_value=0.75),
            name="rh_hcpinflated_surf_wf",
        )

        # fmt:off
        workflow.connect([
            (rh_32k_hcpmidthick_apply_warpfield, rh_32k_hcpinflated_surf_wf, [
                ("out_file", "anatomical_surface_in"),
            ]),
        ])
        # fmt:on

        ds_hcpmidLsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                space="fsLR",
                den="32k",
                hemi="L",
                desc="hcp",
                suffix="midthickness",
                extension=".surf.gii",
            ),
            name="ds_hcpmidLsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_hcpmidLsurf_wf, [
                ("lh_midthickness_surf", "source_file"),
            ]),
            (lh_32k_hcpmidthick_apply_warpfield, ds_hcpmidLsurf_wf, [
                ("out_file", "in_file"),
            ]),
        ])
        # fmt:on

        ds_hcpmidRsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                space="fsLR",
                den="32k",
                hemi="R",
                desc="hcp",
                suffix="midthickness",
                extension=".surf.gii",
            ),
            name="ds_hcpmidRsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_hcpmidRsurf_wf, [
                ("rh_midthickness_surf", "source_file"),
            ]),
            (rh_32k_hcpmidthick_apply_warpfield, ds_hcpmidRsurf_wf, [
                ("out_file", "in_file"),
            ]),
        ])
        # fmt:on

        ds_hcpinfLsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                space="fsLR",
                den="32k",
                hemi="L",
                desc="hcp",
                suffix="inflated",
                extension=".surf.gii",
            ),
            name="ds_hcpinfLsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_hcpinfLsurf_wf, [
                ("lh_inflated_surf", "source_file"),
            ]),
            (lh_32k_hcpinflated_surf_wf, ds_hcpinfLsurf_wf, [
                ("inflated_out_file", "in_file"),
            ]),
        ])
        # fmt:on

        ds_hcpinfRsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                space="fsLR",
                den="32k",
                hemi="R",
                desc="hcp",
                suffix="inflated",
                extension=".surf.gii",
            ),
            name="ds_hcpinfRsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_hcpinfRsurf_wf, [
                ("rh_inflated_surf", "source_file"),
            ]),
            (rh_32k_hcpinflated_surf_wf, ds_hcpinfRsurf_wf, [
                ("inflated_out_file", "in_file"),
            ]),
        ])
        # fmt:on

        ds_hcpveryinfLsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                space="fsLR",
                den="32k",
                hemi="L",
                desc="hcp",
                suffix="vinflated",
                extension=".surf.gii",
            ),
            name="ds_hcpveryinfLsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_hcpveryinfLsurf_wf, [
                ("lh_inflated_surf", "source_file"),
            ]),
            (lh_32k_hcpinflated_surf_wf, ds_hcpveryinfLsurf_wf, [
                ("very_inflated_out_file", "in_file"),
            ]),
        ])
        # fmt:on

        ds_hcpveryinfRsurf_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                space="fsLR",
                den="32k",
                hemi="R",
                desc="hcp",
                suffix="vinflated",
                extension=".surf.gii",
            ),
            name="ds_hcpveryinfRsurf_wf",
            run_without_submitting=False,
            mem_gb=2,
        )

        # fmt:off
        workflow.connect([
            (inputnode, ds_hcpveryinfRsurf_wf, [
                ("rh_inflated_surf", "source_file"),
            ]),
            (rh_32k_hcpinflated_surf_wf, ds_hcpveryinfRsurf_wf, [
                ("very_inflated_out_file", "in_file"),
            ]),
        ])
        # fmt:on

    # Write out standard-space surfaces to output directory
    ds_standard_space_surfaces = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            **derivative_entities,
        ),
        name="ds_standard_space_surfaces",
        run_without_submitting=True,
        mem_gb=1,
        iterfield=["in_file", "source_file"],
    )

    # fmt:off
    workflow.connect([
        (merge_files_to_list, ds_standard_space_surfaces, [
            ("out", "in_file"),
            ("out", "source_file"),
        ])
    ])
    # fmt:on

    return workflow


def init_update_xform_wf(mem_gb, omp_nthreads, name="update_xform_wf"):
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w_to_template_xform", "template_to_t1w_xform"]),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["world_xfm", "merged_warpfield", "merged_inv_warpfield"]),
        name="outputnode",
    )

    mnitemplate = get_template(
        template="MNI152NLin6Asym",
        resolution=2,
        desc=None,
        suffix="T1w",
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

    # fmt:off
    workflow.connect([
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

    # merge new components
    merge_xfms_list = pe.Node(
        niu.Merge(2),
        name="merge_xfms_list",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    merge_inv_xfms_list = pe.Node(
        niu.Merge(2),
        name="merge_inv_xfms_list",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (disassemble_h5, merge_xfms_list, [("displacement_field", "in1")]),
        (disassemble_h5, merge_xfms_list, [("affine_transform", "in2")]),
        (disassemble_h5_inv, merge_inv_xfms_list, [("displacement_field", "in2")]),
        (disassemble_h5_inv, merge_inv_xfms_list, [("affine_transform", "in1")]),
    ])
    # fmt:on

    # combine the affine and warpfield xfms from the
    # disassembled h5 into a single warpfield xfm
    combine_xfms = pe.Node(
        ApplyTransforms(
            reference_image=mnitemplate,
            interpolation="LanczosWindowedSinc",
            print_out_composite_warp_file=True,
            output_image="ants_composite_xfm.nii.gz",
        ),
        name="combine_xfms",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    combine_inv_xfms = pe.Node(
        ApplyTransforms(
            reference_image=mnitemplate,
            interpolation="LanczosWindowedSinc",
            print_out_composite_warp_file=True,
            output_image="ants_composite_inv_xfm.nii.gz",
        ),
        name="combine_inv_xfms",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (inputnode, combine_xfms, [("t1w", "input_image")]),
        (merge_xfms_list, combine_xfms, [("out", "transforms")]),
        (inputnode, combine_inv_xfms, [("t1w", "input_image")]),
        (merge_inv_xfms_list, combine_inv_xfms, [("out", "transforms")]),
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

    # merge new components
    merge_new_components = pe.Node(
        niu.Merge(3),
        name="merge_new_components",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )
    merge_new_inv_components = pe.Node(
        niu.Merge(3),
        name="merge_new_inv_components",
        mem_gb=mem_gb,
        n_procs=omp_nthreads,
    )

    # fmt:off
    workflow.connect([
        (select_x_component, merge_new_components, [("out", "in1")]),
        (reverse_y_component, merge_new_components, [("out_file", "in2")]),
        (select_z_component, merge_new_components, [("out", "in3")]),
        (select_inv_x_component, merge_new_inv_components, [("out", "in1")]),
        (reverse_inv_y_component, merge_new_inv_components, [("out_file", "in2")]),
        (select_inv_z_component, merge_new_inv_components, [("out", "in3")]),
    ])
    # fmt:on

    # re-merge warpfield in FSL FNIRT format, with the reversed y-component from above
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
        (merge_new_components, remerge_warpfield, [("out", "in_files")]),
        (merge_new_inv_components, remerge_inv_warpfield, [("out", "in_files")]),
        (convert_xfm2world, outputnode, [("out_file", "world_xfm")]),
        (remerge_warpfield, outputnode, [("out_file", "merged_warpfield")]),
        (remerge_inv_warpfield, outputnode, [("out_file", "merged_inv_warpfield")]),
    ])
    # fmt:on

    return workflow
