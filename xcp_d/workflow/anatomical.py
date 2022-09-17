# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Anatomical post-processing workflows.

fectch anatomical files/resmapleing surfaces to fsl32k
^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: init_structral_wf

"""

import fnmatch
import os
import shutil
from pathlib import Path

from nipype.interfaces import utility as niu
from nipype.interfaces.ants import CompositeTransformUtil  # MB
from nipype.interfaces.ants.resampling import (
    ApplyTransforms as antsapplytransforms,  # TM
)
from nipype.interfaces.freesurfer import MRIsConvert
from nipype.interfaces.fsl import Merge as fslmerge  # TM
from nipype.interfaces.fsl.maths import BinaryMaths as fslbinarymaths  # TM
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from templateflow.api import get as get_template

from xcp_d.interfaces import BrainPlotx, RibbontoStatmap
from xcp_d.interfaces.ants import CompositeInvTransformUtil, ConvertTransformFile
from xcp_d.interfaces.c3 import C3d  # TM
from xcp_d.interfaces.connectivity import ApplyTransformsx
from xcp_d.interfaces.workbench import (  # MB,TM
    ApplyAffine,
    ApplyWarpfield,
    ChangeXfmType,
    ConvertAffine,
    SurfaceAverage,
    SurfaceGenerateInflated,
    SurfaceSphereProjectUnproject,
)
from xcp_d.utils import CiftiSurfaceResample, bid_derivative, collect_data


class DerivativesDataSink(bid_derivative):
    out_path_base = "xcp_d"


def init_anatomical_wf(
    omp_nthreads,
    fmri_dir,
    subject_id,
    output_dir,
    t1w_to_mni,
    input_type,
    mem_gb,
    name="anatomical_wf",
):
    """
    This workflow is convert surfaces (gifti) from fMRI to standard space-fslr-32k
    It also resamples the t1w segmnetation to standard space, MNI

    Workflow Graph
        .. workflow::
        :graph2use: orig
        :simple_form: yes
        from xcp_d.workflows import init_anatomical_wf
        wf = init_anatomical_wf(
        omp_nthreads,
        fmri_dir,
        subject_id,
        output_dir,
        t1w_to_mni,
        name="anatomical_wf",
     )
     Parameters
     ----------
     omp_nthreads : int
          number of threads
     fmri_dir : str
          fmri output directory
     subject_id : str
          subject id
     output_dir : str
          output directory
     t1w_to_mni : str
          t1w to MNI transform
     name : str
          workflow name

     Inputs
     ------
     t1w: str
          t1w file
     t1w_seg: str
          t1w segmentation file

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["t1w", "t1seg"]), name="inputnode"
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
    layout, subj_data = collect_data(
        bids_dir=fmri_dir, participant_label=subject_id, bids_validate=False
    )

    if input_type == "dcan" or input_type == "hcp":
        ds_t1wmni_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space="MNI152NLin6Asym",
                desc="preproc",
                suffix="T1w",
                extension=".nii.gz",
            ),
            name="ds_t1wmni_wf",
            run_without_submitting=False,
        )

        ds_t1wseg_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space="MNI152NLin6Asym",
                suffix="dseg",
                extension=".nii.gz",
            ),
            name="ds_t1wseg_wf",
            run_without_submitting=False,
        )
        workflow.connect(
            [
                (inputnode, ds_t1wmni_wf, [("t1w", "in_file")]),
                (inputnode, ds_t1wseg_wf, [("t1seg", "in_file")]),
                (inputnode, ds_t1wmni_wf, [("t1w", "source_file")]),
                (inputnode, ds_t1wseg_wf, [("t1w", "source_file")]),
            ]
        )

        all_files = list(layout.get_files())
        L_inflated_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-L_inflated.surf.gii"
        )[0]
        R_inflated_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-R_inflated.surf.gii"
        )[0]
        L_midthick_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-L_midthickness.surf.gii"
        )[0]
        R_midthick_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-R_midthickness.surf.gii"
        )[0]
        L_pial_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-L_pial.surf.gii"
        )[0]
        R_pial_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-R_pial.surf.gii"
        )[0]
        L_wm_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-L_smoothwm.surf.gii"
        )[0]
        R_wm_surf = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*hemi-R_smoothwm.surf.gii"
        )[0]

        ribbon = fnmatch.filter(
            all_files, "*sub-*" + subject_id + "*desc-ribbon.nii.gz"
        )[0]

        ses_id = _getsesid(ribbon)
        anatdir = output_dir + "/xcp_d/sub-" + subject_id + "/ses-" + ses_id + "/anat"
        if not os.path.exists(anatdir):
            os.makedirs(anatdir)

        surf = [
            L_inflated_surf,
            R_inflated_surf,
            L_midthick_surf,
            R_midthick_surf,
            L_pial_surf,
            R_pial_surf,
            L_wm_surf,
            R_wm_surf,
        ]

        for ss in surf:
            shutil.copy(ss, anatdir)

        ribbon2statmap_wf = pe.Node(
            RibbontoStatmap(ribbon=ribbon),
            name="ribbon2statmap",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        brainspritex_wf = pe.Node(
            BrainPlotx(), name="brainsprite", mem_gb=mem_gb, n_procs=omp_nthreads
        )

        ds_brainspriteplot_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                check_hdr=False,
                dismiss_entities=["desc"],
                desc="brainplot",
                datatype="figures",
            ),
            name="brainspriteplot",
            run_without_submitting=True,
        )

        workflow.connect(
            [
                (ribbon2statmap_wf, brainspritex_wf, [("out_file", "in_file")]),
                (inputnode, brainspritex_wf, [("t1w", "template")]),
                (brainspritex_wf, ds_brainspriteplot_wf, [("out_html", "in_file")]),
                (inputnode, ds_brainspriteplot_wf, [("t1w", "source_file")]),
            ]
        )

    else:
        all_files = list(layout.get_files())

        # #TM: need to replace MNI92FSL xfm with the correct
        # xfm from the MNI output space of fMRIPrep/NiBabies
        # (MNI2009, MNIInfant, or for cifti output MNI152NLin6Asym)
        # to MNI152NLin6Asym.
        t1w_transform_wf = pe.Node(
            ApplyTransformsx(
                num_threads=2,
                reference_image=mnitemplate,
                # transforms=[str(t1w_to_mni), str(MNI92FSL)],
                transforms=t1w_to_mni,
                interpolation="LanczosWindowedSinc",
                input_image_type=3,
                dimension=3,
            ),
            name="t1w_transform",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        seg_transform_wf = pe.Node(
            ApplyTransformsx(
                num_threads=2,
                reference_image=mnitemplate,
                # transforms=[str(t1w_to_mni), str(MNI92FSL)],
                transforms=t1w_to_mni,
                interpolation="MultiLabel",
                input_image_type=3,
                dimension=3,
            ),
            name="seg_transform",
            mem_gb=mem_gb,
            n_procs=omp_nthreads,
        )

        ds_t1wmni_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space="MNI152NLin6Asym",
                desc="preproc",
                suffix="T1w",
                extension=".nii.gz",
            ),
            name="ds_t1wmni_wf",
            run_without_submitting=False,
        )

        ds_t1wseg_wf = pe.Node(
            DerivativesDataSink(
                base_directory=output_dir,
                space="MNI152NLin6Asym",
                suffix="dseg",
                extension=".nii.gz",
            ),
            name="ds_t1wseg_wf",
            run_without_submitting=False,
        )

        workflow.connect(
            [
                (inputnode, t1w_transform_wf, [("t1w", "input_image")]),
                (inputnode, seg_transform_wf, [("t1seg", "input_image")]),
                (t1w_transform_wf, ds_t1wmni_wf, [("output_image", "in_file")]),
                (seg_transform_wf, ds_t1wseg_wf, [("output_image", "in_file")]),
                (inputnode, ds_t1wmni_wf, [("t1w", "source_file")]),
                (inputnode, ds_t1wseg_wf, [("t1w", "source_file")]),
            ]
        )

        # verify freesurfer directory

        p = Path(fmri_dir)
        import glob as glob

        freesurfer_paths = glob.glob(
            str(p.parent) + "/freesurfer*"
        )  # for fmriprep and nibabies
        if len(freesurfer_paths) == 0:
            freesurfer_paths = glob.glob(
                str(p) + "/sourcedata/*freesurfer*"
            )  # nibabies

        if len(freesurfer_paths) > 0 and "freesurfer" in os.path.basename(
            freesurfer_paths[0]
        ):
            freesurfer_path = freesurfer_paths[0]
        else:
            freesurfer_path = None

        if freesurfer_path is not None and os.path.isdir(freesurfer_path):

            L_inflated_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-L_inflated.surf.gii"
            )[0]
            R_inflated_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-R_inflated.surf.gii"
            )[0]
            L_midthick_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-L_midthickness.surf.gii"
            )[0]
            R_midthick_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-R_midthickness.surf.gii"
            )[0]
            L_pial_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-L_pial.surf.gii"
            )[0]
            R_pial_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-R_pial.surf.gii"
            )[0]
            L_wm_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-L_smoothwm.surf.gii"
            )[0]
            R_wm_surf = fnmatch.filter(
                all_files, "*sub-*" + subject_id + "*hemi-R_smoothwm.surf.gii"
            )[0]

            # get sphere surfaces to be converted
            if "sub-" not in subject_id:
                subid = "sub-" + subject_id
            else:
                subid = subject_id

            left_sphere_fsLR = str(
                get_template(template="fsLR", hemi="L", density="32k", suffix="sphere")[
                    0
                ]
            )
            right_sphere_fsLR = str(
                get_template(template="fsLR", hemi="R", density="32k", suffix="sphere")[
                    0
                ]
            )

            left_sphere_raw = (
                str(freesurfer_path) + "/" + subid + "/surf/lh.sphere.reg"
            )  # MB, TM
            right_sphere_raw = (
                str(freesurfer_path) + "/" + subid + "/surf/rh.sphere.reg"
            )  # MB, TM

            # use ANTs CompositeTransformUtil to separate the .h5 into affine and warpfield xfms
            h5_file = fnmatch.filter(
                all_files,
                "*sub-*"
                + subject_id
                + "*from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5",
            )[
                0
            ]  # MB

            h5_inv_file = fnmatch.filter(
                all_files,
                "*sub-*"
                + subject_id
                + "*from-MNI152NLin6Asym_to-T1w_mode-image_xfm.h5",
            )[
                0
            ]  # MB
            disassemble_h5 = pe.Node(
                CompositeTransformUtil(
                    process="disassemble",
                    in_file=h5_file,
                    output_prefix="T1w_to_MNI152Lin6Asym",
                ),
                name="disassemble_h5",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # MB

            # Nipype's CompositeTransformUtil assumes a certain file naming and
            # concatenation order of xfms which does not work for the inverse .h5,
            # so we use our modified class, "CompositeInvTransformUtil"
            disassemble_h5_inv = pe.Node(
                CompositeInvTransformUtil(
                    process="disassemble",
                    in_file=h5_inv_file,
                    output_prefix="MNI152Lin6Asym_to_T1w",
                ),
                name="disassemble_h5_inv",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # TM

            # convert affine from ITK binary to txt
            convert_ants_transform = pe.Node(
                ConvertTransformFile(dimension=3), name="convert_ants_transform"
            )  # MB

            # change xfm type from "AffineTransform" to "MatrixOffsetTransformBase"
            # since wb_command doesn't recognize "AffineTransform"
            # (AffineTransform is a subclass of MatrixOffsetTransformBase
            # which makes this okay to do AFAIK)
            change_xfm_type = pe.Node(ChangeXfmType(), name="change_xfm_type")  # MB

            # convert affine xfm to "world" so it works with -surface-apply-affine
            convert_xfm2world = pe.Node(
                ConvertAffine(fromwhat="itk", towhat="world"), name="convert_xfm2world"
            )  # MB

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

            # combine the affine and warpfield xfms from the
            # disassembled h5 into a single warpfield xfm
            #

            combine_xfms = pe.Node(
                antsapplytransforms(
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
                antsapplytransforms(
                    reference_image=mnitemplate,
                    interpolation="LanczosWindowedSinc",
                    print_out_composite_warp_file=True,
                    output_image="ants_composite_inv_xfm.nii.gz",
                ),
                name="combine_inv_xfms",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            # use C3d to separate the combined warpfield xfm
            # into x, y, and z components

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

            # reverse y-component of the warpfield
            # (need to do this when converting a warpfield from ANTs to FNIRT format
            # for use with wb_command -surface-apply-warpfield)

            reverse_y_component = pe.Node(
                fslbinarymaths(operation="mul", operand_value=-1.0),
                name="reverse_y_component",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            reverse_inv_y_component = pe.Node(
                fslbinarymaths(operation="mul", operand_value=-1.0),
                name="reverse_inv_y_component",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

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

            # re-merge warpfield in FSL FNIRT format, with the reversed y-component from above
            remerge_warpfield = pe.Node(
                fslmerge(dimension="t"),
                name="remerge_warpfield",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            remerge_inv_warpfield = pe.Node(
                fslmerge(dimension="t"),
                name="remerge_inv_warpfield",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            # convert spheres (from FreeSurfer surf dir) to gifti #MB
            left_sphere_raw_mris = pe.Node(
                MRIsConvert(out_datatype="gii", in_file=left_sphere_raw),
                name="left_sphere_raw_mris",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # MB
            right_sphere_raw_mris = pe.Node(
                MRIsConvert(
                    out_datatype="gii",
                    in_file=right_sphere_raw,
                ),
                name="right_sphere_raw_mris",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # MB

            # apply affine to native surfs
            lh_surface_apply_affine = pe.Node(
                ApplyAffine(),
                name="lh_surface_apply_affine",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # TM
            lh_surface_apply_affine.iterables = (
                "in_file",
                [L_midthick_surf, L_pial_surf, L_wm_surf],
            )

            rh_surface_apply_affine = pe.Node(
                ApplyAffine(),
                name="rh_surface_apply_affine",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # TM
            rh_surface_apply_affine.iterables = (
                "in_file",
                [R_midthick_surf, R_pial_surf, R_wm_surf],
            )
            # apply FNIRT-format warpfield
            lh_surface_apply_warpfield = pe.Node(
                ApplyWarpfield(),
                name="lh_surface_apply_warpfield",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # TM

            rh_surface_apply_warpfield = pe.Node(
                ApplyWarpfield(),
                name="rh_surface_apply_warpfield",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )  # TM

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

            fs_L2fsLR = str(
                get_template(
                    template="fsaverage",
                    hemi="L",
                    density="164k",
                    desc="std",
                    suffix="sphere",
                )
            )

            fs_R2fsLR = str(
                get_template(
                    template="fsaverage",
                    hemi="R",
                    density="164k",
                    desc="std",
                    suffix="sphere",
                )
            )

            surface_sphere_project_unproject_lh = pe.Node(
                SurfaceSphereProjectUnproject(
                    sphere_project_to=fs_std_mesh_L,
                    sphere_unproject_from=fs_L2fsLR,
                ),
                name='surface_sphere_project_unproject_lh'
            )
            surface_sphere_project_unproject_rh = pe.Node(
                SurfaceSphereProjectUnproject(
                    sphere_project_to=fs_std_mesh_R,
                    sphere_unproject_from=fs_R2fsLR,
                ),
                name='surface_sphere_project_unproject_rh'
            )

            # resample the mid, pial, wm surfs to fsLR32k

            lh_32k_midthick_wf = pe.Node(
                CiftiSurfaceResample(
                    in_file=L_midthick_surf,
                    new_sphere=left_sphere_fsLR,
                    metric=" BARYCENTRIC ",
                ),
                name="lh_32k_midthick_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            lh_32k_pial_wf = pe.Node(
                CiftiSurfaceResample(
                    in_file=L_pial_surf,
                    new_sphere=left_sphere_fsLR,
                    metric=" BARYCENTRIC ",
                ),
                name="lh_32k_pial_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            lh_32k_wm_wf = pe.Node(
                CiftiSurfaceResample(
                    in_file=L_wm_surf,
                    new_sphere=left_sphere_fsLR,
                    metric=" BARYCENTRIC ",
                ),
                name="lh_32k_wm_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            rh_32k_midthick_wf = pe.Node(
                CiftiSurfaceResample(
                    in_file=R_midthick_surf,
                    new_sphere=right_sphere_fsLR,
                    metric=" BARYCENTRIC ",
                ),
                name="rh_32k_midthick_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            rh_32k_pial_wf = pe.Node(
                CiftiSurfaceResample(
                    in_file=R_pial_surf,
                    new_sphere=right_sphere_fsLR,
                    metric=" BARYCENTRIC ",
                ),
                name="rh_32k_pial_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            rh_32k_wm_wf = pe.Node(
                CiftiSurfaceResample(
                    in_file=R_wm_surf,
                    new_sphere=right_sphere_fsLR,
                    metric=" BARYCENTRIC ",
                ),
                name="rh_32k_wm_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            # write report node
            ds_wmLsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc"],
                    space="fsLR",
                    density="32k",
                    desc="smoothwm",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="L",
                    source_file=L_wm_surf,
                ),
                name="ds_wmLsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )
            ds_wmRsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc"],
                    space="fsLR",
                    density="32k",
                    desc="smoothwm",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="R",
                    source_file=R_wm_surf,
                ),
                name="ds_wmRsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )

            ds_pialLsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc"],
                    space="fsLR",
                    density="32k",
                    desc="pial",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="L",
                    source_file=L_pial_surf,
                ),
                name="ds_pialLsurf_wf",
                run_without_submitting=True,
                mem_gb=2,
            )
            ds_pialRsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc"],
                    space="fsLR",
                    density="32k",
                    desc="pial",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="R",
                    source_file=R_pial_surf,
                ),
                name="ds_pialRsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )

            ds_midLsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc"],
                    space="fsLR",
                    density="32k",
                    desc="midthickness",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="L",
                    source_file=L_midthick_surf,
                ),
                name="ds_midLsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )

            ds_midRsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc"],
                    space="fsLR",
                    density="32k",
                    desc="midthickness",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="R",
                    source_file=R_midthick_surf,
                ),
                name="ds_midRsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )

            workflow.connect(
                [
                    (disassemble_h5, merge_xfms_list, [("displacement_field", "in1")]),
                    (disassemble_h5, merge_xfms_list, [("affine_transform", "in2")]),
                    (inputnode, combine_xfms, [("t1w", "input_image")]),
                    (merge_xfms_list, combine_xfms, [("out", "transforms")]),
                    (disassemble_h5, get_xyz_components, [("displacement_field", "in_file")]),
                    (get_xyz_components, select_x_component, [("out_files", "inlist")]),
                    (get_xyz_components, select_y_component, [("out_files", "inlist")]),
                    (get_xyz_components, select_z_component, [("out_files", "inlist")]),
                    (select_y_component, reverse_y_component, [("out", "in_file")]),
                    (select_x_component, merge_new_components, [("out", "in1")]),
                    (reverse_y_component, merge_new_components, [("out_file", "in2")]),
                    (select_z_component, merge_new_components, [("out", "in3")]),
                    (merge_new_components, remerge_warpfield, [("out", "in_files")]),
                ]
            )
            workflow.connect(
                [
                    (
                        disassemble_h5,
                        convert_ants_transform,
                        [("affine_transform", "in_transform")],
                    ),
                    (
                        convert_ants_transform,
                        change_xfm_type,
                        [("out_transform", "in_transform")],
                    ),
                    (
                        change_xfm_type,
                        convert_xfm2world,
                        [("out_transform", "in_file")],
                    ),
                ]
            )
            workflow.connect(
                [  # concat order is affine 1st, field 2nd for inverse xfm (opposite of fwd xfm)
                    # but input and ref image are same for fwd and inv combine_xfms
                    (
                        disassemble_h5_inv,
                        merge_inv_xfms_list,
                        [("displacement_field", "in2")],
                    ),
                    (
                        disassemble_h5_inv,
                        merge_inv_xfms_list,
                        [("affine_transform", "in1")],
                    ),
                    (inputnode, combine_inv_xfms, [("t1w", "input_image")]),
                    (merge_inv_xfms_list, combine_inv_xfms, [("out", "transforms")]),
                    (
                        disassemble_h5_inv,
                        get_inv_xyz_components,
                        [("displacement_field", "in_file")],
                    ),
                    (
                        get_inv_xyz_components,
                        select_inv_x_component,
                        [("out_files", "inlist")],
                    ),
                    (
                        get_inv_xyz_components,
                        select_inv_y_component,
                        [("out_files", "inlist")],
                    ),
                    (
                        get_inv_xyz_components,
                        select_inv_z_component,
                        [("out_files", "inlist")],
                    ),
                    (
                        select_inv_y_component,
                        reverse_inv_y_component,
                        [("out", "in_file")],
                    ),
                    (
                        select_inv_x_component,
                        merge_new_inv_components,
                        [("out", "in1")],
                    ),
                    (
                        reverse_inv_y_component,
                        merge_new_inv_components,
                        [("out_file", "in2")],
                    ),
                    (
                        select_inv_z_component,
                        merge_new_inv_components,
                        [("out", "in3")],
                    ),
                    (
                        merge_new_inv_components,
                        remerge_inv_warpfield,
                        [("out", "in_files")],
                    ),
                ]
            )
            workflow.connect(
                [
                    (
                        convert_xfm2world,
                        lh_surface_apply_affine,
                        [("out_file", "affine")],
                    ),
                    (
                        lh_surface_apply_affine,
                        lh_surface_apply_warpfield,
                        [("out_file", "in_file")],
                    ),
                    (
                        remerge_warpfield,
                        lh_surface_apply_warpfield,
                        [("merged_file", "forward_warp")],
                    ),
                    (
                        remerge_inv_warpfield,
                        lh_surface_apply_warpfield,
                        [("merged_file", "warpfield")],
                    ),
                    (
                        left_sphere_raw_mris,
                        surface_sphere_project_unproject_lh,
                        [("converted", "in_file")]
                    ),
                    (
                        surface_sphere_project_unproject_lh,
                        lh_32k_midthick_wf,
                        [("out_file", "current_sphere")],
                    ),
                    (
                        surface_sphere_project_unproject_lh,
                        lh_32k_pial_wf,
                        [("out_file", "current_sphere")],
                    ),
                    (
                        surface_sphere_project_unproject_lh,
                        lh_32k_wm_wf,
                        [("out_file", "current_sphere")],
                    ),
                    (lh_32k_midthick_wf, ds_midLsurf_wf, [("out_file", "in_file")]),
                    (lh_32k_pial_wf, ds_pialLsurf_wf, [("out_file", "in_file")]),
                    (lh_32k_wm_wf, ds_wmLsurf_wf, [("out_file", "in_file")]),
                ]
            )
            workflow.connect(
                [
                    (
                        convert_xfm2world,
                        rh_surface_apply_affine,
                        [("out_file", "affine")],
                    ),
                    (
                        rh_surface_apply_affine,
                        rh_surface_apply_warpfield,
                        [("out_file", "in_file")],
                    ),
                    (
                        remerge_warpfield,
                        rh_surface_apply_warpfield,
                        [("merged_file", "forward_warp")],
                    ),
                    (
                        remerge_inv_warpfield,
                        rh_surface_apply_warpfield,
                        [("merged_file", "warpfield")],
                    ),
                    (
                        right_sphere_raw_mris,
                        surface_sphere_project_unproject_rh,
                        [("converted", "in_file")]
                    ),
                    (
                        surface_sphere_project_unproject_rh,
                        rh_32k_midthick_wf,
                        [("out_file", "current_sphere")],
                    ), (
                        surface_sphere_project_unproject_rh,
                        rh_32k_pial_wf,
                        [("out_file", "current_sphere")],
                    ),
                    (
                        surface_sphere_project_unproject_rh,
                        rh_32k_wm_wf,
                        [("out_file", "current_sphere")],
                    ),
                    (rh_32k_midthick_wf, ds_midRsurf_wf, [("out_file", "in_file")]),
                    (rh_32k_pial_wf, ds_pialRsurf_wf, [("out_file", "in_file")]),
                    (rh_32k_wm_wf, ds_wmRsurf_wf, [("out_file", "in_file")]),
                ]
            )

            # make "HCP-style" native midthickness and inflated
            left_hcpmidthick_native_wf = pe.Node(
                SurfaceAverage(surface_in1=L_pial_surf, surface_in2=L_wm_surf),
                name="left_hcpmidthick_native_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            right_hcpmidthick_native_wf = pe.Node(
                SurfaceAverage(surface_in1=R_pial_surf, surface_in2=R_wm_surf),
                name="right_hcpmidthick_native_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            left_hcpmidthick_surf_wf = pe.Node(
                CiftiSurfaceResample(
                    new_sphere=left_sphere_fsLR,
                    metric=" BARYCENTRIC "
                ),
                name="left_hcpmidthick_surf_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            right_hcpmidthick_surf_wf = pe.Node(
                CiftiSurfaceResample(
                    new_sphere=right_sphere_fsLR,
                    metric=" BARYCENTRIC "
                ),
                name="right_hcpmidthick_surf_wf",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            left_hcpinflated_surf_wf = pe.Node(
                SurfaceGenerateInflated(iterations_scale_value=0.75),
                name="left_hcpinflated_surf_wf",
            )
            right_hcpinflated_surf_wf = pe.Node(
                SurfaceGenerateInflated(iterations_scale_value=0.75),
                name="right_hcpinflated_surf_wf",
            )

            ds_hcpmidLsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc", "suffix"],
                    space="fsLR",
                    density="32k",
                    suffix="hcpmidthickness",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="L",
                    source_file=L_midthick_surf,
                ),
                name="ds_hcpmidLsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )
            ds_hcpmidRsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc", "suffix"],
                    space="fsLR",
                    density="32k",
                    suffix="hcpmidthickness",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="R",
                    source_file=R_midthick_surf,
                ),
                name="ds_hcpmidRsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )
            ds_hcpinfLsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc", "suffix"],
                    space="fsLR",
                    density="32k",
                    suffix="hcpinflated",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="L",
                    source_file=L_inflated_surf,
                ),
                name="ds_hcpinfLsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )
            ds_hcpinfRsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc", "suffix"],
                    space="fsLR",
                    density="32k",
                    suffix="hcpinflated",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="R",
                    source_file=R_inflated_surf,
                ),
                name="ds_hcpinfRsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )
            ds_hcpveryinfLsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc", "suffix"],
                    space="fsLR",
                    density="32k",
                    suffix="hcpveryinflated",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="L",
                    source_file=L_inflated_surf,
                ),
                name="ds_hcpveryinfLsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )
            ds_hcpveryinfRsurf_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    dismiss_entities=["desc", "suffix"],
                    space="fsLR",
                    density="32k",
                    suffix="hcpveryinflated",
                    check_hdr=False,
                    extension=".surf.gii",
                    hemi="R",
                    source_file=R_inflated_surf,
                ),
                name="ds_hcpveryinfRsurf_wf",
                run_without_submitting=False,
                mem_gb=2,
            )

            workflow.connect(
                [
                    (
                        surface_sphere_project_unproject_lh,
                        left_hcpmidthick_surf_wf,
                        [("out_file", "current_sphere")],
                    ),
                    (
                        left_hcpmidthick_native_wf,
                        left_hcpmidthick_surf_wf,
                        [("out_file", "in_file")],
                    ),
                    (
                        left_hcpmidthick_surf_wf,
                        ds_hcpmidLsurf_wf,
                        [("out_file", "in_file")],
                    ),
                ]
            )

            workflow.connect(
                [
                    (
                        surface_sphere_project_unproject_rh,
                        right_hcpmidthick_surf_wf,
                        [("out_file", "current_sphere")],
                    ),
                    (
                        right_hcpmidthick_native_wf,
                        right_hcpmidthick_surf_wf,
                        [("out_file", "in_file")],
                    ),
                    (
                        right_hcpmidthick_surf_wf,
                        ds_hcpmidRsurf_wf,
                        [("out_file", "in_file")],
                    ),
                ]
            )

            workflow.connect(
                [
                    (
                        left_hcpmidthick_surf_wf,
                        left_hcpinflated_surf_wf,
                        [("out_file", "anatomical_surface_in")],
                    ),
                    (
                        left_hcpinflated_surf_wf,
                        ds_hcpinfLsurf_wf,
                        [("inflated_out_file", "in_file")],
                    ),
                    (
                        left_hcpinflated_surf_wf,
                        ds_hcpveryinfLsurf_wf,
                        [("very_inflated_out_file", "in_file")],
                    ),
                ]
            )

            workflow.connect(
                [
                    (
                        right_hcpmidthick_surf_wf,
                        right_hcpinflated_surf_wf,
                        [("out_file", "anatomical_surface_in")],
                    ),
                    (
                        right_hcpinflated_surf_wf,
                        ds_hcpinfRsurf_wf,
                        [("inflated_out_file", "in_file")],
                    ),
                    (
                        right_hcpinflated_surf_wf,
                        ds_hcpveryinfRsurf_wf,
                        [("very_inflated_out_file", "in_file")],
                    ),
                ]
            )

            ribbon = str(freesurfer_path) + "/" + subid + "/mri/ribbon.mgz"

            t1w_mgz = str(freesurfer_path) + "/" + subid + "/mri/orig.mgz"

            # nibabies outputs do not  have ori.mgz, ori is the same as norm.mgz
            if not Path(t1w_mgz).is_file():
                t1w_mgz = str(freesurfer_path) + "/" + subid + "/mri/norm.mgz"

            ribbon2statmap_wf = pe.Node(
                RibbontoStatmap(ribbon=ribbon),
                name="ribbon2statmap",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )

            # brainplot
            brainspritex_wf = pe.Node(
                BrainPlotx(), name="brainsprite", mem_gb=mem_gb, n_procs=omp_nthreads
            )

            ds_brainspriteplot_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    check_hdr=False,
                    dismiss_entities=["desc"],
                    desc="brainplot",
                    datatype="figures",
                ),
                name="brainspriteplot",
            )

            workflow.connect(
                [
                    # (pial2vol_wf,addwmpial_wf,[('out_file','in_file')]),
                    # (wm2vol_wf,addwmpial_wf,[('out_file','operand_files')]),
                    (inputnode, brainspritex_wf, [("t1w", "template")]),
                    (ribbon2statmap_wf, brainspritex_wf, [("out_file", "in_file")]),
                    (brainspritex_wf, ds_brainspriteplot_wf, [("out_html", "in_file")]),
                    (inputnode, ds_brainspriteplot_wf, [("t1w", "source_file")]),
                ]
            )

        else:
            ribbon2statmap_wf = pe.Node(
                RibbontoStatmap(),
                name="ribbon2statmap",
                mem_gb=mem_gb,
                n_procs=omp_nthreads,
            )
            brainspritex_wf = pe.Node(
                BrainPlotx(), name="brainsprite", mem_gb=mem_gb, n_procs=omp_nthreads
            )
            ds_brainspriteplot_wf = pe.Node(
                DerivativesDataSink(
                    base_directory=output_dir,
                    check_hdr=False,
                    dismiss_entities=[
                        "desc",
                    ],
                    desc="brainplot",
                    datatype="figures",
                ),
                name="brainspriteplot",
            )

            workflow.connect(
                [
                    (inputnode, brainspritex_wf, [("t1w", "template")]),
                    (inputnode, ribbon2statmap_wf, [("t1seg", "ribbon")]),
                    (ribbon2statmap_wf, brainspritex_wf, [("out_file", "in_file")]),
                    (brainspritex_wf, ds_brainspriteplot_wf, [("out_html", "in_file")]),
                    (inputnode, ds_brainspriteplot_wf, [("t1w", "source_file")]),
                ]
            )

    return workflow


def _getsesid(filename):
    ses_id = None
    filex = os.path.basename(filename)

    file_id = filex.split("_")
    for k in file_id:
        if "ses" in k:
            ses_id = k.split("-")[1]
            break

    return ses_id
