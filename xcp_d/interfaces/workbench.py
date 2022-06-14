"""workbench command for wb_command -convert-affine -from-itk"""

from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLineInputSpec, SimpleInterface
from nipype import logging
from nipype.utils.filemanip import fname_presuffix

iflogger = logging.getLogger("nipype.interface")


class ConvertAffineInputSpec(CommandLineInputSpec):

    fromwhat = traits.Str(
        mandatory=True,
        argstr="-from-%s ",
        position=0,
        desc="world, itk, or flirt",
    )

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="The input file",
    )

    towhat = traits.Str(
        mandatory=True,
        argstr="-to-%s ",
        position=2,
        desc="world, itk, or flirt",
    )
    out_file = traits.File(argstr="%s",
                           name_source='in_file',
                           name_template='%s_world.nii.gz',
                           keep_extension=False)


class ConvertAffineOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output file")


class ConvertAffine(WBCommand):
    input_spec = ConvertAffineInputSpec
    output_spec = ConvertAffineOutputSpec
    _cmd = "wb_command -convert-affine "


class ApplyAffineInputSpec(CommandLineInputSpec):

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input file",
    )

    affine = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="The affine file",
    )

    out_file = File(argstr="%s",
                    name_source='in_file',
                    name_template='%s-MNIaffine.nii.gz',
                    keep_extension=False,
                    position=2)


class ApplyAffineOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output file")


class ApplyAffine(WBCommand):
    #    wb_command -surface-apply-affine
    #       <in-surf> - the surface to transform
    #       <affine> - the affine file
    #       <out-surf> - output - the output transformed surface

    #       [-flirt] - MUST be used if affine is a flirt affine
    #          <source-volume> - the source volume used when generating the affine
    #          <target-volume> - the target volume used when generating the affine

    #       For flirt matrices, you must use the -flirt option, because flirt
    #       matrices are not a complete description of the coordinate transform they
    #       represent.  If the -flirt option is not present, the affine must be a
    #       nifti 'world' affine, which can be obtained with the -convert-affine
    #       command, or aff_conv from the 4dfp suite.
    input_spec = ApplyAffineInputSpec
    output_spec = ApplyAffineOutputSpec
    _cmd = "wb_command -surface-apply-affine "


class ApplyWarpfieldInputSpec(CommandLineInputSpec):

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input file",
    )

    warpfield = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="The warpfield file",
    )

    out_file = File(argstr="%s",
                    name_source='in_file',
                    name_template='%s-MNIwarped.nii.gz',
                    keep_extension=False)


class ApplyWarpfieldOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output file")


class ApplyWarpfield(WBCommand):
    # APPLY WARPFIELD TO SURFACE FILE
    # wb_command -surface-apply-warpfield
    # <in-surf> - the surface to transform
    # <warpfield> - the INVERSE warpfield
    # <out-surf> - output - the output transformed surface

    # [-fnirt] - MUST be used if using a fnirt warpfield
    #     <forward-warp> - the forward warpfield

    # warping a surface requires the INVERSE of the warpfield used to
    # warp the volume it lines up with.  The header of the forward warp is
    # needed by the -fnirt option in order to correctly interpret the
    # displacements in the fnirt warpfield.

    # If the -fnirt option is not present, the warpfield must be a nifti
    # 'world' warpfield, which can be obtained with the -convert-warpfield
    # command.

    input_spec = ApplyWarpfieldInputSpec
    output_spec = ApplyWarpfieldOutputSpec
    _cmd = "wb_command -surface-apply-warpfield "


class SurfaceSphereProjectUnprojectInputSpec(CommandLineInputSpec):

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="a sphere with the desired output mesh",
    )

    sphere_project_to = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="a sphere that aligns with sphere-in",
    )

    sphere_unproject_from = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=2,
        desc="deformed to the desired output space",
    )

    out_file = File(
        keep_extension=False,
        argstr="%s",
        position=3,
        desc="The sphere output file",
    )


class SurfaceSphereProjectUnprojectOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output file")


class SurfaceSphereProjectUnproject(WBCommand):
    # COPY REGISTRATION DEFORMATIONS TO DIFFERENT SPHERE
    # wb_command -surface-sphere-project-unproject
    # <sphere-in> - a sphere with the desired output mesh
    # <sphere-project-to> - a sphere that aligns with sphere-in
    # <sphere-unproject-from> - <sphere-project-to> deformed to the desired output space
    # <sphere-out> - output - the output sphere

    input_spec = SurfaceSphereProjectUnprojectInputSpec
    output_spec = SurfaceSphereProjectUnprojectOutputSpec
    _cmd = "wb_command -surface-sphere-project-unproject "


class _ChangeXfmTypeInputSpec(CommandLineInputSpec):
    in_transform = traits.File(exists=True,
                               argstr="%s",
                               mandatory=True,
                               position=0)


class _ChangeXfmTypeOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ChangeXfmType(SimpleInterface):
    input_spec = _ChangeXfmTypeInputSpec
    output_spec = _ChangeXfmTypeOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.in_transform) as f:
            lines = f.readlines()
        listcomp = [
            line.replace('AffineTransform', 'MatrixOffsetTransformBase')
            for line in lines
        ]
        outfile = fname_presuffix(self.inputs.in_transform,
                                  suffix='_MatrixOffsetTransformBase',
                                  newpath=runtime.cwd)
        with open(outfile, 'w') as write_file:
            write_file.write(''.join(listcomp))
        self._results['out_transform'] = outfile
        return runtime
