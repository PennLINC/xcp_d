"""workbench command for wb_command -convert-affine -from-itk"""

from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLineInputSpec
from nipype import logging
iflogger = logging.getLogger("nipype.interface")

class ConvertAffineInputSpec(CommandLineInputSpec):

    from_what = traits.Str(
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

    to_what = traits.Str(
        mandatory=True,
        argstr="-to-%s ",
        position=2,
        desc="world, itk, or flirt",
    )

    out_file = File(
        argstr="%s",
        position=3,
        desc="The output file",
    )

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
        argstr="%s ",
        position=0,
        desc="The input file",
    )

    affine = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="The affine file",
    )

    out_file = File(
        keep_extension=False,
        argstr="%s",
        position=2,
        desc="The output file",
    )

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
    _cmd = "wb_command surface-apply-affine "


class ApplyWarpfieldInputSpec(CommandLineInputSpec):

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
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

    out_file = File(
        keep_extension=False,
        argstr="%s",
        position=2,
        desc="The output file",
    )

class ApplyAffineOutputSpec(TraitedSpec):
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
    _cmd = "wb_command surface-apply-warpfield "

class SurfaceSphereProjectUnprojectInputSpec(CommandLineInputSpec):

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="The input sphere file",
    )

    sphere_project_to = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="The warpfield file",
    )

    sphere_unproject_from = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=2,
        desc="The warpfield file",
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

