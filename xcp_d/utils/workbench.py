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



