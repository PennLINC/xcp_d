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
        name_source=["in_file"],
        name_template="%s_world.nii.gz",
        keep_extension=False,
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