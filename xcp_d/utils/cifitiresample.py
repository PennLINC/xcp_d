# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""workbench command for  cifti metric separation"""

from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLineInputSpec
from nipype import logging

iflogger = logging.getLogger("nipype.interface")


class CiftiSurfaceResampleInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="the gifti file",
    )

    current_sphere = File(
        exists=True,
        position=1,
        argstr=" %s",
        desc=" the current sphere surface in gifti for in_file",
    )

    new_sphere = File(
        exists=True,
        position=2,
        argstr=" %s",
        desc=" the new sphere surface to be resample the in_file to, eg fsaverag5 or fsl32k",
    )

    metric = traits.Str(argstr=" %s ",
                        position=3,
                        desc=" fixed for anatomic",
                        default="  BARYCENTRIC  ")

    out_file = File(
        name_source=["in_file"],
        name_template="resampled_%s.surf.gii",
        keep_extension=True,
        argstr=" %s",
        position=4,
        desc="The gifti output, either left and right",
    )


class CiftiSurfaceResampleOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output gifti file")


class CiftiSurfaceResample(WBCommand):
    r"""
    Resample a surface from one sphere to another.
    will comeback for documentation  @Azeez Adebimpe
    """
    input_spec = CiftiSurfaceResampleInputSpec
    output_spec = CiftiSurfaceResampleOutputSpec
    _cmd = "wb_command  -surface-resample"
