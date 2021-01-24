# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""workbench command for  cifti metric separation"""

from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLineInputSpec
from nipype import logging
iflogger = logging.getLogger("nipype.interface")



class CiftiSeparateMetricInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="The input dense series",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="which dimension to smooth along, ROW or COLUMN",
    )
    metric = traits.Str(
        mandatory=True,
        argstr=" -metric %s ",
        position=2,
        desc="which of the structure eg CORTEX_LEFT CORTEX_RIGHT" \
            "check https://www.humanconnectome.org/software/workbench-command/-cifti-separate ",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="correlation_matrix_%s.func.gii",
        keep_extension=True,
        argstr=" %s",
        position=3,
        desc="The gifti output, iether left and right",
    )
    
class CiftiSeparateMetricOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output CIFTI file")

class CiftiSeparateMetric(WBCommand):
    input_spec = CiftiSeparateMetricInputSpec
    output_spec = CiftiSeparateMetricOutputSpec
    _cmd = "wb_command  -cifti-separate "
