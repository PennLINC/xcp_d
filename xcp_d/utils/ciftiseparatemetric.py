# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""workbench command for  cifti metric separation"""

from nipype import logging
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, traits
from nipype.interfaces.workbench.base import WBCommand

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
        desc="which of the structure eg CORTEX_LEFT CORTEX_RIGHT"
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
    r"""
    Extract left or right hemisphere surface from CIFTI file (.dtseries)
    other structure can also be extracted
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries,

    >>> ciftiseparate = CiftiSeparateMetric()
    >>> ciftiseparate.inputs.in_file = 'sub-01XX_task-rest.dtseries.nii'
    >>> ciftiseparate.inputs.metric = "CORTEX_LEFT" # extract left hemisphere
    >>> ciftiseparate.inputs.out_file = 'sub_01XX_task-rest_hemi-L.func.gii'
    >>> ciftiseparate.inputs.direction = 'COLUMN'
    >>> ciftiseparate.cmdline
    wb_command  -cifti-separate 'sub-01XX_task-rest.dtseries.nii'  COLUMN \
      -metric CORTEX_LEFT 'sub_01XX_task-rest_hemi-L.func.gii'
    """
    input_spec = CiftiSeparateMetricInputSpec
    output_spec = CiftiSeparateMetricOutputSpec
    _cmd = "wb_command  -cifti-separate "
