# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for parcellating CIFTI files."""

from nipype import logging
from nipype.interfaces.base import CommandLineInputSpec, File, TraitedSpec, traits
from nipype.interfaces.workbench.base import WBCommand

iflogger = logging.getLogger("nipype.interface")


class CiftiParcellateInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="The input CIFTI file",
    )
    atlas_label = traits.File(
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="atlas label, in mm",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s ",
        position=2,
        desc="which dimension to smooth along, ROW or COLUMN",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="parcelated_%s.nii",
        keep_extension=True,
        argstr=" %s",
        position=3,
        desc="The output CIFTI",
    )

    spatial_weights = traits.Str(
        argstr="-spatial-weights ",
        position=4,
        desc=" spatial weight file",
    )

    left_area_surf = File(
        exists=True,
        position=5,
        argstr="-left-area-surface %s",
        desc="Specify the left surface to use",
    )

    right_area_surf = File(
        exists=True,
        position=6,
        argstr="-right-area-surface %s",
        desc="Specify the right surface to use",
    )
    cerebellum_area_surf = File(
        exists=True,
        position=7,
        argstr="-cerebellum-area-surf %s",
        desc="specify the cerebellum surface to use",
    )

    left_area_metric = File(
        exists=True,
        position=8,
        argstr="-left-area-metric %s",
        desc="Specify the left surface metric to use",
    )

    right_area_metric = File(
        exists=True,
        position=9,
        argstr="-right-area-metric %s",
        desc="Specify the right surface  metric to use",
    )
    cerebellum_area_metric = File(
        exists=True,
        position=10,
        argstr="-cerebellum-area-metric %s",
        desc="specify the cerebellum surface  metricto use",
    )

    cifti_weights = File(
        exists=True,
        position=11,
        argstr="-cifti-weights %s",
        desc="cifti file containing weights",
    )
    cor_method = traits.Str(
        position=12,
        default='MEAN ',
        argstr="-method %s",
        desc=" correlation method, option inlcude MODE",
    )


class CiftiParcellateOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output CIFTI file")


class CiftiParcellate(WBCommand):
    """Extract timeseries from CIFTI file.

    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries.

    Examples
    --------
    >>> ciftiparcel = CiftiParcellate()
    >>> ciftiparcel.inputs.in_file = 'sub-01XX_task-rest.dtseries.nii'
    >>> ciftiparcel.inputs.out_file = 'sub_01XX_task-rest.ptseries.nii'
    >>> ciftiparcel.inputs.atlas_label = 'schaefer_space-fsLR_den-32k_desc-400_atlas.dlabel.nii'
    >>> ciftiparcel.inputs.direction = 'COLUMN'
    >>> ciftiparcel.cmdline
    wb_command -cifti-parcellate sub-01XX_task-rest.dtseries.nii \
    schaefer_space-fsLR_den-32k_desc-400_atlas.dlabel.nii   COLUMN \
    sub_01XX_task-rest.ptseries.nii
    """

    input_spec = CiftiParcellateInputSpec
    output_spec = CiftiParcellateOutputSpec
    _cmd = "wb_command -cifti-parcellate"
