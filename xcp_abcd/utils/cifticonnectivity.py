# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""workbench command for  cifti correlation"""

from nipype.interfaces.workbench.base import WBCommand
from nipype.interfaces.base import TraitedSpec, File, traits, CommandLineInputSpec
from nipype import logging
iflogger = logging.getLogger("nipype.interface")


class CiftiCorrelationInputSpec(CommandLineInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="The input ptseries or dense series",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="correlation_matrix_%s.nii",
        keep_extension=True,
        argstr=" %s",
        position=1,
        desc="The output CIFTI",
    )
    
    roi_override = traits.Bool(
        exists=True,
        argstr="-roi-override %s ",
        position=2,
        desc=" perform correlation from a subset of rows to all rows",
    )
    
    left_roi = File(
        exists=True,
        position=3,
        argstr="-left-roi %s",
        desc="Specify the left roi metric  to use",
    )

    right_roi = File(
        exists=True,
        position=5,
        argstr="-right-roi %s",
        desc="Specify the right  roi metric  to use",
    )
    cerebellum_roi = File(
        exists=True,
        position=6,
        argstr="-cerebellum-roi %s",
        desc="specify the cerebellum meytric to use",
    )
    
    vol_roi = File(
        exists=True,
        position=7,
        argstr="-vol-roi %s",
        desc="volume roi to use",
    )

    cifti_roi = File(
        exists=True,
        position=8,
        argstr="-cifti-roi %s",
        desc="cifti roi to use",
    )
    weights_file= File(
        exists=True,
        position=9,
        argstr="-weights %s",
        desc="specify the cerebellum surface  metricto use",
    )

    fisher_ztrans = traits.Bool(
        position=10,
        argstr="-fisher-z",
        desc=" fisherz transfrom",
    )
    no_demean = traits.Bool(
        position=11,
        argstr="-fisher-z",
        desc=" fisherz transfrom",
    )
    compute_covariance = traits.Bool(
        position=12,
        argstr="-covariance ",
        desc=" compute covariance instead of correlation",
    )

class CiftiCorrelationOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output CIFTI file")

class CiftiCorrelation(WBCommand):
    r"""
    Compute correlation from CIFTI file
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .ptseries or .dtseries,  
    >>> cifticorr = CiftiCorrelation()
    >>> cifticorr.inputs.in_file = 'sub-01XX_task-rest.ptseries.nii'
    >>> cifticorr.inputs.out_file = 'sub_01XX_task-rest.pconn.nii'
    >>> cifticorr.cmdline
    wb_command  -cifti-correlation sub-01XX_task-rest.ptseries.nii \
        'sub_01XX_task-rest.pconn.nii'
    """

    input_spec = CiftiCorrelationInputSpec
    output_spec = CiftiCorrelationOutputSpec
    _cmd = "OMP_NUM_THREADS=1 wb_command  -cifti-correlation" 