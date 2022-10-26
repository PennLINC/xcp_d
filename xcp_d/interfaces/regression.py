# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Regression interfaces."""
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.write_save import despikedatacifti

LOGGER = logging.getLogger('nipype.interface')


class _CiftiDespikeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc=" cifti  file ")
    TR = traits.Float(exists=True, mandatory=True, desc="repetition time")


class _CiftiDespikeOutputSpec(TraitedSpec):
    des_file = File(exists=True, mandatory=True, desc=" despike cifti")


class CiftiDespike(SimpleInterface):
    """Despike a CIFTI file."""

    input_spec = _CiftiDespikeInputSpec
    output_spec = _CiftiDespikeOutputSpec

    def _run_interface(self, runtime):

        # write the output out
        self._results['des_file'] = fname_presuffix(
            'ciftidepike',
            suffix='.dtseries.nii',
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results['des_file'] = despikedatacifti(cifti=self.inputs.in_file,
                                                     TR=self.inputs.TR,
                                                     basedir=runtime.cwd)
        return runtime
