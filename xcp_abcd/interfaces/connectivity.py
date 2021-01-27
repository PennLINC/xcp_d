# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling functional connectvity.
    .. testsetup::
    # will comeback
"""
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)
LOGGER = logging.getLogger('nipype.interface')
from utils import extract_timeseries_funct

# nifti functional connectivity

class _nifticonnectInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="nifti preprocessed file")
    regressed_file = File(exists=True,mandatory=True, desc="regressed file")
    atlas = File(exists=True,mandatory=True, desc="atlas file")

class _nifticonnectOutputSpec(TraitedSpec):
    time_series_tsv = File(exists=True, manadatory=True,
                                  desc=" time series file")
    fcon_matrix_tsv = File(exists=True, manadatory=True,
                                  desc=" time series file")


class nifticonnect(SimpleInterface):
    r"""
    coming back 
    """
    input_spec = _nifticonnectInputSpec
    output_spec = _nifticonnectOutputSpec
    
    def _run_interface(self, runtime):
     
        self._results['time_series_tsv'] = fname_presuffix(
                self.inputs.in_file,
                suffix='time_series', newpath=runtime.cwd,
                use_ext=False)
        self._results['fcon_matrix_tsv'] = fname_presuffix(
                self.inputs.in_file,
                suffix='fcon_matrix', newpath=runtime.cwd,
                use_ext=False)
    
        self._results['time_series_tsv'],self._results['fcon_matrix_tsv'] = extract_timeseries_funct( 
                                 in_file=self.inputs.regressed_file,
                                 atlas=self.inputs.atlas,
                                 timeseries=self._results['time_series_tsv'],
                                 fconmatrix=self._results['fcon_matrix_tsv'])


