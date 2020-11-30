# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling confound.
    .. testsetup::
    # will comeback
"""

from utils import load_confound_matrix
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)
from utils import(read_ndata, write_ndata)
import pandas as pd

LOGGER = logging.getLogger('nipype.interface') 

class _confoundInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="Input file: either cifti or nifti file from \
                                  fMRIPrep directory")
    params = traits.Str(exists=True,mandatory=True, 
                            default_value='6P',desc="nuissance regressors from Ciric etal 2017 \
                             2P: wm and csf, 6P: six motion paramters, 9P: 6P + 2P + global signal \
                             24P: (6P + their derivative) and their square , \
                             36P: (9P + their derivative) and their square  ")

class _confoundOutputSpec(TraitedSpec):
    confound_file = File(exists=True, manadatory=True,
                                  desc="confound matrix file")


class ConfoundMatrix(SimpleInterface):
    r"""
    select the confound matrix.
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> conf = ConfoundMatrix()
    >>> conf = ConfoundMatrix()
    >>> conf.inputs.in_file = datafile
    >>> conf.inputs.params = "9P"
    >>> conf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()
    
    """

    input_spec = _confoundInputSpec
    output_spec = _confoundOutputSpec

    def _run_interface(self, runtime):
        
        # get the nifti/cifti into  matrix
        data_matrix = load_confound_matrix(datafile=self.inputs.in_file,
                        params=self.inputs.params)
        #write the output out
        self._results['confound_file'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_confound_matrix.tsv', newpath=runtime.cwd,
                use_ext=False)
        datax = pd.DataFrame(data_matrix)
        datax.to_csv(self._results['confound_file'] , header=None,index=None)
        return runtime