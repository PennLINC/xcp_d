# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
selection of confound matrices
    .. testsetup::
    # will comeback
"""

from ..utils import load_confound_matrix
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)
from ..utils import(read_ndata, write_ndata)
import pandas as pd

LOGGER = logging.getLogger('nipype.interface') 

class _confoundInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="Input file: either cifti or nifti file from \
                                  fMRIPrep directory")
    head_radius = traits.Float(exits=True,mandatory=False,default_value=50,desc=" head raidus for to convert rotxyz to arc length \
                                               for baby, 35m is recommended")
    params = traits.Str(exists=True,mandatory=True, 
                            default_value='24P',desc= "nuissance confound model from Ciric etal 2017 \
                             24P: (6P + their derivative) and their square , \
                             27P: 24P + 2P + global signal \
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
                        params=self.inputs.params,head_radius=self.inputs.head_radius)
        #write the output out
        self._results['confound_file'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_confound_matrix.tsv', newpath=runtime.cwd,
                use_ext=False)
        datax = pd.DataFrame(data_matrix)
        datax.to_csv(self._results['confound_file'] , header=None,index=None)
        return runtime