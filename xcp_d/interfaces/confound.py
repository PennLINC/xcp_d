# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
selection of confound matrices
    .. testsetup::
    # will comeback
"""
import pandas as pd
import numpy as np
from ..utils import load_confound_matrix
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface
)


LOGGER = logging.getLogger('nipype.interface')


class _confoundInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="Input file: either cifti or nifti file from fMRI directory")
    confounds_file = File(
        exists=True,
        mandatory=True,
        desc="Confounds file (tsv) from fmriprep")
    confounds_json = File(
        exists=True,
        mandatory=True,
        desc="Confounds file (json) from fmriprep")
    head_radius = traits.Float(
        exits=True,
        mandatory=False,
        default_value=50,
        desc="head radius for to convert rotxyz to arc length for baby, 35m is recommended")
    TR = traits.Float(
        mandatory=True, 
        desc='repetition time in seconds')
    filtertype = traits.Enum(
        'lp', 'notch',
        desc='filter type for filtering regressors, either lp or notch')
    filterorder = traits.Int(
        default=4,
        usedefault=True, 
        desc='motion filter order')
    cutoff = traits.Float(
        default=12, 
        usedefault=True,
        desc='cutoff frequency for lp filter in breathe per min (bpm)')
    low_freq = traits.Float(
        default=12,
        usedefault=True,
        desc='low frequency band for notch filter in breaths per min (bpm)')
    high_freq = traits.Float(
        default=16,
        usedefault=True,
        desc='high frequency for notch filter in breaths per min (bpm)')
    custom_conf = traits.Either(
        traits.Undefined, File,
        desc="name of output file with field or true")
    params = traits.Enum(
        '24P', '27P', '36P',
        mandatory=True,
        desc="nuissance confound model from Ciric etal 2017 \
            24P: (6P + their derivative) and their square , \
            27P: 24P + 2P + global signal \
            36P: (9P + their derivative) and their square ")


class _confoundOutputSpec(TraitedSpec):
    confound_file = File(
        exists=True,
        manadatory=True,
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

        cutoff = np.float(0.1)
        freqband = np.array([self.inputs.low_freq, self.inputs.high_freq])/60.

        # get the nifti/cifti into  matrix
        data_matrix = load_confound_matrix(
            datafile=self.inputs.in_file,
            confounds_file=self.inputs.confounds_file,
            confounds_json=self.inputs.confounds_json,
            filtertype=self.inputs.filtertype,
            custom_conf=self.inputs.custom_conf,
            freqband=freqband,
            cutoff=cutoff,
            params=self.inputs.params,
            TR=self.inputs.TR,
            order=self.inputs.filterorder)

        # write the output out
        self._results['confound_file'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_confound_matrix.tsv', newpath=runtime.cwd,
                use_ext=False)
        datax = pd.DataFrame(data_matrix)
        datax.to_csv(self._results['confound_file'], header=None, index=None)
        return runtime
