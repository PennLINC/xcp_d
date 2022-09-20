# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling functional connectvity.

.. testsetup::
# will comeback
"""
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from nilearn.plotting import plot_matrix
from nipype import logging
from nipype.interfaces.ants.resampling import ApplyTransforms, ApplyTransformsInputSpec
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from xcp_d.utils.fcon import extract_timeseries_funct
from xcp_d.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')
# nifti functional connectivity


class _NiftiConnectInputSpec(BaseInterfaceInputSpec):
    filtered_file = File(exists=True, mandatory=True, desc="filtered file")
    atlas = File(exists=True, mandatory=True, desc="atlas file")


class _NiftiConnectOutputSpec(TraitedSpec):
    time_series_tsv = File(exists=True,
                           manadatory=True,
                           desc=" time series file")
    fcon_matrix_tsv = File(exists=True,
                           manadatory=True,
                           desc=" time series file")


class NiftiConnect(SimpleInterface):
    """Extract timeseries and compute connectivity matrices."""

    input_spec = _NiftiConnectInputSpec
    output_spec = _NiftiConnectOutputSpec

    def _run_interface(self, runtime):
        # Write out time series using Nilearn's NiftiLabelMasker
        # Then write out functional correlation matrix of
        # timeseries using numpy.
        self._results['time_series_tsv'] = fname_presuffix(
            self.inputs.filtered_file,
            suffix='time_series.tsv',
            newpath=runtime.cwd,
            use_ext=False)
        self._results['fcon_matrix_tsv'] = fname_presuffix(
            self.inputs.filtered_file,
            suffix='fcon_matrix.tsv',
            newpath=runtime.cwd,
            use_ext=False)

        self._results['time_series_tsv'], self._results['fcon_matrix_tsv'] = \
            extract_timeseries_funct(
                in_file=self.inputs.filtered_file,
                atlas=self.inputs.atlas,
                timeseries=self._results['time_series_tsv'],
                fconmatrix=self._results['fcon_matrix_tsv'])
        return runtime


class _ApplyTransformsInputSpec(ApplyTransformsInputSpec):
    transforms = InputMultiObject(
        traits.Either(File(exists=True), 'identity'),
        argstr="%s",
        mandatory=True,
        desc="transform files",
    )


class ApplyTransformsx(ApplyTransforms):
    """ApplyTransforms from nipype as workflow.

    This is a modification of the ApplyTransforms interface,
    with an updated set of inputs and a different default output image name.
    """

    input_spec = _ApplyTransformsInputSpec

    def _run_interface(self, runtime):
        # Run normally
        self.inputs.output_image = fname_presuffix(self.inputs.input_image,
                                                   suffix='_trans.nii.gz',
                                                   newpath=runtime.cwd,
                                                   use_ext=False)
        runtime = super(ApplyTransformsx, self)._run_interface(runtime)
        return runtime


class _ConnectplotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="bold file")
    sc217_timeseries = File(exists=True, mandatory=True, desc="sc217 atlas")
    sc417_timeseries = File(exists=True, mandatory=True, desc="sc417 atlas")
    gd333_timeseries = File(exists=True, mandatory=True, desc="gordon atlas")
    gs360_timeseries = File(exists=True, mandatory=True, desc="glasser atlas")


class _ConnectPlotOutputSpec(TraitedSpec):
    connectplot = File(
        exists=True,
        manadatory=True,
    )


class ConnectPlot(SimpleInterface):
    """Extract timeseries and compute connectivity matrices."""

    input_spec = _ConnectPlotInputSpec
    output_spec = _ConnectPlotOutputSpec

    def _run_interface(self, runtime):

        if self.inputs.in_file.endswith('dtseries.nii'):  # for cifti
            #  Get the correlation coefficient of the data
            sc217 = np.corrcoef(
                nb.load(self.inputs.sc217_timeseries).get_fdata().T)
            sc417 = np.corrcoef(
                nb.load(self.inputs.sc417_timeseries).get_fdata().T)
            gd333 = np.corrcoef(
                nb.load(self.inputs.gd333_timeseries).get_fdata().T)
            gs360 = np.corrcoef(
                nb.load(self.inputs.gs360_timeseries).get_fdata().T)

        else:  # for nifti
            #  Get the correlation coefficient of the data
            sc217 = np.corrcoef(
                np.loadtxt(self.inputs.sc217_timeseries, delimiter=',').T)
            sc417 = np.corrcoef(
                np.loadtxt(self.inputs.sc417_timeseries, delimiter=',').T)
            gd333 = np.corrcoef(
                np.loadtxt(self.inputs.gd333_timeseries, delimiter=',').T)
            gs360 = np.corrcoef(
                np.loadtxt(self.inputs.gs360_timeseries, delimiter=',').T)

        # Generate a plot of each matrix's correlation coefficients
        fig, ax1 = plt.subplots(2, 2)
        fig.set_size_inches(20, 20)
        font = {'weight': 'normal', 'size': 20}
        plot_matrix(mat=sc217, colorbar=False, vmax=1, vmin=-1, axes=ax1[0, 0])
        ax1[0, 0].set_title('schaefer 200  17 networks', fontdict=font)
        plot_matrix(mat=sc417, colorbar=False, vmax=1, vmin=-1, axes=ax1[0, 1])
        ax1[0, 1].set_title('schaefer 400  17 networks', fontdict=font)
        plot_matrix(mat=gd333, colorbar=False, vmax=1, vmin=-1, axes=ax1[1, 0])
        ax1[1, 0].set_title('Gordon 333', fontdict=font)
        plot_matrix(mat=gs360, colorbar=False, vmax=1, vmin=-1, axes=ax1[1, 1])
        ax1[1, 1].set_title('Glasser 360', fontdict=font)

        # Write the results out
        self._results['connectplot'] = fname_presuffix(
            'connectivityplot',
            suffix='_matrixplot.svg',
            newpath=runtime.cwd,
            use_ext=False)

        fig.savefig(self._results['connectplot'],
                    bbox_inches="tight",
                    pad_inches=None)

        return runtime
