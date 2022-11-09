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

LOGGER = logging.getLogger("nipype.interface")
# nifti functional connectivity


class _NiftiConnectInputSpec(BaseInterfaceInputSpec):
    filtered_file = File(exists=True, mandatory=True, desc="filtered file")
    atlas = File(exists=True, mandatory=True, desc="atlas file")


class _NiftiConnectOutputSpec(TraitedSpec):
    time_series_tsv = File(exists=True, mandatory=True, desc=" time series file")
    fcon_matrix_tsv = File(exists=True, mandatory=True, desc=" time series file")


class NiftiConnect(SimpleInterface):
    """Extract timeseries and compute connectivity matrices."""

    input_spec = _NiftiConnectInputSpec
    output_spec = _NiftiConnectOutputSpec

    def _run_interface(self, runtime):
        # Write out time series using Nilearn's NiftiLabelMasker
        # Then write out functional correlation matrix of
        # timeseries using numpy.
        self._results["time_series_tsv"] = fname_presuffix(
            self.inputs.filtered_file, suffix="time_series.tsv", newpath=runtime.cwd, use_ext=False
        )
        self._results["fcon_matrix_tsv"] = fname_presuffix(
            self.inputs.filtered_file, suffix="fcon_matrix.tsv", newpath=runtime.cwd, use_ext=False
        )

        (
            self._results["time_series_tsv"],
            self._results["fcon_matrix_tsv"],
        ) = extract_timeseries_funct(
            in_file=self.inputs.filtered_file,
            atlas=self.inputs.atlas,
            timeseries=self._results["time_series_tsv"],
            fconmatrix=self._results["fcon_matrix_tsv"],
        )
        return runtime


class _ApplyTransformsInputSpec(ApplyTransformsInputSpec):
    transforms = InputMultiObject(
        traits.Either(File(exists=True), "identity"),
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
        self.inputs.output_image = fname_presuffix(
            self.inputs.input_image, suffix="_trans.nii.gz", newpath=runtime.cwd, use_ext=False
        )
        runtime = super(ApplyTransformsx, self)._run_interface(runtime)
        return runtime


class _ConnectPlotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="bold file")
    atlas_names = InputMultiObject(
        traits.Str,
        mandatory=True,
        desc="List of atlases. Aligned with the list of time series in time_series_tsv.",
    )
    time_series_tsv = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="List of TSV file with time series. Aligned with the list of atlases in atlas_names",
    )


class _ConnectPlotOutputSpec(TraitedSpec):
    connectplot = File(
        exists=True,
        mandatory=True,
        desc="Path to SVG file with four correlation heat maps.",
    )


class ConnectPlot(SimpleInterface):
    """Extract timeseries and compute connectivity matrices."""

    input_spec = _ConnectPlotInputSpec
    output_spec = _ConnectPlotOutputSpec

    def _run_interface(self, runtime):
        ATLAS_LOOKUP = {
            "Schaefer217": {
                "title": "schaefer 200  17 networks",
                "axes": [0, 0],
            },
            "Schaefer417": {
                "title": "schaefer 400  17 networks",
                "axes": [0, 1],
            },
            "Gordon": {
                "title": "Gordon 333",
                "axes": [1, 0],
            },
            "Glasser": {
                "title": "Glasser 360",
                "axes": [1, 1],
            },
        }

        # Generate a plot of each matrix's correlation coefficients
        fig, axes = plt.subplots(2, 2)
        fig.set_size_inches(20, 20)
        font = {"weight": "normal", "size": 20}

        for atlas_name, subdict in ATLAS_LOOKUP.items():
            atlas_idx = self.inputs.atlas_names.index(atlas_name)
            atlas_file = self.inputs.time_series_tsv[atlas_idx]

            if self.inputs.in_file.endswith("dtseries.nii"):  # for cifti
                #  Get the correlation coefficient of the data
                corrs = np.corrcoef(nb.load(atlas_file).get_fdata().T)

            else:  # for nifti
                #  Get the correlation coefficient of the data
                corrs = np.corrcoef(np.loadtxt(atlas_file, delimiter="\t").T)

            plot_matrix(
                mat=corrs,
                colorbar=False,
                vmax=1,
                vmin=-1,
                axes=axes[subdict["axes"][0], subdict["axes"][1]],
            )
            axes[subdict["axes"][0], subdict["axes"][1]].set_title(
                subdict["title"],
                fontdict=font,
            )

        # Write the results out
        self._results["connectplot"] = fname_presuffix(
            "connectivityplot", suffix="_matrixplot.svg", newpath=runtime.cwd, use_ext=False
        )

        fig.savefig(self._results["connectplot"], bbox_inches="tight", pad_inches=None)

        return runtime
