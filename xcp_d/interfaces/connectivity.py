# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling functional connectvity."""
import matplotlib.pyplot as plt
import pandas as pd
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

from xcp_d.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger("nipype.interface")


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
        desc="List of atlases. Aligned with the list of time series in correlation_tsvs.",
    )
    correlation_tsvs = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc=(
            "List of TSV file with correlation matrices. "
            "Aligned with the list of atlases in atlas_names."
        ),
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
            atlas_file = self.inputs.correlation_tsvs[atlas_idx]

            correlations_df = pd.read_table(atlas_file, index_col="Node")

            plot_matrix(
                mat=correlations_df.to_numpy(),
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
