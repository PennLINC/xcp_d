# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling functional connectvity."""
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from nilearn.input_data import NiftiLabelsMasker
from nilearn.plotting import plot_matrix
from nipype import logging
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


class _NiftiConnectInputSpec(BaseInterfaceInputSpec):
    filtered_file = File(exists=True, mandatory=True, desc="filtered file")
    mask = File(exists=True, mandator=True, desc="brain mask file")
    atlas = File(exists=True, mandatory=True, desc="atlas file")


class _NiftiConnectOutputSpec(TraitedSpec):
    time_series_tsv = File(exists=True, mandatory=True, desc=" time series file")
    fcon_matrix_tsv = File(exists=True, mandatory=True, desc=" time series file")
    parcel_coverage_file = File(exists=True, mandatory=True, desc="Parcel-wise coverage file.")


class NiftiConnect(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _NiftiConnectInputSpec
    output_spec = _NiftiConnectOutputSpec

    def _run_interface(self, runtime):
        filtered_file = self.inputs.filtered_file
        mask = self.inputs.mask
        atlas = self.inputs.atlas

        self._results["time_series_tsv"] = fname_presuffix(
            filtered_file,
            suffix="time_series.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["fcon_matrix_tsv"] = fname_presuffix(
            filtered_file,
            suffix="fcon_matrix.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["parcel_coverage_file"] = fname_presuffix(
            filtered_file,
            suffix="parcel_coverage_file.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )

        coverage_threshold = 0.5

        # Before anything, we need to measure coverage
        atlas_img = nb.load(atlas)
        atlas_data = atlas_img.get_fdata()
        atlas_data_bin = (atlas_data > 0).astype(np.float32)
        atlas_img_bin = nb.Nifti1Image(atlas_data_bin, atlas_img.affine, atlas_img.header)

        sum_masker_masked = NiftiLabelsMasker(
            labels_img=atlas,
            mask_img=mask,
            smoothing_fwhm=None,
            standardize=False,
            strategy="sum",
        )
        sum_masker_unmasked = NiftiLabelsMasker(
            labels_img=atlas,
            smoothing_fwhm=None,
            standardize=False,
            strategy="sum",
        )
        n_voxels_in_masked_parcels = sum_masker_masked.fit_transform(atlas_img_bin)
        n_voxels_in_parcels = sum_masker_unmasked.fit_transform(atlas_img_bin)
        parcel_coverage = np.squeeze(n_voxels_in_masked_parcels / n_voxels_in_parcels)
        coverage_thresholded = parcel_coverage < coverage_threshold

        n_nodes = coverage_thresholded.size
        n_bad_nodes = np.sum(parcel_coverage == 0)
        n_poor_nodes = np.sum(np.logical_and(parcel_coverage > 0, parcel_coverage < 0.5))
        n_partial_nodes = np.sum(np.logical_and(parcel_coverage >= 0.5, parcel_coverage < 1))

        if n_bad_nodes:
            LOGGER.warning(f"{n_bad_nodes}/{n_nodes} of parcels have 0% coverage.")

        if n_poor_nodes:
            LOGGER.warning(
                f"{n_poor_nodes}/{n_nodes} of parcels have <50% coverage. "
                "These parcels' time series will be replaced with zeros."
            )

        if n_partial_nodes:
            LOGGER.warning(
                f"{n_partial_nodes}/{n_nodes} of parcels have at least one uncovered "
                "voxel, but have enough good voxels to be useable. "
                "The bad voxels will be ignored and the parcels' time series will be "
                "calculated from the remaining voxels."
            )

        masker = NiftiLabelsMasker(
            labels_img=atlas,
            mask_img=mask,
            smoothing_fwhm=None,
            standardize=False,
        )

        # Use nilearn for time_series
        time_series = masker.fit_transform(filtered_file)

        # Apply the coverage mask
        time_series[:, coverage_thresholded] = 0

        # Use numpy for correlation matrix
        correlation_matrices = np.corrcoef(time_series.T)

        np.savetxt(self._results["time_series_tsv"], time_series, delimiter="\t")
        np.savetxt(self._results["fcon_matrix_tsv"], correlation_matrices, delimiter="\t")
        np.savetxt(self._results["parcel_coverage_file"], parcel_coverage, delimiter="\t")

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
