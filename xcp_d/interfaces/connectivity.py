# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling functional connectvity."""
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
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
from xcp_d.utils.write_save import get_cifti_intents

LOGGER = logging.getLogger("nipype.interface")


class _NiftiConnectInputSpec(BaseInterfaceInputSpec):
    filtered_file = File(exists=True, mandatory=True, desc="filtered file")
    mask = File(exists=True, mandator=True, desc="brain mask file")
    atlas = File(exists=True, mandatory=True, desc="atlas file")
    atlas_labels = File(exists=True, mandatory=True, desc="atlas labels file")


class _NiftiConnectOutputSpec(TraitedSpec):
    timeseries = File(exists=True, mandatory=True, desc=" time series file")
    correlations = File(exists=True, mandatory=True, desc=" time series file")
    coverage = File(exists=True, mandatory=True, desc="Parcel-wise coverage file.")


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
        atlas_labels = self.inputs.atlas_labels

        node_labels_df = pd.read_table(atlas_labels, index_col="index")

        # Explicitly remove label corresponding to background (index=0), if present.
        if 0 in node_labels_df.index:
            node_labels_df = node_labels_df.drop(index=[0])

        node_labels = node_labels_df["name"].tolist()

        self._results["timeseries"] = fname_presuffix(
            filtered_file,
            suffix="_timeseries.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["correlations"] = fname_presuffix(
            filtered_file,
            suffix="_corr.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["coverage"] = fname_presuffix(
            filtered_file,
            suffix="_coverage.tsv",
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
            labels=node_labels,
            mask_img=mask,
            smoothing_fwhm=None,
            standardize=False,
            strategy="sum",
            resampling_target=None,  # they should be in the same space/resolution already
        )
        sum_masker_unmasked = NiftiLabelsMasker(
            labels_img=atlas,
            labels=node_labels,
            smoothing_fwhm=None,
            standardize=False,
            strategy="sum",
            resampling_target=None,  # they should be in the same space/resolution already
        )
        n_voxels_in_masked_parcels = sum_masker_masked.fit_transform(atlas_img_bin)
        n_voxels_in_parcels = sum_masker_unmasked.fit_transform(atlas_img_bin)
        parcel_coverage = np.squeeze(n_voxels_in_masked_parcels / n_voxels_in_parcels)
        coverage_thresholded = parcel_coverage < coverage_threshold

        n_nodes = len(node_labels)
        n_found_nodes = coverage_thresholded.size
        n_bad_nodes = np.sum(parcel_coverage == 0)
        n_poor_nodes = np.sum(np.logical_and(parcel_coverage > 0, parcel_coverage < 0.5))
        n_partial_nodes = np.sum(np.logical_and(parcel_coverage >= 0.5, parcel_coverage < 1))

        if n_found_nodes != n_nodes:
            LOGGER.warning(
                f"{n_nodes - n_found_nodes}/{n_nodes} of parcels not found in atlas file."
            )

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
            labels=node_labels,
            mask_img=mask,
            smoothing_fwhm=None,
            standardize=False,
            resampling_target=None,  # they should be in the same space/resolution already
        )

        # Use nilearn for time_series
        timeseries_arr = masker.fit_transform(filtered_file)
        assert timeseries_arr.shape[1] == n_found_nodes

        # Apply the coverage mask
        timeseries_arr[:, coverage_thresholded] = np.nan

        # Region indices in the atlas may not be sequential, so we map them to sequential ints.
        seq_mapper = {idx: i for i, idx in enumerate(masker.labels_)}

        # Fill in any missing nodes with NaNs.
        if n_found_nodes != n_nodes:
            new_timeseries_arr = np.full(
                (timeseries_arr.shape[0], n_nodes),
                fill_value=np.nan,
                dtype=timeseries_arr.dtype,
            )
            for col in range(timeseries_arr.shape[1]):
                label_col = seq_mapper[masker.labels_[col]]
                new_timeseries_arr[:, label_col] = timeseries_arr[:, col]

            timeseries_arr = new_timeseries_arr

        # The time series file is tab-delimited, with node names included in the first row.
        timeseries_df = pd.DataFrame(data=timeseries_arr, columns=node_labels)
        correlations_df = timeseries_df.corr()
        coverage_df = pd.DataFrame(data=parcel_coverage, index=node_labels)

        timeseries_df.to_csv(self._results["timeseries"], sep="\t", index=False)
        correlations_df.to_csv(self._results["correlations"], sep="\t", index_label="Node")
        coverage_df.to_csv(self._results["coverage"], sep="\t", index_label="Node")

        return runtime


class _CiftiConnectInputSpec(BaseInterfaceInputSpec):
    ptseries = File(exists=True, mandatory=True, desc="Timeseries ptseries.nii file.")
    atlas_labels = File(exists=True, mandatory=True, desc="atlas labels file")


class _CiftiConnectOutputSpec(TraitedSpec):
    timeseries_tsv = File(exists=True, mandatory=True, desc="Timeseries tsv file.")
    pconn = File(exists=True, mandatory=True, desc="Correlation matrix pconn.nii file.")
    correlations = File(exists=True, mandatory=True, desc="Correlation matrix tsv file.")


class CiftiConnect(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _NiftiConnectInputSpec
    output_spec = _NiftiConnectOutputSpec

    def _run_interface(self, runtime):
        ptseries = self.inputs.ptseries
        atlas_labels = self.inputs.atlas_labels

        node_labels_df = pd.read_table(atlas_labels, index_col="index")

        # Explicitly remove label corresponding to background (index=0), if present.
        if 0 in node_labels_df.index:
            node_labels_df = node_labels_df.drop(index=[0])

        node_labels = node_labels_df["name"].tolist()
        expected_cifti_node_labels = node_labels_df["cifti_name"].tolist()

        ptseries_img = nb.load(ptseries)
        timeseries_arr = ptseries_img.get_fdata()

        # CiftiParcellate should fill missing
        assert "ConnParcelSries" in ptseries_img.nifti_header.get_intent()
        assert len(node_labels) == timeseries_arr.shape[1]

        correlations = np.corrcoef(timeseries_arr.T)
        parcels_axis = ptseries_img.header.get_axis(1)
        new_header = nb.cifti2.Cifti2Header.from_axes((parcels_axis, parcels_axis))
        conn_img = nb.Cifti2Image(correlations, new_header, nifti_header=ptseries_img.nifti_header)
        conn_img.nifti_header.set_intent(get_cifti_intents()[".pconn.nii"])

        # Load node names from CIFTI file.
        # First axis should be time, second should be parcels
        ax = ptseries_img.header.get_axis(1)
        detected_node_labels = ax.name

        # If there are nodes in the CIFTI that aren't in the node labels file, raise an error.
        found_but_not_expected = sorted(
            list(set(detected_node_labels) - set(expected_cifti_node_labels))
        )
        expected_but_not_found = sorted(
            list(set(expected_cifti_node_labels) - set(detected_node_labels))
        )
        error_msg = ""
        if found_but_not_expected:
            error_msg += (
                "Mismatch found between atlas nodes and node labels file: "
                f"{', '.join(found_but_not_expected)}\n"
            )

        if expected_but_not_found:
            error_msg += (
                "Mismatch found between node labels file and atlas nodes: "
                f"{', '.join(expected_but_not_found)}\n"
            )

        if error_msg:
            raise ValueError(error_msg)

        self._results["timeseries_tsv"] = fname_presuffix(
            ptseries,
            suffix="_timeseries.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["pconn"] = fname_presuffix(
            ptseries,
            suffix="_corr.pconn.nii",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["correlations"] = fname_presuffix(
            ptseries,
            suffix="_corr.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )

        # Place the data in a DataFrame and save to a TSV
        df = pd.DataFrame(columns=node_labels, data=timeseries_arr)
        df.to_csv(timeseries_file, index=False, sep="\t")

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
