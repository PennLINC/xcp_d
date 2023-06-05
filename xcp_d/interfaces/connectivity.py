# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling functional connectvity."""
import warnings

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
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
from xcp_d.utils.write_save import get_cifti_intents

LOGGER = logging.getLogger("nipype.interface")


class _NiftiConnectInputSpec(BaseInterfaceInputSpec):
    filtered_file = File(exists=True, mandatory=True, desc="filtered file")
    mask = File(exists=True, mandatory=True, desc="brain mask file")
    atlas = File(exists=True, mandatory=True, desc="atlas file")
    atlas_labels = File(exists=True, mandatory=True, desc="atlas labels file")
    min_coverage = traits.Float(
        default=0.5,
        usedefault=True,
        desc=(
            "Coverage threshold to apply to parcels. "
            "Any parcels with lower coverage than the threshold will be replaced with NaNs. "
            "Must be a value between zero and one. "
            "Default is 0.5."
        ),
    )


class _NiftiConnectOutputSpec(TraitedSpec):
    timeseries = File(exists=True, mandatory=True, desc="Parcellated time series file.")
    correlations = File(exists=True, mandatory=True, desc="Correlation matrix file.")
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
        min_coverage = self.inputs.min_coverage

        node_labels_df = pd.read_table(atlas_labels, index_col="index")
        node_labels_df.sort_index(inplace=True)  # ensure index is in order

        # Explicitly remove label corresponding to background (index=0), if present.
        if 0 in node_labels_df.index:
            LOGGER.warning(
                "Index value of 0 found in atlas labels file. "
                "Will assume this describes the background and ignore it."
            )
            node_labels_df = node_labels_df.drop(index=[0])

        node_labels = node_labels_df["name"].tolist()

        self._results["timeseries"] = fname_presuffix(
            "timeseries.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        self._results["correlations"] = fname_presuffix(
            "correlations.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        self._results["coverage"] = fname_presuffix(
            "coverage.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )

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
        coverage_thresholded = parcel_coverage < min_coverage

        n_nodes = len(node_labels)
        n_found_nodes = coverage_thresholded.size
        n_bad_nodes = np.sum(parcel_coverage == 0)
        n_poor_parcels = np.sum(
            np.logical_and(parcel_coverage > 0, parcel_coverage < min_coverage)
        )
        n_partial_parcels = np.sum(
            np.logical_and(parcel_coverage >= min_coverage, parcel_coverage < 1)
        )

        if n_found_nodes != n_nodes:
            LOGGER.warning(
                f"{n_nodes - n_found_nodes}/{n_nodes} of parcels not found in atlas file."
            )

        if n_bad_nodes:
            LOGGER.warning(f"{n_bad_nodes}/{n_nodes} of parcels have 0% coverage.")

        if n_poor_parcels:
            LOGGER.warning(
                f"{n_poor_parcels}/{n_nodes} of parcels have <50% coverage. "
                "These parcels' time series will be replaced with zeros."
            )

        if n_partial_parcels:
            LOGGER.warning(
                f"{n_partial_parcels}/{n_nodes} of parcels have at least one uncovered "
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
        seq_mapper = {idx: i for i, idx in enumerate(node_labels_df.index.tolist())}

        if n_found_nodes != n_nodes:  # parcels lost by warping/downsampling atlas
            # Fill in any missing nodes in the timeseries array with NaNs.
            new_timeseries_arr = np.full(
                (timeseries_arr.shape[0], n_nodes),
                fill_value=np.nan,
                dtype=timeseries_arr.dtype,
            )
            for col in range(timeseries_arr.shape[1]):
                label_col = seq_mapper[masker.labels_[col]]
                new_timeseries_arr[:, label_col] = timeseries_arr[:, col]

            timeseries_arr = new_timeseries_arr

            # Fill in any missing nodes in the coverage array with zero.
            new_parcel_coverage = np.zeros(n_nodes, dtype=parcel_coverage.dtype)
            for row in range(parcel_coverage.shape[0]):
                label_row = seq_mapper[masker.labels_[row]]
                new_parcel_coverage[label_row] = parcel_coverage[row]

            parcel_coverage = new_parcel_coverage

        # The time series file is tab-delimited, with node names included in the first row.
        timeseries_df = pd.DataFrame(data=timeseries_arr, columns=node_labels)
        correlations_df = timeseries_df.corr()
        coverage_df = pd.DataFrame(data=parcel_coverage, index=node_labels, columns=["coverage"])

        timeseries_df.to_csv(self._results["timeseries"], sep="\t", na_rep="n/a", index=False)
        correlations_df.to_csv(
            self._results["correlations"],
            sep="\t",
            na_rep="n/a",
            index_label="Node",
        )
        coverage_df.to_csv(self._results["coverage"], sep="\t", index_label="Node")

        return runtime


class _CiftiConnectInputSpec(BaseInterfaceInputSpec):
    min_coverage = traits.Float(
        default=0.5,
        usedefault=True,
        desc=(
            "Coverage threshold to apply to parcels. "
            "Any parcels with lower coverage than the threshold will be replaced with NaNs. "
            "Must be a value between zero and one. "
            "Default is 0.5."
        ),
    )
    data_file = File(
        exists=True,
        mandatory=True,
        desc="Dense CIFTI time series file to parcellate.",
    )
    atlas_file = File(
        exists=True,
        mandatory=True,
        desc=(
            "Atlas CIFTI file to use to parcellate data_file. "
            "This file must already be resampled to the same structure as data_file."
        ),
    )
    parcellated_atlas = File(
        exists=True,
        mandatory=True,
        desc=(
            "Atlas CIFTI that has been parcellated with itself to make a .pscalar.nii file. "
            "This is just used for its ParcelsAxis."
        ),
    )
    atlas_labels = File(exists=True, mandatory=True, desc="atlas labels file")


class _CiftiConnectOutputSpec(TraitedSpec):
    coverage_ciftis = File(exists=True, mandatory=True, desc="Coverage CIFTI file.")
    timeseries_ciftis = File(
        exists=True,
        mandatory=True,
        desc="Parcellated data ptseries.nii file.",
    )
    correlation_ciftis = File(
        exists=True,
        mandatory=True,
        desc="Correlation matrix pconn.nii file.",
    )
    coverage = File(exists=True, mandatory=True, desc="Coverage tsv file.")
    timeseries = File(exists=True, mandatory=True, desc="Parcellated data tsv file.")
    correlations = File(exists=True, mandatory=True, desc="Correlation matrix tsv file.")


class CiftiConnect(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _CiftiConnectInputSpec
    output_spec = _CiftiConnectOutputSpec

    def _run_interface(self, runtime):
        min_coverage = self.inputs.min_coverage
        data_file = self.inputs.data_file
        atlas_file = self.inputs.atlas_file
        pscalar_file = self.inputs.parcellated_atlas
        atlas_labels = self.inputs.atlas_labels

        cifti_intents = get_cifti_intents()

        assert data_file.endswith(".dtseries.nii"), data_file
        assert atlas_file.endswith(".dlabel.nii"), atlas_file
        assert pscalar_file.endswith(".pscalar.nii"), pscalar_file

        data_img = nb.load(data_file)
        atlas_img = nb.load(atlas_file)
        pscalar_img = nb.load(pscalar_file)
        node_labels_df = pd.read_table(atlas_labels, index_col="index")
        node_labels_df.sort_index(inplace=True)  # ensure index is in order

        data_arr = data_img.get_fdata()
        atlas_arr = np.squeeze(atlas_img.get_fdata())  # first dim is singleton

        # First, replace all bad vertices' time series with NaNs.
        # This way, any partially-covered parcels will have NaNs in the bad portions,
        # so those vertices will be ignored by wb_command -cifti-parcellate.
        bad_vertices_idx = np.where(
            np.all(np.logical_or(data_arr == 0, np.isnan(data_arr)), axis=0)
        )[0]
        data_arr[:, bad_vertices_idx] = np.nan

        # Now we can work to parcellate the data
        label_axis = atlas_img.header.get_axis(0)
        parcels_axis = pscalar_img.header.get_axis(1)
        atlas_label_mapper = label_axis.label[0]
        atlas_label_mapper = {k: v[0] for k, v in atlas_label_mapper.items()}

        if 0 in atlas_label_mapper.keys():
            atlas_label_mapper.pop(0)

        # Explicitly remove label corresponding to background (index=0), if present.
        if 0 in node_labels_df.index:
            LOGGER.warning(
                "Index value of 0 found in atlas labels file. "
                "Will assume this describes the background and ignore it."
            )
            node_labels_df = node_labels_df.drop(index=[0])

        expected_cifti_node_labels = node_labels_df["cifti_name"].tolist()
        parcel_name_mapper = dict(zip(node_labels_df["cifti_name"], node_labels_df["name"]))

        # Load node names from CIFTI file.
        # First axis should be time, second should be parcels
        detected_node_labels = parcels_axis.name

        # If there are nodes in the CIFTI that aren't in the node labels file, raise an error.
        # And vice versa as well.
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

        # Sort labels by corresponding values in atlas.
        # This step is probably unnecessary
        # (the axis labels are probably already sorted by the atlas values),
        # but I wanted to be extra safe.
        # Code from https://stackoverflow.com/a/6618543/2589328.
        sorted_parcel_names = [
            x for _, x in sorted(atlas_label_mapper.items(), key=lambda pair: pair[0])
        ]

        timeseries_arr = np.zeros((data_arr.shape[0], len(atlas_label_mapper)), dtype=np.float32)
        timeseries_df = pd.DataFrame(columns=sorted_parcel_names, data=timeseries_arr)

        coverage_arr = np.zeros((len(atlas_label_mapper), 1), dtype=np.float32)
        coverage_df = pd.DataFrame(
            index=sorted_parcel_names,
            columns=["coverage"],
            data=coverage_arr,
        )

        parcels_in_atlas = []  # list of labels for parcels
        for parcel_val, parcel_name in atlas_label_mapper.items():
            parcel_idx = np.where(atlas_arr == parcel_val)[0]

            if parcel_idx.size:  # parcel is found in atlas
                # Determine which, if any, vertices in the parcel are missing.
                bad_vertices_in_parcel_idx = np.intersect1d(parcel_idx, bad_vertices_idx)

                # Determine the percentage of vertices with good data
                parcel_coverage = 1 - (bad_vertices_in_parcel_idx.size / parcel_idx.size)
                coverage_df.loc[parcel_name, "coverage"] = parcel_coverage

                if parcel_coverage < min_coverage:
                    # If the parcel has >=50% bad data, replace all of the values with zeros.
                    data_arr[:, parcel_idx] = np.nan

                parcels_in_atlas.append(parcel_name)
                parcel_data = data_arr[:, parcel_idx]
                with warnings.catch_warnings():
                    # Ignore warning if calculating mean from only NaNs.
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    label_timeseries = np.nanmean(parcel_data, axis=1)

            else:  # parcel not found in atlas
                # Label was probably erased by downsampling or something.
                label_timeseries = np.full(
                    data_arr.shape[0],
                    fill_value=np.nan,
                    dtype=data_arr.dtype,
                )
                coverage_df.loc[parcel_name, "coverage"] = 0

            timeseries_df[parcel_name] = label_timeseries

        # Use parcel names from tsv file instead of internal CIFTI parcel names for tsvs.
        timeseries_df = timeseries_df.rename(columns=parcel_name_mapper)
        correlations_df = timeseries_df.corr()

        # Save out the coverage tsv
        self._results["coverage"] = fname_presuffix(
            "coverage.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        coverage_df.to_csv(self._results["coverage"], sep="\t", index_label="Node")

        # Save out the timeseries tsv
        self._results["timeseries"] = fname_presuffix(
            "timeseries.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        timeseries_df.to_csv(self._results["timeseries"], sep="\t", na_rep="n/a", index=False)

        # Save out the correlation matrix tsv
        self._results["correlations"] = fname_presuffix(
            "correlations.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        correlations_df.to_csv(
            self._results["correlations"],
            sep="\t",
            na_rep="n/a",
            index_label="Node",
        )

        # Save out the coverage CIFTI
        coverage_img = nb.Cifti2Image(
            coverage_df.to_numpy().T,  # (1 x n_parcels) array
            pscalar_img.header,
            nifti_header=pscalar_img.nifti_header,
        )
        self._results["coverage_ciftis"] = fname_presuffix(
            "coverage.pscalar.nii",
            newpath=runtime.cwd,
            use_ext=True,
        )
        coverage_img.to_filename(self._results["coverage_ciftis"])

        # Save out the timeseries CIFTI
        time_axis = data_img.header.get_axis(0)
        new_header = nb.cifti2.Cifti2Header.from_axes((time_axis, parcels_axis))
        nifti_header = data_img.nifti_header.copy()
        nifti_header.set_intent(cifti_intents[".ptseries.nii"])
        timeseries_img = nb.Cifti2Image(
            timeseries_df.to_numpy(),  # (n_vols x n_parcels) array
            new_header,
            nifti_header=nifti_header,
        )
        self._results["timeseries_ciftis"] = fname_presuffix(
            "timeseries.ptseries.nii",
            newpath=runtime.cwd,
            use_ext=True,
        )
        timeseries_img.to_filename(self._results["timeseries_ciftis"])

        # Save out the correlation matrix CIFTI
        new_header = nb.cifti2.Cifti2Header.from_axes((parcels_axis, parcels_axis))
        nifti_header = nifti_header.copy()
        nifti_header.set_intent(cifti_intents[".pconn.nii"])
        conn_img = nb.Cifti2Image(
            correlations_df.to_numpy(),
            new_header,
            nifti_header=nifti_header,
        )
        self._results["correlation_ciftis"] = fname_presuffix(
            "correlations.pconn.nii",
            newpath=runtime.cwd,
            use_ext=True,
        )
        conn_img.to_filename(self._results["correlation_ciftis"])

        return runtime


class _ConnectPlotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="bold file")
    atlas_names = InputMultiObject(
        traits.Str,
        mandatory=True,
        desc="List of atlases. Aligned with the list of time series in time_series_tsv.",
    )
    correlations_tsv = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc=(
            "List of TSV file with correlation matrices. "
            "Aligned with the list of atlases in atlas_names"
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
            atlas_file = self.inputs.correlations_tsv[atlas_idx]

            corrs_df = pd.read_table(atlas_file, index_col="Node")

            plot_matrix(
                mat=corrs_df.to_numpy(),
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
            "connectivityplot",
            suffix="_matrixplot.svg",
            newpath=runtime.cwd,
            use_ext=False,
        )

        fig.savefig(self._results["connectplot"], bbox_inches="tight", pad_inches=None)

        return runtime
