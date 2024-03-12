# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling functional connectvity."""
import gc
import warnings

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.write_save import get_cifti_intents

LOGGER = logging.getLogger("nipype.interface")


class _NiftiParcellateInputSpec(BaseInterfaceInputSpec):
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


class _NiftiParcellateOutputSpec(TraitedSpec):
    coverage = File(exists=True, desc="Parcel-wise coverage file.")
    timeseries = File(exists=True, desc="Parcellated time series file.")


class NiftiParcellate(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _NiftiParcellateInputSpec
    output_spec = _NiftiParcellateOutputSpec

    def _run_interface(self, runtime):
        mask = self.inputs.mask
        atlas = self.inputs.atlas
        min_coverage = self.inputs.min_coverage

        node_labels_df = pd.read_table(self.inputs.atlas_labels, index_col="index")
        node_labels_df.sort_index(inplace=True)  # ensure index is in order

        # Explicitly remove label corresponding to background (index=0), if present.
        if 0 in node_labels_df.index:
            LOGGER.warning(
                "Index value of 0 found in atlas labels file. "
                "Will assume this describes the background and ignore it."
            )
            node_labels_df = node_labels_df.drop(index=[0])

        node_labels = node_labels_df["label"].tolist()

        # Before anything, we need to measure coverage
        atlas_img = nb.load(atlas)
        atlas_data = atlas_img.get_fdata()
        atlas_data_bin = (atlas_data > 0).astype(np.float32)
        atlas_img_bin = nb.Nifti1Image(atlas_data_bin, atlas_img.affine, atlas_img.header)
        del atlas_img, atlas_data, atlas_data_bin
        gc.collect()

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
        del sum_masker_masked, sum_masker_unmasked, n_voxels_in_masked_parcels, n_voxels_in_parcels
        gc.collect()

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

        # Use nilearn to parcellate the file
        timeseries_arr = masker.fit_transform(self.inputs.filtered_file)
        assert timeseries_arr.shape[1] == n_found_nodes
        masker_labels = masker.labels_[:]
        del masker
        gc.collect()

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
                label_col = seq_mapper[masker_labels[col]]
                new_timeseries_arr[:, label_col] = timeseries_arr[:, col]

            timeseries_arr = new_timeseries_arr
            del new_timeseries_arr
            gc.collect()

            # Fill in any missing nodes in the coverage array with zero.
            new_parcel_coverage = np.zeros(n_nodes, dtype=parcel_coverage.dtype)
            for row in range(parcel_coverage.shape[0]):
                label_row = seq_mapper[masker_labels[row]]
                new_parcel_coverage[label_row] = parcel_coverage[row]

            parcel_coverage = new_parcel_coverage
            del new_parcel_coverage
            gc.collect()

        # The time series file is tab-delimited, with node names included in the first row.
        self._results["timeseries"] = fname_presuffix(
            "timeseries.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        timeseries_df = pd.DataFrame(data=timeseries_arr, columns=node_labels)
        timeseries_df.to_csv(self._results["timeseries"], sep="\t", na_rep="n/a", index=False)

        # Save out the coverage tsv
        coverage_df = pd.DataFrame(
            data=parcel_coverage,
            index=node_labels,
            columns=["coverage"],
        )
        self._results["coverage"] = fname_presuffix(
            "coverage.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        coverage_df.to_csv(self._results["coverage"], sep="\t", na_rep="n/a", index_label="Node")

        return runtime


class _TSVConnectInputSpec(BaseInterfaceInputSpec):
    timeseries = File(exists=True, desc="Parcellated time series TSV file.")
    temporal_mask = File(
        exists=True,
        mandatory=False,
        desc="Temporal mask, after dummy scan removal.",
    )


class _TSVConnectOutputSpec(TraitedSpec):
    correlations = File(exists=True, desc="Correlation matrix file.")
    correlations_exact = traits.Either(
        None,
        traits.List(File(exists=True)),
        desc="Correlation matrix files limited to an exact number of volumes.",
    )


def correlate_timeseries(timeseries, temporal_mask):
    """Correlate timeseries stored in a TSV file."""
    timeseries_df = pd.read_table(timeseries)
    correlations_df = timeseries_df.corr()

    # Create correlation matrices limited to exact scan numbers
    correlations_exact = {}
    if isdefined(temporal_mask):
        censoring_df = pd.read_table(temporal_mask)
        censored_censoring_df = censoring_df.loc[censoring_df["framewise_displacement"] == 0]
        censored_censoring_df.reset_index(drop=True, inplace=True)
        exact_columns = [c for c in censoring_df.columns if c.startswith("exact_")]
        for exact_column in exact_columns:
            exact_timeseries_df = timeseries_df.loc[censored_censoring_df[exact_column] == 0]
            exact_correlations_df = exact_timeseries_df.corr()
            correlations_exact[exact_column] = exact_correlations_df

    return correlations_df, correlations_exact


class TSVConnect(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _TSVConnectInputSpec
    output_spec = _TSVConnectOutputSpec

    def _run_interface(self, runtime):
        correlations_df, correlations_exact = correlate_timeseries(
            self.inputs.timeseries,
            temporal_mask=self.inputs.temporal_mask,
        )

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
        del correlations_df
        gc.collect()

        if not self.inputs.temporal_mask:
            self._results["correlations_exact"] = None
            return runtime

        self._results["correlations_exact"] = []
        for exact_column, exact_correlations_df in correlations_exact.items():
            exact_correlations_file = fname_presuffix(
                f"correlations_{exact_column}.tsv",
                newpath=runtime.cwd,
                use_ext=True,
            )
            exact_correlations_df.to_csv(
                exact_correlations_file,
                sep="\t",
                na_rep="n/a",
                index_label="Node",
            )
            self._results["correlations_exact"].append(exact_correlations_file)

        return runtime


class _CiftiParcellateInputSpec(BaseInterfaceInputSpec):
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
    atlas = File(
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


class _CiftiParcellateOutputSpec(TraitedSpec):
    coverage_ciftis = File(exists=True, desc="Coverage CIFTI file.")
    timeseries_ciftis = File(exists=True, desc="Parcellated data ptseries.nii file.")
    coverage = File(exists=True, desc="Coverage tsv file.")
    timeseries = File(exists=True, desc="Parcellated data tsv file.")


class CiftiParcellate(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _CiftiParcellateInputSpec
    output_spec = _CiftiParcellateOutputSpec

    def _run_interface(self, runtime):
        min_coverage = self.inputs.min_coverage
        data_file = self.inputs.data_file
        atlas = self.inputs.atlas
        pscalar_file = self.inputs.parcellated_atlas
        atlas_labels = self.inputs.atlas_labels

        cifti_intents = get_cifti_intents()

        assert data_file.endswith((".dtseries.nii", ".dscalar.nii")), data_file
        assert atlas.endswith(".dlabel.nii"), atlas
        assert pscalar_file.endswith(".pscalar.nii"), pscalar_file

        data_img = nb.load(data_file)
        atlas_img = nb.load(atlas)
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

        if 0 in atlas_label_mapper:
            atlas_label_mapper.pop(0)

        # Explicitly remove label corresponding to background (index=0), if present.
        if 0 in node_labels_df.index:
            LOGGER.warning(
                "Index value of 0 found in atlas labels file. "
                "Will assume this describes the background and ignore it."
            )
            node_labels_df = node_labels_df.drop(index=[0])

        if "cifti_label" in node_labels_df.columns:
            expected_cifti_node_labels = node_labels_df["cifti_label"].tolist()
            parcel_label_mapper = dict(zip(node_labels_df["cifti_label"], node_labels_df["label"]))
        elif "label_7network" in node_labels_df.columns:
            node_labels_df["label_7network"] = node_labels_df["label_7network"].fillna(
                node_labels_df["label"]
            )
            expected_cifti_node_labels = node_labels_df["label_7network"].tolist()
            parcel_label_mapper = dict(
                zip(node_labels_df["label_7network"], node_labels_df["label"])
            )
        else:
            raise Exception(atlas_labels)

        # Load node labels from CIFTI file.
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
        sorted_parcel_labels = [
            x for _, x in sorted(atlas_label_mapper.items(), key=lambda pair: pair[0])
        ]

        timeseries_arr = np.zeros((data_arr.shape[0], len(atlas_label_mapper)), dtype=np.float32)
        timeseries_df = pd.DataFrame(columns=sorted_parcel_labels, data=timeseries_arr)

        coverage_arr = np.zeros((len(atlas_label_mapper), 1), dtype=np.float32)
        coverage_df = pd.DataFrame(
            index=sorted_parcel_labels,
            columns=["coverage"],
            data=coverage_arr,
        )

        parcels_in_atlas = []  # list of labels for parcels
        for parcel_val, parcel_label in atlas_label_mapper.items():
            parcel_idx = np.where(atlas_arr == parcel_val)[0]

            if parcel_idx.size:  # parcel is found in atlas
                # Determine which, if any, vertices in the parcel are missing.
                bad_vertices_in_parcel_idx = np.intersect1d(parcel_idx, bad_vertices_idx)

                # Determine the percentage of vertices with good data
                parcel_coverage = 1 - (bad_vertices_in_parcel_idx.size / parcel_idx.size)
                coverage_df.loc[parcel_label, "coverage"] = parcel_coverage

                if parcel_coverage < min_coverage:
                    # If the parcel has >=50% bad data, replace all of the values with zeros.
                    data_arr[:, parcel_idx] = np.nan

                parcels_in_atlas.append(parcel_label)
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
                coverage_df.loc[parcel_label, "coverage"] = 0

            timeseries_df[parcel_label] = label_timeseries

        del data_arr
        gc.collect()

        # Use parcel names from tsv file instead of internal CIFTI parcel names for tsvs.
        timeseries_df = timeseries_df.rename(columns=parcel_label_mapper)

        # Save out the timeseries tsv
        self._results["timeseries"] = fname_presuffix(
            "timeseries.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        timeseries_df.to_csv(self._results["timeseries"], sep="\t", na_rep="n/a", index=False)

        self._results["coverage"] = fname_presuffix(
            "coverage.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        coverage_df.to_csv(self._results["coverage"], sep="\t", na_rep="n/a", index_label="Node")

        # Prepare to create output CIFTIs
        time_axis = data_img.header.get_axis(0)
        new_header = nb.cifti2.Cifti2Header.from_axes((time_axis, parcels_axis))
        nifti_header = data_img.nifti_header.copy()
        nifti_header.set_intent(cifti_intents[".ptseries.nii"])

        # Save out the timeseries CIFTI
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

        return runtime


class _CiftiConnectInputSpec(BaseInterfaceInputSpec):
    timeseries = File(exists=True, desc="Parcellated time series file.")
    temporal_mask = File(
        exists=True,
        mandatory=False,
        desc="Temporal mask, after dummy scan removal. Only necessary if correlate is True.",
    )
    data_file = File(
        exists=True,
        mandatory=True,
        desc="Dense CIFTI time series. Used to extract a NIfTI header for the output CIFTIs.",
    )
    parcellated_atlas = File(
        exists=True,
        mandatory=True,
        desc=(
            "Atlas CIFTI that has been parcellated with itself to make a .pscalar.nii file. "
            "This is just used for its ParcelsAxis."
        ),
    )


class _CiftiConnectOutputSpec(TraitedSpec):
    correlation_ciftis = File(exists=True, desc="Correlation matrix pconn.nii file.")
    correlation_ciftis_exact = traits.Either(
        None,
        traits.List(File(exists=True)),
        desc="Correlation matrix files limited to an exact number of volumes.",
    )
    correlations = File(exists=True, desc="Correlation matrix tsv file.")
    correlations_exact = traits.Either(
        None,
        traits.List(File(exists=True)),
        desc="Correlation matrix files limited to an exact number of volumes.",
    )


class CiftiConnect(SimpleInterface):
    """Extract timeseries and compute connectivity matrices.

    Write out time series using Nilearn's NiftiLabelMasker
    Then write out functional correlation matrix of
    timeseries using numpy.
    """

    input_spec = _CiftiConnectInputSpec
    output_spec = _CiftiConnectOutputSpec

    def _run_interface(self, runtime):
        data_file = self.inputs.data_file
        pscalar_file = self.inputs.parcellated_atlas

        assert data_file.endswith((".dtseries.nii", ".dscalar.nii")), data_file
        assert pscalar_file.endswith(".pscalar.nii"), pscalar_file

        correlations_df, correlations_exact = correlate_timeseries(
            self.inputs.timeseries,
            temporal_mask=self.inputs.temporal_mask,
        )

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

        # Save out the correlation matrix CIFTI
        cifti_intents = get_cifti_intents()
        nifti_header = nb.load(data_file).nifti_header.copy()
        parcels_axis = nb.load(pscalar_file).header.get_axis(1)
        cifti_header = nb.cifti2.Cifti2Header.from_axes((parcels_axis, parcels_axis))

        nifti_header.set_intent(cifti_intents[".pconn.nii"])
        conn_img = nb.Cifti2Image(
            correlations_df.to_numpy(),
            cifti_header,
            nifti_header=nifti_header,
        )
        self._results["correlation_ciftis"] = fname_presuffix(
            "correlations.pconn.nii",
            newpath=runtime.cwd,
            use_ext=True,
        )
        conn_img.to_filename(self._results["correlation_ciftis"])
        del conn_img, correlations_df
        gc.collect()

        if not correlations_exact:
            self._results["correlations_exact"] = None
            self._results["correlation_ciftis_exact"] = None
            return runtime

        self._results["correlations_exact"] = []
        self._results["correlation_ciftis_exact"] = []
        for exact_column, exact_correlations_df in correlations_exact.items():
            exact_correlations_file = fname_presuffix(
                f"correlations_{exact_column}.tsv",
                newpath=runtime.cwd,
                use_ext=True,
            )
            exact_correlations_df.to_csv(
                exact_correlations_file,
                sep="\t",
                na_rep="n/a",
                index_label="Node",
            )
            self._results["correlations_exact"].append(exact_correlations_file)

            exact_conn_img = nb.Cifti2Image(
                exact_correlations_df.to_numpy(),
                cifti_header,
                nifti_header=nifti_header,
            )
            exact_correlations_cifti_file = fname_presuffix(
                f"correlations_{exact_column}.pconn.nii",
                newpath=runtime.cwd,
                use_ext=True,
            )
            exact_conn_img.to_filename(exact_correlations_cifti_file)
            self._results["correlation_ciftis_exact"].append(exact_correlations_cifti_file)

        return runtime


class _ConnectPlotInputSpec(BaseInterfaceInputSpec):
    atlases = InputMultiObject(
        traits.Str,
        mandatory=True,
        desc="List of atlases. Aligned with the list of time series in time_series_tsv.",
    )
    atlas_tsvs = InputMultiObject(
        traits.Str,
        mandatory=True,
        desc="The dseg.tsv associated with each atlas.",
    )
    correlations_tsv = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc=(
            "List of TSV file with correlation matrices. "
            "Aligned with the list of atlases in 'atlases'."
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

    def plot_matrix(self, corr_mat, network_labels, ax):
        """Plot matrix in subplot Axes."""
        assert corr_mat.shape[0] == len(network_labels)
        assert corr_mat.shape[1] == len(network_labels)

        # Determine order of nodes while retaining original order of networks
        unique_labels = []
        for label in network_labels:
            if label not in unique_labels:
                unique_labels.append(label)

        mapper = {label: f"{i:03d}_{label}" for i, label in enumerate(unique_labels)}
        mapped_network_labels = [mapper[label] for label in network_labels]
        community_order = np.argsort(mapped_network_labels)

        # Sort parcels by community
        corr_mat = corr_mat[community_order, :]
        corr_mat = corr_mat[:, community_order]
        np.fill_diagonal(corr_mat, 0)

        # Get the community name associated with each network
        labels = np.array(network_labels)[community_order]
        unique_labels = sorted(list(set(labels)))
        unique_labels = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)

        # Find the locations for the community-separating lines
        break_idx = [0]
        end_idx = None
        for label in unique_labels:
            start_idx = np.where(labels == label)[0][0]
            if end_idx:
                break_idx.append(np.mean([start_idx, end_idx]))

            end_idx = np.where(labels == label)[0][-1]

        break_idx.append(len(labels))
        break_idx = np.array(break_idx)

        # Find the locations for the labels in the middles of the communities
        label_idx = np.mean(np.vstack((break_idx[1:], break_idx[:-1])), axis=0)

        # Plot the correlation matrix
        ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="seismic")

        # Add lines separating networks
        for idx in break_idx[1:-1]:
            ax.axes.axvline(idx, color="black")
            ax.axes.axhline(idx, color="black")

        # Add network names
        ax.axes.set_yticks(label_idx)
        ax.axes.set_xticks(label_idx)
        ax.axes.set_yticklabels(unique_labels)
        ax.axes.set_xticklabels(unique_labels, rotation=90)
        return ax

    def _run_interface(self, runtime):
        ATLAS_LOOKUP = {
            "4S156Parcels": {
                "title": "4S 156 Parcels",
                "axes": [0, 0],
            },
            "4S456Parcels": {
                "title": "4S 456 Parcels",
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

        COMMUNITY_LOOKUP = {
            "4S156Parcels": "network_label",
            "4S456Parcels": "network_label",
            "Glasser": "community_yeo",
            "Gordon": "community",
        }

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        for atlas, subdict in ATLAS_LOOKUP.items():
            if atlas not in self.inputs.atlases:
                continue

            atlas_idx = self.inputs.atlases.index(atlas)
            atlas_file = self.inputs.correlations_tsv[atlas_idx]
            dseg_file = self.inputs.atlas_tsvs[atlas_idx]

            column_name = COMMUNITY_LOOKUP[atlas]
            dseg_df = pd.read_table(dseg_file)
            corrs_df = pd.read_table(atlas_file, index_col="Node")

            if atlas.startswith("4S"):
                atlas_mapper = {
                    "CIT168Subcortical": "Subcortical",
                    "ThalamusHCP": "Thalamus",
                    "SubcorticalHCP": "Subcortical",
                }
                network_labels = dseg_df[column_name].fillna(dseg_df["atlas_name"]).tolist()
                network_labels = [atlas_mapper.get(network, network) for network in network_labels]
            else:
                network_labels = dseg_df[column_name].fillna("None").tolist()

            ax = axes[subdict["axes"][0], subdict["axes"][1]]
            ax = self.plot_matrix(
                corr_mat=corrs_df.to_numpy(),
                network_labels=network_labels,
                ax=ax,
            )
            ax.set_title(
                subdict["title"],
                fontdict={"weight": "normal", "size": 20},
            )

        fig.tight_layout()

        # Write the results out
        self._results["connectplot"] = fname_presuffix(
            "connectivityplot",
            suffix="_matrixplot.svg",
            newpath=runtime.cwd,
            use_ext=False,
        )

        fig.savefig(self._results["connectplot"], bbox_inches="tight", pad_inches=None)
        plt.close(fig)

        return runtime
