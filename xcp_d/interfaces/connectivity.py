# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling functional connectivity."""

import gc

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
from xcp_d.utils.write_save import write_ndata

LOGGER = logging.getLogger('nipype.interface')


class _NiftiParcellateInputSpec(BaseInterfaceInputSpec):
    filtered_file = File(exists=True, mandatory=True, desc='filtered file')
    mask = File(exists=True, mandatory=True, desc='brain mask file')
    atlas = File(exists=True, mandatory=True, desc='atlas file')
    atlas_labels = File(exists=True, mandatory=True, desc='atlas labels file')
    min_coverage = traits.Float(
        0.5,
        usedefault=True,
        desc=(
            'Coverage threshold to apply to parcels. '
            'Any parcels with lower coverage than the threshold will be replaced with NaNs. '
            'Must be a value between zero and one. '
            'Default is 0.5.'
        ),
    )


class _NiftiParcellateOutputSpec(TraitedSpec):
    coverage = File(exists=True, desc='Parcel-wise coverage file.')
    timeseries = File(exists=True, desc='Parcellated time series file.')


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
        atlas_img = nb.load(self.inputs.atlas)
        min_coverage = self.inputs.min_coverage

        node_labels_df = pd.read_table(self.inputs.atlas_labels)
        # The index tells us which row/column in the matrix the parcel is associated with.
        # The 'index' column tells us what that parcel's value in the atlas image is.
        # One requirement for later is that the index values are sorted in ascending order.
        node_labels_df = node_labels_df.sort_values(by='index').reset_index(drop=True)
        node_labels = node_labels_df['label'].tolist()
        # Create a dictionary mapping df['index'] to df.index
        full_parcel_mapper = {v: k for k, v in enumerate(node_labels_df['index'].tolist())}
        masker_lut = node_labels_df.copy()
        masker_lut['name'] = masker_lut['label']
        atlas_values = np.unique(atlas_img.get_fdata())
        atlas_values = atlas_values[atlas_values != 0]
        atlas_values = atlas_values.astype(int)
        masker_lut = masker_lut.loc[masker_lut['index'].isin(atlas_values)].reset_index(drop=True)

        # Before anything, we need to measure coverage
        atlas_img_bin = nb.Nifti1Image(
            (atlas_img.get_fdata() > 0).astype(np.uint8),
            atlas_img.affine,
            atlas_img.header,
        )

        sum_masker_masked = NiftiLabelsMasker(
            labels_img=atlas_img,
            lut=masker_lut,
            background_label=0,
            mask_img=mask,
            smoothing_fwhm=None,
            standardize=False,
            strategy='sum',
            resampling_target=None,  # they should be in the same space/resolution already
        )
        sum_masker_unmasked = NiftiLabelsMasker(
            labels_img=atlas_img,
            lut=masker_lut,
            background_label=0,
            smoothing_fwhm=None,
            standardize=False,
            strategy='sum',
            resampling_target=None,  # they should be in the same space/resolution already
        )
        n_voxels_in_masked_parcels = sum_masker_masked.fit_transform(atlas_img_bin)
        n_voxels_in_parcels = sum_masker_unmasked.fit_transform(atlas_img_bin)
        parcel_coverage = np.squeeze(n_voxels_in_masked_parcels / n_voxels_in_parcels)
        coverage_thresholded = parcel_coverage < min_coverage
        del sum_masker_masked, sum_masker_unmasked, n_voxels_in_masked_parcels, n_voxels_in_parcels
        gc.collect()

        n_nodes = node_labels_df.shape[0]
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
                f'{n_nodes - n_found_nodes}/{n_nodes} of parcels not found in atlas file.'
            )

        if n_bad_nodes:
            LOGGER.warning(f'{n_bad_nodes}/{n_nodes} of parcels have 0% coverage.')

        if n_poor_parcels:
            LOGGER.warning(
                f'{n_poor_parcels}/{n_nodes} of parcels have <50% coverage. '
                "These parcels' time series will be replaced with zeros."
            )

        if n_partial_parcels:
            LOGGER.warning(
                f'{n_partial_parcels}/{n_nodes} of parcels have at least one uncovered '
                'voxel, but have enough good voxels to be usable. '
                "The bad voxels will be ignored and the parcels' time series will be "
                'calculated from the remaining voxels.'
            )

        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            lut=masker_lut,
            background_label=0,
            mask_img=mask,
            smoothing_fwhm=None,
            standardize=False,
            resampling_target=None,  # they should be in the same space/resolution already
        )

        # Use nilearn to parcellate the file
        timeseries_arr = masker.fit_transform(self.inputs.filtered_file)
        if timeseries_arr.ndim == 1:
            # Add singleton first dimension representing time.
            timeseries_arr = timeseries_arr[None, :]

        assert timeseries_arr.shape[1] == n_found_nodes
        # Map from atlas value to column index for parcels found in the atlas image
        # Keys are cols/rows in the matrix, values are atlas values
        masker_parcel_mapper = masker.region_ids_
        del masker
        gc.collect()

        # Apply the coverage mask
        timeseries_arr[:, coverage_thresholded] = np.nan

        if n_found_nodes != n_nodes:  # parcels lost by warping/downsampling atlas
            # Fill in any missing nodes in the timeseries array with NaNs.
            new_timeseries_arr = np.full(
                (timeseries_arr.shape[0], n_nodes),
                fill_value=np.nan,
                dtype=timeseries_arr.dtype,
            )
            for col_num, node_value in masker_parcel_mapper.items():
                full_col_num = full_parcel_mapper[node_value]
                new_timeseries_arr[:, full_col_num] = timeseries_arr[:, col_num]

            timeseries_arr = new_timeseries_arr
            del new_timeseries_arr
            gc.collect()

            # Fill in any missing nodes in the coverage array with zero.
            new_parcel_coverage = np.zeros(n_nodes, dtype=parcel_coverage.dtype)
            for col_num, node_value in masker_parcel_mapper.items():
                full_col_num = full_parcel_mapper[node_value]
                new_parcel_coverage[full_col_num] = parcel_coverage[col_num]

            parcel_coverage = new_parcel_coverage
            del new_parcel_coverage
            gc.collect()

        # The time series file is tab-delimited, with node names included in the first row.
        self._results['timeseries'] = fname_presuffix(
            'timeseries.tsv',
            newpath=runtime.cwd,
            use_ext=True,
        )
        timeseries_df = pd.DataFrame(data=timeseries_arr, columns=node_labels)
        timeseries_df.to_csv(self._results['timeseries'], sep='\t', na_rep='n/a', index=False)

        # Save out the coverage tsv
        coverage_df = pd.DataFrame(
            data=parcel_coverage.astype(np.float32),
            index=node_labels,
            columns=['coverage'],
        )
        self._results['coverage'] = fname_presuffix(
            'coverage.tsv',
            newpath=runtime.cwd,
            use_ext=True,
        )
        coverage_df.to_csv(self._results['coverage'], sep='\t', na_rep='n/a', index_label='Node')

        return runtime


class _TSVConnectInputSpec(BaseInterfaceInputSpec):
    timeseries = File(exists=True, desc='Parcellated time series TSV file.')
    temporal_mask = File(
        exists=True,
        mandatory=False,
        desc='Temporal mask, after dummy scan removal.',
    )


class _TSVConnectOutputSpec(TraitedSpec):
    correlations = File(exists=True, desc='Correlation matrix file.')
    correlations_exact = traits.Either(
        None,
        traits.List(File(exists=True)),
        desc='Correlation matrix files limited to an exact number of volumes.',
    )


def correlate_timeseries(timeseries, temporal_mask):
    """Correlate timeseries stored in a TSV file."""
    timeseries_df = pd.read_table(timeseries)
    correlations_exact = {}
    if isdefined(temporal_mask):
        censoring_df = pd.read_table(temporal_mask)

        # Determine if the time series is censored
        if censoring_df.shape[0] == timeseries_df.shape[0]:
            # The time series is not censored
            timeseries_df = timeseries_df.loc[censoring_df['framewise_displacement'] == 0]
            timeseries_df.reset_index(drop=True, inplace=True)

        # Now create correlation matrices limited to exact scan numbers
        censored_censoring_df = censoring_df.loc[censoring_df['framewise_displacement'] == 0]
        censored_censoring_df.reset_index(drop=True, inplace=True)
        exact_columns = [c for c in censoring_df.columns if c.startswith('exact_')]
        for exact_column in exact_columns:
            exact_timeseries_df = timeseries_df.loc[censored_censoring_df[exact_column] == 0]
            exact_correlations_df = exact_timeseries_df.corr()
            correlations_exact[exact_column] = exact_correlations_df

    # Create correlation matrix from low-motion volumes only
    correlations_df = timeseries_df.corr()

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

        self._results['correlations'] = fname_presuffix(
            'correlations.tsv',
            newpath=runtime.cwd,
            use_ext=True,
        )
        correlations_df.to_csv(
            self._results['correlations'],
            sep='\t',
            na_rep='n/a',
            index_label='Node',
        )
        del correlations_df
        gc.collect()

        if not self.inputs.temporal_mask:
            self._results['correlations_exact'] = None
            return runtime

        self._results['correlations_exact'] = []
        for exact_column, exact_correlations_df in correlations_exact.items():
            exact_correlations_file = fname_presuffix(
                f'correlations_{exact_column}.tsv',
                newpath=runtime.cwd,
                use_ext=True,
            )
            exact_correlations_df.to_csv(
                exact_correlations_file,
                sep='\t',
                na_rep='n/a',
                index_label='Node',
            )
            self._results['correlations_exact'].append(exact_correlations_file)

        return runtime


class _ConnectPlotInputSpec(BaseInterfaceInputSpec):
    atlases = InputMultiObject(
        traits.Str,
        mandatory=True,
        desc='List of atlases. Aligned with the list of time series in time_series_tsv.',
    )
    atlas_tsvs = InputMultiObject(
        traits.Str,
        mandatory=True,
        desc='The dseg.tsv associated with each atlas.',
    )
    correlations_tsv = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc=(
            'List of TSV file with correlation matrices. '
            "Aligned with the list of atlases in 'atlases'."
        ),
    )


class _ConnectPlotOutputSpec(TraitedSpec):
    connectplot = File(
        exists=True,
        mandatory=True,
        desc='Path to SVG file with four correlation heat maps.',
    )


class ConnectPlot(SimpleInterface):
    """Extract timeseries and compute connectivity matrices."""

    input_spec = _ConnectPlotInputSpec
    output_spec = _ConnectPlotOutputSpec

    def plot_matrix(self, corr_mat, network_labels, ax):
        """Plot matrix in subplot Axes."""
        if corr_mat.shape[0] != len(network_labels):
            raise ValueError(
                f'Correlation matrix shape {corr_mat.shape[0]} does not match number of network labels {len(network_labels)}'
            )
        if corr_mat.shape[1] != len(network_labels):
            raise ValueError(
                f'Correlation matrix shape {corr_mat.shape[1]} does not match number of network labels {len(network_labels)}'
            )

        # Determine order of nodes while retaining original order of networks
        unique_labels = []
        for label in network_labels:
            if label not in unique_labels:
                unique_labels.append(label)

        mapper = {label: f'{i:03d}_{label}' for i, label in enumerate(unique_labels)}
        mapped_network_labels = [mapper[label] for label in network_labels]
        community_order = np.argsort(mapped_network_labels)

        # Sort parcels by community
        corr_mat = corr_mat[community_order, :]
        corr_mat = corr_mat[:, community_order]

        # Get the community name associated with each network
        labels = np.array(network_labels)[community_order]
        unique_labels = sorted(set(labels))
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
                break_idx.append(np.nanmean([start_idx, end_idx]))

            end_idx = np.where(labels == label)[0][-1]

        break_idx.append(len(labels))
        break_idx = np.array(break_idx)

        # Find the locations for the labels in the middles of the communities
        label_idx = np.nanmean(np.vstack((break_idx[1:], break_idx[:-1])), axis=0)

        np.fill_diagonal(corr_mat, 0)

        # Plot the correlation matrix
        im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap='seismic')

        # Add lines separating networks
        for idx in break_idx[1:-1]:
            ax.axes.axvline(idx, color='black')
            ax.axes.axhline(idx, color='black')

        # Add network names
        ax.axes.set_yticks(label_idx)
        ax.axes.set_xticks(label_idx)
        ax.axes.set_yticklabels(unique_labels)
        ax.axes.set_xticklabels(unique_labels, rotation=90)

        return im, ax

    def _run_interface(self, runtime):
        from matplotlib.gridspec import GridSpec

        priority_list = [
            'MIDB',
            'MyersLabonte',
            '4S156Parcels',
            '4S456Parcels',
            'Gordon',
            'Glasser',
            'Tian',
            'HCP',
            '4S256Parcels',
            '4S356Parcels',
            '4S556Parcels',
            '4S656Parcels',
            '4S756Parcels',
            '4S856Parcels',
            '4S956Parcels',
            '4S1056Parcels',
        ]
        external_atlases = [a for a in self.inputs.atlases if a not in priority_list]
        priority_list += external_atlases
        selected_atlases = []
        c = 0
        for atlas in priority_list:
            if atlas in self.inputs.atlases:
                selected_atlases.append(atlas)
                c += 1

            if c == 4:
                break

        COMMUNITY_LOOKUP = {
            'Glasser': 'community_yeo',
        }

        if len(selected_atlases) == 4:
            nrows, ncols, figsize = 2, 2, (20, 20)
        else:
            nrows, ncols, figsize = 1, len(selected_atlases), (10 * len(selected_atlases), 10)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows, ncols + 1, width_ratios=[*[1] * ncols, 0.05])

        # Create axes for each matrix
        for i_ax, atlas in enumerate(selected_atlases):
            i_row, i_col = divmod(i_ax, ncols)
            ax = fig.add_subplot(gs[i_row, i_col])
            atlas_idx = self.inputs.atlases.index(atlas)
            atlas_file = self.inputs.correlations_tsv[atlas_idx]
            dseg_file = self.inputs.atlas_tsvs[atlas_idx]

            column_name = COMMUNITY_LOOKUP.get(atlas, 'network_label')
            dseg_df = pd.read_table(dseg_file, index_col='index')
            # Remove background label
            if 0 in dseg_df.index:
                dseg_df = dseg_df.drop(index=[0])

            corrs_df = pd.read_table(atlas_file, index_col='Node')

            if atlas.startswith('4S'):
                atlas_mapper = {
                    'CIT168Subcortical': 'Subcortical',
                    'ThalamusHCP': 'Thalamus',
                    'SubcorticalHCP': 'Subcortical',
                }
                network_labels = dseg_df[column_name].fillna(dseg_df['atlas_name']).tolist()
                network_labels = [atlas_mapper.get(network, network) for network in network_labels]
            elif column_name in dseg_df.columns:
                network_labels = dseg_df[column_name].fillna('None').tolist()
            else:
                network_labels = ['None'] * dseg_df.shape[0]

            im, ax = self.plot_matrix(
                corr_mat=corrs_df.to_numpy(),
                network_labels=network_labels,
                ax=ax,
            )
            ax.set_title(
                atlas,
                fontdict={'weight': 'normal', 'size': 20},
            )

        # Add colorbar in the reserved space
        cbar_ax = fig.add_subplot(gs[0, -1])
        plt.colorbar(im, cax=cbar_ax)
        cbar_ax.set_yticks([-1, 0, 1])
        fig.tight_layout()

        # Write the results out
        self._results['connectplot'] = fname_presuffix(
            'connectivityplot',
            suffix='_matrixplot.svg',
            newpath=runtime.cwd,
            use_ext=False,
        )

        fig.savefig(self._results['connectplot'], bbox_inches='tight', pad_inches=None)
        plt.close(fig)

        return runtime


class _CiftiToTSVInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='Parcellated CIFTI file to extract into a TSV.',
    )
    atlas_labels = File(exists=True, mandatory=True, desc='atlas labels file')


class _CiftiToTSVOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Parcellated data TSV file.')


class CiftiToTSV(SimpleInterface):
    """Extract data from a parcellated CIFTI file into a TSV file."""

    input_spec = _CiftiToTSVInputSpec
    output_spec = _CiftiToTSVOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        atlas_labels = self.inputs.atlas_labels

        if not in_file.endswith(('.ptseries.nii', '.pscalar.nii', '.pconn.nii')):
            raise ValueError(f"Unsupported CIFTI extension for 'in_file': {in_file}")

        img = nb.load(in_file)
        node_labels_df = pd.read_table(atlas_labels, index_col='index')
        node_labels_df.sort_index(inplace=True)  # ensure index is in order

        # Explicitly remove label corresponding to background (index=0), if present.
        if 0 in node_labels_df.index:
            LOGGER.warning(
                'Index value of 0 found in atlas labels file. '
                'Will assume this describes the background and ignore it.'
            )
            node_labels_df = node_labels_df.drop(index=[0])

        if 'cifti_label' in node_labels_df.columns:
            parcel_label_mapper = dict(
                zip(node_labels_df['cifti_label'], node_labels_df['label'], strict=False)
            )
        elif 'label_7network' in node_labels_df.columns:
            node_labels_df['cifti_label'] = node_labels_df['label_7network'].fillna(
                node_labels_df['label']
            )
            parcel_label_mapper = dict(
                zip(node_labels_df['cifti_label'], node_labels_df['label'], strict=False)
            )
        else:
            LOGGER.warning(
                "No 'cifti_label' column found in atlas labels file. "
                'Assuming labels in TSV exactly match node names in CIFTI atlas.'
            )
            parcel_label_mapper = dict(
                zip(node_labels_df['label'], node_labels_df['label'], strict=False)
            )

        if in_file.endswith('.pconn.nii'):
            ax0 = img.header.get_axis(0)
            ax1 = img.header.get_axis(1)
            ax0_labels = ax0.name
            ax1_labels = ax1.name
            df = pd.DataFrame(columns=ax1_labels, index=ax0_labels, data=img.get_fdata())
            check_axes = [0, 1]
        else:
            # Second axis is the parcels
            ax1 = img.header.get_axis(1)
            if not isinstance(ax1, nb.cifti2.ParcelsAxis):
                raise ValueError(f'Expected ax1 to be ParcelsAxis, but got {type(ax1)}')
            df = pd.DataFrame(columns=ax1.name, data=img.get_fdata())
            check_axes = [1]

        # Check that all node labels in the CIFTI are present in the TSV, and vice versa.
        if 0 in check_axes:
            # Replace values in index, which should match the keys in the parcel_label_mapper
            # dictionary, with the corresponding values in the dictionary.
            # If any index values are not in the dictionary, raise an error with a list of the
            # missing index values.
            # If any dictionary keys are not in the index, raise an error with a list of the
            # missing dictionary keys.
            missing_index_values = []
            missing_dict_values = []
            for index_value in df.index:
                if index_value not in parcel_label_mapper:
                    missing_index_values.append(index_value)

                for dict_value in parcel_label_mapper.keys():
                    if dict_value not in df.index:
                        missing_dict_values.append(dict_value)

                if missing_index_values:
                    raise ValueError(
                        f'Missing CIFTI labels in atlas labels DataFrame: {missing_index_values}'
                    )

                if missing_dict_values:
                    raise ValueError(f'Missing atlas labels in CIFTI file: {missing_dict_values}')

            # Replace the index values with the corresponding dictionary values.
            df.index = [parcel_label_mapper[i] for i in df.index]

        if 1 in check_axes:
            # Repeat with columns
            missing_columns = []
            missing_dict_values = []
            for column_value in df.columns:
                if column_value not in parcel_label_mapper:
                    missing_columns.append(column_value)

                for dict_value in parcel_label_mapper.keys():
                    if dict_value not in df.columns:
                        missing_dict_values.append(dict_value)

                if missing_columns:
                    raise ValueError(
                        f'Missing CIFTI labels in atlas labels DataFrame: {missing_columns}'
                    )

                if missing_dict_values:
                    raise ValueError(f'Missing atlas labels in CIFTI file: {missing_dict_values}')

            # Replace the column names with the corresponding dictionary values.
            df.columns = [parcel_label_mapper[i] for i in df.columns]

        # Save out the TSV
        self._results['out_file'] = fname_presuffix(
            'extracted.tsv',
            newpath=runtime.cwd,
            use_ext=True,
        )

        if in_file.endswith('.pconn.nii'):
            df.to_csv(self._results['out_file'], sep='\t', na_rep='n/a', index_label='Node')
        else:
            df.to_csv(self._results['out_file'], sep='\t', na_rep='n/a', index=False)

        return runtime


class _CiftiMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='CIFTI file to mask.',
    )
    mask = File(
        exists=True,
        mandatory=True,
        desc='Mask pscalar or dscalar to apply to in_file.',
    )


class _CiftiMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Masked CIFTI file.')


class CiftiMask(SimpleInterface):
    """Mask a CIFTI file by replacing masked values with NaNs.

    I (TS) created this interface because I couldn't find a way to do this with
    wb_command -cifti-math.
    """

    input_spec = _CiftiMaskInputSpec
    output_spec = _CiftiMaskOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        mask = self.inputs.mask

        supported_extensions = ('.ptseries.nii', '.pscalar.nii', '.dtseries.nii', '.dscalar.nii')
        if not in_file.endswith(supported_extensions):
            raise ValueError(f"Unsupported CIFTI extension for 'in_file': {in_file}")

        if not mask.endswith(('.pscalar.nii', '.dscalar.nii')):
            raise ValueError(f"Unsupported CIFTI extension for 'mask': {mask}")

        in_img = nb.load(in_file)
        mask_img = nb.load(mask)
        if in_img.shape[1] != mask_img.shape[1]:
            raise ValueError(
                'CIFTI files have different number of parcels/vertices. '
                f'{in_file} ({in_img.shape}) vs {mask} ({mask_img.shape})'
            )

        in_data = in_img.get_fdata()
        mask_data = mask_img.get_fdata()[0, :]
        mask_data = mask_data.astype(bool)
        in_data[:, ~mask_data] = np.nan

        # Save out the TSV
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file,
            prefix='masked_',
            newpath=runtime.cwd,
            use_ext=True,
        )
        write_ndata(in_data.T, template=in_file, filename=self._results['out_file'])

        return runtime


class _CiftiVertexMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='CIFTI file to mask.',
    )


class _CiftiVertexMaskOutputSpec(TraitedSpec):
    mask_file = File(exists=True, desc='CIFTI mask.')


class CiftiVertexMask(SimpleInterface):
    """Create a vertex-wise mask."""

    input_spec = _CiftiVertexMaskInputSpec
    output_spec = _CiftiVertexMaskOutputSpec

    def _run_interface(self, runtime):
        data_file = self.inputs.in_file

        data_img = nb.load(data_file)
        data_arr = data_img.get_fdata()

        # Flag vertices where the time series is all zeros or NaNs
        bad_vertices_idx = np.where(
            np.all(np.logical_or(data_arr == 0, np.isnan(data_arr)), axis=0)
        )[0]
        data_arr[:, bad_vertices_idx] = np.nan
        # Set any vertex with a NaN to 0 and all others to 1 in the mask file
        vertex_weights_arr = np.all(~np.isnan(data_arr), axis=0).astype(int)

        # Save out the TSV
        self._results['mask_file'] = fname_presuffix(
            self.inputs.in_file,
            suffix='.dscalar.nii',
            newpath=runtime.cwd,
            use_ext=False,
        )
        write_ndata(vertex_weights_arr, template=data_file, filename=self._results['mask_file'])

        return runtime
