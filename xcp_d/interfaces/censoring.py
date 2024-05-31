"""Interfaces for the post-processing workflows."""

import json
import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from xcp_d.utils.confounds import (
    _infer_dummy_scans,
    _modify_motion_filter,
    calculate_outliers,
    filter_motion,
    load_confound_matrix,
)
from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.modified_data import _drop_dummy_scans, compute_fd
from xcp_d.utils.write_save import write_ndata

LOGGER = logging.getLogger("nipype.interface")


class _RemoveDummyVolumesInputSpec(BaseInterfaceInputSpec):
    dummy_scans = traits.Either(
        traits.Int,
        "auto",
        mandatory=True,
        desc=(
            "Number of volumes to drop from the beginning, "
            "calculated in an earlier workflow from dummy_scans."
        ),
    )
    bold_file = File(exists=True, mandatory=True, desc="Either cifti or nifti ")
    design_matrix = traits.Either(
        File(exists=True),
        None,
        mandatory=True,
        desc=(
            "TSV file with selected confounds for denoising. "
            "May be None if denoising is disabled."
        ),
    )
    full_confounds = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv. Used for motion-based censoring.",
    )
    modified_full_confounds = File(
        exists=True,
        mandatory=True,
        desc="Confounds file containing only filtered motion parameters.",
    )
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc="Temporal mask file.",
    )


class _RemoveDummyVolumesOutputSpec(TraitedSpec):
    design_matrix_dropped_TR = traits.Either(
        File(exists=True),
        None,
        desc="TSV file with selected confounds for denoising, after removing TRs.",
    )
    full_confounds_dropped_TR = File(
        exists=True,
        desc="fMRIPrep confounds tsv after removing TRs. Used for motion-based censoring.",
    )
    bold_file_dropped_TR = File(
        exists=True,
        desc="bold or cifti with volumes dropped",
    )
    dummy_scans = traits.Int(desc="Number of volumes dropped.")
    modified_full_confounds_dropped_TR = File(
        exists=True,
        desc="Confounds file containing only filtered motion parameters.",
    )
    temporal_mask_dropped_TR = File(
        exists=True,
        desc="Temporal mask file.",
    )


class RemoveDummyVolumes(SimpleInterface):
    """Removes initial volumes from a nifti or cifti file.

    A bold file and its corresponding confounds TSV (fmriprep format)
    are adjusted to remove the first n seconds of data.
    """

    input_spec = _RemoveDummyVolumesInputSpec
    output_spec = _RemoveDummyVolumesOutputSpec

    def _run_interface(self, runtime):
        dummy_scans = _infer_dummy_scans(
            dummy_scans=self.inputs.dummy_scans,
            confounds_file=self.inputs.full_confounds,
        )

        self._results["dummy_scans"] = dummy_scans

        # Check if we need to do anything
        if dummy_scans == 0:
            # write the output out
            self._results["bold_file_dropped_TR"] = self.inputs.bold_file
            self._results["full_confounds_dropped_TR"] = self.inputs.full_confounds
            self._results["modified_full_confounds_dropped_TR"] = (
                self.inputs.modified_full_confounds
            )
            self._results["temporal_mask_dropped_TR"] = self.inputs.temporal_mask
            self._results["design_matrix_dropped_TR"] = self.inputs.design_matrix
            return runtime

        # get the file names to output to
        self._results["bold_file_dropped_TR"] = fname_presuffix(
            self.inputs.bold_file,
            newpath=runtime.cwd,
            suffix="_dropped",
            use_ext=True,
        )
        self._results["full_confounds_dropped_TR"] = fname_presuffix(
            self.inputs.full_confounds,
            newpath=runtime.cwd,
            suffix="_dropped",
            use_ext=True,
        )
        self._results["modified_full_confounds_dropped_TR"] = fname_presuffix(
            self.inputs.bold_file,
            suffix="_motion_dropped.tsv",
            newpath=os.getcwd(),
            use_ext=False,
        )
        self._results["temporal_mask_dropped_TR"] = fname_presuffix(
            self.inputs.bold_file,
            suffix="_tmask_dropped.tsv",
            newpath=os.getcwd(),
            use_ext=False,
        )

        # Remove the dummy volumes
        dropped_image = _drop_dummy_scans(self.inputs.bold_file, dummy_scans=dummy_scans)
        dropped_image.to_filename(self._results["bold_file_dropped_TR"])

        # Drop the first N rows from the pandas dataframe
        full_confounds_df = pd.read_table(self.inputs.full_confounds)
        full_confounds_df_dropped = full_confounds_df.drop(np.arange(dummy_scans))
        full_confounds_df_dropped.to_csv(
            self._results["full_confounds_dropped_TR"],
            sep="\t",
            index=False,
        )

        # Drop the first N rows from the confounds file
        self._results["design_matrix_dropped_TR"] = None
        if self.inputs.design_matrix:
            self._results["design_matrix_dropped_TR"] = fname_presuffix(
                self.inputs.bold_file,
                suffix="_selected_confounds_dropped.tsv",
                newpath=os.getcwd(),
                use_ext=False,
            )
            confounds_df = pd.read_table(self.inputs.design_matrix)
            confounds_df_dropped = confounds_df.drop(np.arange(dummy_scans))
            confounds_df_dropped.to_csv(
                self._results["design_matrix_dropped_TR"],
                sep="\t",
                index=False,
            )

        # Drop the first N rows from the motion file
        modified_full_confounds_df = pd.read_table(self.inputs.modified_full_confounds)
        modified_full_confounds_df_dropped = modified_full_confounds_df.drop(
            np.arange(dummy_scans)
        )
        modified_full_confounds_df_dropped.to_csv(
            self._results["modified_full_confounds_dropped_TR"],
            sep="\t",
            index=False,
        )

        # Drop the first N rows from the temporal mask
        censoring_df = pd.read_table(self.inputs.temporal_mask)
        censoring_df_dropped = censoring_df.drop(np.arange(dummy_scans))
        censoring_df_dropped.to_csv(
            self._results["temporal_mask_dropped_TR"],
            sep="\t",
            index=False,
        )

        return runtime


class _CensorInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="BOLD file after denoising, interpolation, and filtering",
    )
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc=(
            "Temporal mask; all motion outlier volumes set to 1. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )


class _CensorOutputSpec(TraitedSpec):
    censored_denoised_bold = File(
        exists=True,
        desc="Censored BOLD file",
    )


class Censor(SimpleInterface):
    """Apply temporal mask to data."""

    input_spec = _CensorInputSpec
    output_spec = _CensorOutputSpec

    def _run_interface(self, runtime):
        # Read in temporal mask
        censoring_df = pd.read_table(self.inputs.temporal_mask)
        motion_outliers = censoring_df["interpolation"].to_numpy()

        if np.sum(motion_outliers) == 0:  # No censoring needed
            self._results["censored_denoised_bold"] = self.inputs.in_file
            return runtime

        # Read in other files
        bold_img_interp = nb.load(self.inputs.in_file)
        bold_data_interp = bold_img_interp.get_fdata()

        is_nifti = bold_img_interp.ndim > 2
        if is_nifti:
            bold_data_censored = bold_data_interp[:, :, :, motion_outliers == 0]

            bold_img_censored = nb.Nifti1Image(
                bold_data_censored,
                affine=bold_img_interp.affine,
                header=bold_img_interp.header,
            )
        else:
            bold_data_censored = bold_data_interp[motion_outliers == 0, :]

            time_axis, brain_model_axis = [
                bold_img_interp.header.get_axis(i) for i in range(bold_img_interp.ndim)
            ]
            new_total_volumes = bold_data_censored.shape[0]
            censored_time_axis = time_axis[:new_total_volumes]
            # Note: not an error. A time axis cannot be accessed with irregularly
            # spaced values. Since we use the temporal_mask for marking the volumes removed,
            # the time axis also is not used further in XCP-D.
            censored_header = nb.cifti2.Cifti2Header.from_axes(
                (censored_time_axis, brain_model_axis)
            )
            bold_img_censored = nb.Cifti2Image(
                bold_data_censored,
                header=censored_header,
                nifti_header=bold_img_interp.nifti_header,
            )

        # get the output
        self._results["censored_denoised_bold"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_censored",
            newpath=runtime.cwd,
            use_ext=True,
        )

        bold_img_censored.to_filename(self._results["censored_denoised_bold"])
        return runtime


class _RandomCensorInputSpec(BaseInterfaceInputSpec):
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc=(
            "Temporal mask; all motion outlier volumes set to 1. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )
    temporal_mask_metadata = traits.Dict(
        desc="Metadata associated with the temporal_mask output.",
    )
    exact_scans = traits.List(
        traits.Int,
        mandatory=True,
        desc="Numbers of scans to retain. If None, no additional censoring will be performed.",
    )
    random_seed = traits.Either(
        None,
        traits.Int,
        usedefault=True,
        mandatory=False,
        desc="Random seed.",
    )


class _RandomCensorOutputSpec(TraitedSpec):
    temporal_mask = File(
        exists=True,
        desc="Temporal mask file.",
    )
    temporal_mask_metadata = traits.Dict(
        desc="Metadata associated with the temporal_mask output.",
    )


class RandomCensor(SimpleInterface):
    """Randomly flag volumes to censor."""

    input_spec = _RandomCensorInputSpec
    output_spec = _RandomCensorOutputSpec

    def _run_interface(self, runtime):
        # Read in temporal mask
        censoring_df = pd.read_table(self.inputs.temporal_mask)
        temporal_mask_metadata = self.inputs.temporal_mask_metadata.copy()

        if not self.inputs.exact_scans:
            self._results["temporal_mask"] = self.inputs.temporal_mask
            self._results["temporal_mask_metadata"] = temporal_mask_metadata
            return runtime

        self._results["temporal_mask"] = fname_presuffix(
            self.inputs.temporal_mask,
            suffix="_random",
            newpath=runtime.cwd,
            use_ext=True,
        )
        rng = np.random.default_rng(self.inputs.random_seed)
        low_motion_idx = censoring_df.loc[censoring_df["interpolation"] != 1].index.values
        for exact_scan in self.inputs.exact_scans:
            random_censor = rng.choice(low_motion_idx, size=exact_scan, replace=False)
            column_name = f"exact_{exact_scan}"
            censoring_df[column_name] = 0
            censoring_df.loc[low_motion_idx, column_name] = 1
            censoring_df.loc[random_censor, column_name] = 0
            temporal_mask_metadata[column_name] = {
                "Description": (
                    f"Randomly selected low-motion volumes to retain exactly {exact_scan} "
                    "volumes."
                ),
                "Levels": {
                    "0": "Retained or high-motion volume",
                    "1": "Randomly censored volume",
                },
            }

        censoring_df.to_csv(self._results["temporal_mask"], sep="\t", index=False)
        self._results["temporal_mask_metadata"] = temporal_mask_metadata

        return runtime


class _ModifyConfoundsInputSpec(BaseInterfaceInputSpec):
    head_radius = traits.Float(mandatory=False, default_value=50, desc="Head radius in mm ")
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
    full_confounds = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv.",
    )
    full_confounds_json = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds json.",
    )
    motion_filter_type = traits.Either(
        None,
        traits.Str,
        mandatory=True,
    )
    motion_filter_order = traits.Either(
        None,
        traits.Int,
        mandatory=True,
    )
    band_stop_min = traits.Either(
        None,
        traits.Float,
        mandatory=True,
        desc="Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )
    band_stop_max = traits.Either(
        None,
        traits.Float,
        mandatory=True,
        desc="Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )


class _ModifyConfoundsOutputSpec(TraitedSpec):
    modified_full_confounds = File(
        exists=True,
        desc="Modified confounds file.",
    )
    modified_full_confounds_metadata = traits.Dict(
        exists=True,
        desc="Metadata associated with the modified_full_confounds output.",
    )


class ModifyConfounds(SimpleInterface):
    """Apply motion filter to confounds and recalculate framewise displacement."""

    input_spec = _ModifyConfoundsInputSpec
    output_spec = _ModifyConfoundsOutputSpec

    def _run_interface(self, runtime):
        full_confounds_df = pd.read_table(self.inputs.full_confounds)
        with open(self.inputs.full_confounds_json, "r") as f:
            confounds_metadata = json.load(f)

        # Modify motion filter if requested parameters are below the Nyquist frequency
        band_stop_min_adjusted, band_stop_max_adjusted, _ = _modify_motion_filter(
            motion_filter_type=self.inputs.motion_filter_type,
            band_stop_min=self.inputs.band_stop_min,
            band_stop_max=self.inputs.band_stop_max,
            TR=self.inputs.TR,
        )

        # Filter motion parameters
        full_confounds_df = filter_motion(
            confounds_df=full_confounds_df,
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            band_stop_min=band_stop_min_adjusted,
            band_stop_max=band_stop_max_adjusted,
        )

        # Calculate and add in framewise displacement
        full_confounds_df["framewise_displacement"] = compute_fd(
            confound=full_confounds_df,
            head_radius=self.inputs.head_radius,
        )

        # Write out the modified fMRIPrep confounds.
        self._results["modified_full_confounds"] = fname_presuffix(
            self.inputs.full_confounds,
            suffix="_filtered",
            newpath=runtime.cwd,
            use_ext=True,
        )
        full_confounds_df.to_csv(
            self._results["modified_full_confounds"],
            sep="\t",
            index=False,
        )

        # Add filtering info to affected columns' metadata.
        # This includes any columns that start with any of the following strings.
        motion_columns = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "framewise_displacement",
        ]
        for col in full_confounds_df.columns.tolist():
            if not any(col.startswith(mc) for mc in motion_columns):
                continue

            col_metadata = confounds_metadata.get(col, {})
            if col == "framewise_displacement":
                col_metadata["Description"] = (
                    "Framewise displacement calculated according to Power et al. (2012)."
                )
                col_metadata["Units"] = "mm"
                col_metadata["HeadRadius"] = self.inputs.head_radius

            if self.inputs.motion_filter_type == "lp":
                filters = col_metadata.get("SoftwareFilters", {})
                filters["Butterworth low-pass filter"] = {
                    "cutoff": band_stop_min_adjusted / 60,
                    "order": self.inputs.motion_filter_order,
                    "cutoff units": "Hz",
                    "function": "scipy.signal.filtfilt",
                }
                col_metadata["SoftwareFilters"] = filters

            elif self.inputs.motion_filter_type == "notch":
                filters = col_metadata.get("SoftwareFilters", {})
                filters["IIR notch digital filter"] = {
                    "cutoff": [
                        band_stop_max_adjusted / 60,
                        band_stop_min_adjusted / 60,
                    ],
                    "order": self.inputs.motion_filter_order,
                    "cutoff units": "Hz",
                    "function": "scipy.signal.filtfilt",
                }
                col_metadata["SoftwareFilters"] = filters

            confounds_metadata[col] = col_metadata

        self._results["modified_full_confounds_metadata"] = confounds_metadata

        return runtime


class _GenerateTemporalMaskInputSpec(BaseInterfaceInputSpec):
    full_confounds = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv.",
    )
    fd_thresh = traits.List(
        traits.Float,
        minlen=2,
        maxlen=2,
        desc="Framewise displacement threshold. All values above this will be dropped.",
    )
    dvars_thresh = traits.List(
        traits.Float,
        minlen=2,
        maxlen=2,
        desc="DVARS threshold. All values above this will be dropped.",
    )
    censor_before = traits.List(
        traits.Int,
        minlen=2,
        maxlen=2,
        desc="Number of volumes to censor before each FD or DVARS outlier volume.",
    )
    censor_after = traits.List(
        traits.Int,
        minlen=2,
        maxlen=2,
        desc="Number of volumes to censor after each FD or DVARS outlier volume.",
    )
    censor_between = traits.List(
        traits.Int,
        minlen=2,
        maxlen=2,
        desc="Number of volumes to censor between each FD or DVARS outlier volume.",
    )


class _GenerateTemporalMaskOutputSpec(TraitedSpec):
    temporal_mask = File(
        exists=True,
        desc=(
            "Temporal mask; all values above fd_thresh set to 1. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )
    temporal_mask_metadata = traits.Dict(
        desc="Metadata associated with the temporal_mask output.",
    )


class GenerateTemporalMask(SimpleInterface):
    """Generate the temporal mask."""

    input_spec = _GenerateTemporalMaskInputSpec
    output_spec = _GenerateTemporalMaskOutputSpec

    def _run_interface(self, runtime):
        full_confounds_df = pd.read_table(self.inputs.full_confounds)

        outliers_df = calculate_outliers(
            confounds=full_confounds_df,
            fd_thresh=self.inputs.fd_thresh,
            dvars_thresh=self.inputs.dvars_thresh,
            before=self.inputs.censor_before,
            after=self.inputs.censor_after,
            between=self.inputs.censor_between,
        )

        self._results["temporal_mask"] = fname_presuffix(
            "outliers.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        outliers_df = outliers_df.astype(int)
        outliers_df.to_csv(
            self._results["temporal_mask"],
            index=False,
            header=True,
            sep="\t",
        )

        outliers_metadata = {
            "framewise_displacement": {
                "Description": "Outlier time series based on framewise displacement.",
                "Levels": {
                    "0": "Non-outlier volume",
                    "1": "Outlier volume",
                },
                "Threshold": self.inputs.fd_thresh[0],
            },
            "dvars": {
                "Description": "Outlier time series based on DVARS.",
                "Levels": {
                    "0": "Non-outlier volume",
                    "1": "Outlier volume",
                },
                "Threshold": self.inputs.dvars_thresh[0],
            },
            "denoising": {
                "Description": (
                    "Outlier time series based on framewise displacement and DVARS. "
                    "Any volumes marked as outliers by either metric are marked as outliers in "
                    "this column. "
                    "This initial set of outliers is then expanded to mark "
                    f"{self.inputs.censor_before[0]} volumes before and "
                    f"{self.inputs.censor_after[0]} volumes after each outlier volume. "
                    "Finally, any sequences of non-outliers shorter than or equal to "
                    f"{self.inputs.censor_between[0]} volumes were marked as outliers."
                ),
                "Levels": {
                    "0": "Non-outlier volume",
                    "1": "Outlier volume",
                },
            },
            "framewise_displacement_interpolation": {
                "Description": "Outlier time series based on framewise displacement.",
                "Levels": {
                    "0": "Non-outlier volume",
                    "1": "Outlier volume",
                },
                "Threshold": self.inputs.fd_thresh[1],
            },
            "dvars_interpolation": {
                "Description": "Outlier time series based on DVARS.",
                "Levels": {
                    "0": "Non-outlier volume",
                    "1": "Outlier volume",
                },
                "Threshold": self.inputs.dvars_thresh[1],
            },
            "interpolation": {
                "Description": (
                    "Outlier time series based on framewise displacement and DVARS. "
                    "Any volumes marked as outliers by either metric, or the 'denoising' column', "
                    "are marked as outliers in this column. "
                    "This initial set of outliers is then expanded to mark "
                    f"{self.inputs.censor_before[1]} volumes before and "
                    f"{self.inputs.censor_after[1]} volumes after each outlier volume. "
                    "Finally, any sequences of non-outliers shorter than or equal to "
                    f"{self.inputs.censor_between[1]} volumes were marked as outliers."
                ),
                "Levels": {
                    "0": "Non-outlier volume",
                    "1": "Outlier volume",
                },
            },
        }
        self._results["temporal_mask_metadata"] = outliers_metadata

        return runtime


class _GenerateDesignMatrixInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="BOLD file after denoising, interpolation, and filtering",
    )
    params = traits.Str(mandatory=True, desc="Parameter set for regression.")
    full_confounds = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv.",
    )
    full_confounds_metadata = traits.Dict(
        mandatory=True,
        desc="fMRIPrep confounds metadata.",
    )
    custom_confounds_file = traits.Either(
        None,
        File(exists=True),
        mandatory=True,
        desc="Custom confounds tsv.",
    )


class _GenerateDesignMatrixOutputSpec(TraitedSpec):
    design_matrix = traits.Either(
        File(exists=True),
        None,
        desc="The selected confounds for denoising. This may include custom confounds as well.",
    )
    design_matrix_metadata = traits.Dict(desc="Metadata associated with the design_matrix output.")


class GenerateDesignMatrix(SimpleInterface):
    """Load, consolidate, and potentially orthogonalize confounds."""

    input_spec = _GenerateDesignMatrixInputSpec
    output_spec = _GenerateDesignMatrixOutputSpec

    def _run_interface(self, runtime):
        full_confounds_json = fname_presuffix(
            self.inputs.full_confounds,
            suffix=".json",
            newpath=runtime.cwd,
            use_ext=False,
        )
        with open(full_confounds_json, "w") as fo:
            json.dump(self.inputs.full_confounds_metadata, fo, sort_keys=True, indent=4)

        # Load nuisance regressors, but use filtered motion parameters.
        design_matrix_df, design_matrix_metadata = load_confound_matrix(
            params=self.inputs.params,
            img_file=self.inputs.in_file,
            confounds_file=self.inputs.full_confounds,
            confounds_json_file=full_confounds_json,
            custom_confounds=self.inputs.custom_confounds_file,
        )

        if design_matrix_df is not None:
            # Orthogonalize full nuisance regressors w.r.t. any signal regressors
            signal_columns = [c for c in design_matrix_df.columns if c.startswith("signal__")]
            if signal_columns:
                LOGGER.warning(
                    "Signal columns detected. "
                    "Orthogonalizing nuisance columns w.r.t. the following signal columns: "
                    f"{', '.join(signal_columns)}"
                )
                noise_columns = [
                    c for c in design_matrix_df.columns if not c.startswith("signal__")
                ]

                orth_cols = [f"{c}_orth" for c in noise_columns]
                orth_design_matrix_df = pd.DataFrame(
                    index=design_matrix_df.index,
                    columns=orth_cols,
                )

                # Do the orthogonalization
                signal_regressors = design_matrix_df[signal_columns].to_numpy()
                noise_regressors = design_matrix_df[noise_columns].to_numpy()
                signal_betas = np.linalg.lstsq(signal_regressors, noise_regressors, rcond=None)[0]
                pred_noise_regressors = np.dot(signal_regressors, signal_betas)
                orth_noise_regressors = noise_regressors - pred_noise_regressors

                # Replace the old data
                orth_design_matrix_df.loc[:, orth_cols] = orth_noise_regressors
                design_matrix_df = orth_design_matrix_df

                for col in noise_columns:
                    desc_str = (
                        "This regressor is orthogonalized with respect to the 'signal' regressors "
                        f"({', '.join(signal_columns)}) after dummy scan removal, "
                        "but prior to any censoring."
                    )

                    col_metadata = {}
                    if col in design_matrix_metadata.keys():
                        col_metadata = design_matrix_metadata.pop(col)
                        if "Description" in col_metadata.keys():
                            desc_str = f"{col_metadata['Description']} {desc_str}"

                    col_metadata["Description"] = desc_str
                    design_matrix_metadata[f"{col}_orth"] = col_metadata

        self._results["design_matrix_metadata"] = design_matrix_metadata

        # get the output
        self._results["design_matrix"] = None
        if design_matrix_df is not None:
            self._results["design_matrix"] = fname_presuffix(
                self.inputs.full_confounds,
                suffix="_confounds",
                newpath=runtime.cwd,
                use_ext=True,
            )
            design_matrix_df.to_csv(self._results["design_matrix"], sep="\t", index=False)


class _ReduceCiftiInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="CIFTI timeseries file to reduce.",
    )
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc=(
            "Temporal mask; all motion outlier volumes set to 1. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )
    column = traits.Str(
        mandatory=True,
        desc="Column name in the temporal mask to use for censoring.",
    )


class _ReduceCiftiOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Censored CIFTI file.",
    )


class ReduceCifti(SimpleInterface):
    """Remove flagged volumes from a CIFTI series file."""

    input_spec = _ReduceCiftiInputSpec
    output_spec = _ReduceCiftiOutputSpec

    def _run_interface(self, runtime):
        # Read in temporal mask
        censoring_df = pd.read_table(self.inputs.temporal_mask)
        img = nb.load(self.inputs.in_file)

        if self.inputs.column not in censoring_df.columns:
            raise ValueError(
                f"Column '{self.inputs.column}' not found in temporal mask file "
                f"({self.inputs.temporal_mask})."
            )

        # Drop the high-motion volumes, because the CIFTI is already censored
        censored_censoring_df = censoring_df.loc[censoring_df["framewise_displacement"] == 0]
        censored_censoring_df.reset_index(drop=True, inplace=True)
        if censored_censoring_df.shape[0] != img.shape[0]:
            raise ValueError(
                f"Number of volumnes in the temporal mask ({censored_censoring_df.shape[0]}) "
                f"does not match the CIFTI ({img.shape[0]})."
            )

        data = img.get_fdata()
        retain_idx = (censored_censoring_df[self.inputs.column] == 0).index.values
        data = data[retain_idx, ...]

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file,
            prefix=f"{self.inputs.column}_",
            newpath=runtime.cwd,
            use_ext=True,
        )
        write_ndata(data.T, template=self.inputs.in_file, filename=self._results["out_file"])

        return runtime
