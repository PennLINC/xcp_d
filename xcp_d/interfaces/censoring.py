"""Interfaces for the post-processing workflows."""
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

from xcp_d.utils.confounds import _infer_dummy_scans, load_confound_matrix, load_motion
from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.modified_data import _drop_dummy_scans, compute_fd

LOGGER = logging.getLogger("nipype.interface")


class _RemoveDummyVolumesInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True, mandatory=True, desc="Either cifti or nifti ")
    dummy_scans = traits.Either(
        traits.Int,
        "auto",
        mandatory=True,
        desc=(
            "Number of volumes to drop from the beginning, "
            "calculated in an earlier workflow from dummy_scans."
        ),
    )
    confounds_file = traits.Either(
        File(exists=True),
        None,
        mandatory=True,
        desc=(
            "TSV file with selected confounds for denoising. "
            "May be None if denoising is disabled."
        ),
    )
    fmriprep_confounds_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv. Used for motion-based censoring.",
    )
    motion_file = File(
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
    confounds_file_dropped_TR = traits.Either(
        File(exists=True),
        None,
        desc="TSV file with selected confounds for denoising, after removing TRs.",
    )
    fmriprep_confounds_file_dropped_TR = File(
        exists=True,
        desc="fMRIPrep confounds tsv after removing TRs. Used for motion-based censoring.",
    )
    bold_file_dropped_TR = File(
        exists=True,
        desc="bold or cifti with volumes dropped",
    )
    dummy_scans = traits.Int(desc="Number of volumes dropped.")
    motion_file_dropped_TR = File(
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
            confounds_file=self.inputs.fmriprep_confounds_file,
        )

        self._results["dummy_scans"] = dummy_scans

        # Check if we need to do anything
        if dummy_scans == 0:
            # write the output out
            self._results["bold_file_dropped_TR"] = self.inputs.bold_file
            self._results[
                "fmriprep_confounds_file_dropped_TR"
            ] = self.inputs.fmriprep_confounds_file
            self._results["confounds_file_dropped_TR"] = self.inputs.confounds_file
            self._results["motion_file_dropped_TR"] = self.inputs.motion_file
            self._results["temporal_mask_dropped_TR"] = self.inputs.temporal_mask
            return runtime

        # get the file names to output to
        self._results["bold_file_dropped_TR"] = fname_presuffix(
            self.inputs.bold_file,
            newpath=runtime.cwd,
            suffix="_dropped",
            use_ext=True,
        )
        self._results["fmriprep_confounds_file_dropped_TR"] = fname_presuffix(
            self.inputs.fmriprep_confounds_file,
            newpath=runtime.cwd,
            suffix="_fmriprep_dropped",
            use_ext=True,
        )
        self._results["motion_file_dropped_TR"] = fname_presuffix(
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
        fmriprep_confounds_df = pd.read_table(self.inputs.fmriprep_confounds_file)
        fmriprep_confounds_df_dropped = fmriprep_confounds_df.drop(np.arange(dummy_scans))
        fmriprep_confounds_df_dropped.to_csv(
            self._results["fmriprep_confounds_file_dropped_TR"],
            sep="\t",
            index=False,
        )

        # Drop the first N rows from the confounds file
        self._results["confounds_file_dropped_TR"] = None
        if self.inputs.confounds_file:
            self._results["confounds_file_dropped_TR"] = fname_presuffix(
                self.inputs.bold_file,
                suffix="_selected_confounds_dropped.tsv",
                newpath=os.getcwd(),
                use_ext=False,
            )
            confounds_df = pd.read_table(self.inputs.confounds_file)
            confounds_df_dropped = confounds_df.drop(np.arange(dummy_scans))
            confounds_df_dropped.to_csv(
                self._results["confounds_file_dropped_TR"],
                sep="\t",
                index=False,
            )

        # Drop the first N rows from the motion file
        motion_df = pd.read_table(self.inputs.motion_file)
        motion_df_dropped = motion_df.drop(np.arange(dummy_scans))
        motion_df_dropped.to_csv(
            self._results["motion_file_dropped_TR"],
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
        motion_outliers = censoring_df["framewise_displacement"].to_numpy()

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
        low_motion_idx = censoring_df.loc[censoring_df["framewise_displacement"] != 1].index.values
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


class _GenerateConfoundsInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="BOLD file after denoising, interpolation, and filtering",
    )
    params = traits.Str(mandatory=True, desc="Parameter set for regression.")
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
    fd_thresh = traits.Float(
        mandatory=False,
        default_value=0.3,
        desc="Framewise displacement threshold. All values above this will be dropped.",
    )
    head_radius = traits.Float(mandatory=False, default_value=50, desc="Head radius in mm ")
    fmriprep_confounds_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv.",
    )
    fmriprep_confounds_json = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds json.",
    )
    custom_confounds_file = traits.Either(
        None,
        File(exists=True),
        mandatory=True,
        desc="Custom confounds tsv.",
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


class _GenerateConfoundsOutputSpec(TraitedSpec):
    filtered_confounds_file = File(
        exists=True,
        desc=(
            "The original fMRIPrep confounds, with the motion parameters and their Volterra "
            "expansion regressors replaced with filtered versions."
        ),
    )
    confounds_file = traits.Either(
        File(exists=True),
        None,
        desc=(
            "The selected confounds. This may include custom confounds as well. "
            "It will also always have the linear trend and a constant column."
        ),
    )
    confounds_metadata = traits.Dict(desc="Metadata associated with the confounds_file output.")
    motion_file = File(
        exists=True,
        desc="The filtered motion parameters.",
    )
    motion_metadata = traits.Dict(desc="Metadata associated with the filtered_motion output.")
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


class GenerateConfounds(SimpleInterface):
    """Load, consolidate, and filter confounds.

    Also, generate the temporal mask.
    """

    input_spec = _GenerateConfoundsInputSpec
    output_spec = _GenerateConfoundsOutputSpec

    def _run_interface(self, runtime):
        fmriprep_confounds_df = pd.read_table(self.inputs.fmriprep_confounds_file)
        motion_df = load_motion(
            fmriprep_confounds_df.copy(),
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            band_stop_min=self.inputs.band_stop_min,
            band_stop_max=self.inputs.band_stop_max,
        )

        # Add in framewise displacement
        fd_timeseries = compute_fd(
            confound=motion_df,
            head_radius=self.inputs.head_radius,
        )
        motion_df["framewise_displacement"] = fd_timeseries

        # A file to house the modified fMRIPrep confounds.
        self._results["filtered_confounds_file"] = fname_presuffix(
            self.inputs.fmriprep_confounds_file,
            suffix="_filtered",
            newpath=runtime.cwd,
            use_ext=True,
        )

        # Replace original motion parameters with filtered versions
        for col in motion_df.columns.tolist():
            # Check that the column already exists, in case fMRIPrep changes column names.
            # We don't want to use the original motion parameters accidentally.
            if col not in fmriprep_confounds_df.columns:
                raise ValueError(
                    f"Column '{col}' not found in confounds "
                    f"{self.inputs.fmriprep_confounds_file}. "
                    "Please open an issue on GitHub."
                )

            fmriprep_confounds_df[col] = motion_df[col]

        fmriprep_confounds_df.to_csv(
            self._results["filtered_confounds_file"],
            sep="\t",
            index=False,
        )

        # Load nuisance regressors, but use filtered motion parameters.
        confounds_df, confounds_metadata = load_confound_matrix(
            params=self.inputs.params,
            img_file=self.inputs.in_file,
            confounds_file=self._results["filtered_confounds_file"],
            confounds_json_file=self.inputs.fmriprep_confounds_json,
            custom_confounds=self.inputs.custom_confounds_file,
        )

        # Orthogonalize full nuisance regressors w.r.t. any signal regressors
        signal_columns = [c for c in confounds_df.columns if c.startswith("signal__")]
        if signal_columns:
            LOGGER.warning(
                "Signal columns detected. "
                "Orthogonalizing nuisance columns w.r.t. the following signal columns: "
                f"{', '.join(signal_columns)}"
            )
            noise_columns = [c for c in confounds_df.columns if not c.startswith("signal__")]

            # Don't orthogonalize the intercept or linear trend regressors
            untouched_cols = ["linear_trend", "intercept"]
            columns_to_denoise = [c for c in noise_columns if c not in untouched_cols]
            orth_confounds_df = confounds_df[noise_columns].copy()
            orth_columns = [f"{c}_orth" for c in columns_to_denoise]
            orth_confounds_df = pd.DataFrame(
                index=confounds_df.index,
                columns=orth_columns + untouched_cols,
            )
            orth_confounds_df.loc[:, untouched_cols] = confounds_df[untouched_cols]

            # Do the orthogonalization
            signal_regressors = confounds_df[signal_columns].to_numpy()
            noise_regressors = confounds_df[columns_to_denoise].to_numpy()
            signal_betas = np.linalg.lstsq(signal_regressors, noise_regressors, rcond=None)[0]
            pred_noise_regressors = np.dot(signal_regressors, signal_betas)
            orth_noise_regressors = noise_regressors - pred_noise_regressors

            # Replace the old data
            orth_confounds_df.loc[:, orth_columns] = orth_noise_regressors
            confounds_df = orth_confounds_df

            for col in columns_to_denoise:
                desc_str = (
                    "This regressor is orthogonalized with respect to the 'signal' regressors "
                    f"({', '.join(signal_columns)}) prior to any censoring."
                )

                col_metadata = {}
                if col in confounds_metadata.keys():
                    col_metadata = confounds_metadata.pop(col)
                    if "Description" in col_metadata.keys():
                        desc_str = f"{col_metadata['Description']} {desc_str}"

                col_metadata["Description"] = desc_str
                confounds_metadata[f"{col}_orth"] = col_metadata

        self._results["confounds_metadata"] = confounds_metadata

        # get the output
        self._results["motion_file"] = fname_presuffix(
            self.inputs.fmriprep_confounds_file,
            suffix="_motion",
            newpath=runtime.cwd,
            use_ext=True,
        )
        motion_df.to_csv(self._results["motion_file"], sep="\t", index=False)

        self._results["confounds_file"] = None
        if confounds_df is not None:
            self._results["confounds_file"] = fname_presuffix(
                self.inputs.fmriprep_confounds_file,
                suffix="_confounds",
                newpath=runtime.cwd,
                use_ext=True,
            )
            confounds_df.to_csv(self._results["confounds_file"], sep="\t", index=False)

        # Generate temporal mask with all timepoints have FD over threshold set to 1.
        outlier_mask = np.zeros(len(fd_timeseries), dtype=int)
        if self.inputs.fd_thresh > 0:
            outlier_mask[fd_timeseries > self.inputs.fd_thresh] = 1
        else:
            LOGGER.info(f"FD threshold set to {self.inputs.fd_thresh}. Censoring is disabled.")

        self._results["temporal_mask"] = fname_presuffix(
            "desc-fd_outliers.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )

        outliers_df = pd.DataFrame(data=outlier_mask, columns=["framewise_displacement"])
        outliers_df.to_csv(
            self._results["temporal_mask"],
            index=False,
            header=True,
            sep="\t",
        )

        # Compile metadata to pass along to outputs.
        motion_metadata = {
            "framewise_displacement": {
                "Description": (
                    "Framewise displacement calculated according to Power et al. (2012)."
                ),
                "Units": "mm",
                "HeadRadius": self.inputs.head_radius,
            }
        }
        if self.inputs.motion_filter_type == "lp":
            motion_metadata["LowpassFilter"] = self.inputs.band_stop_max
            motion_metadata["LowpassFilterOrder"] = self.inputs.motion_filter_order
        elif self.inputs.motion_filter_type == "notch":
            motion_metadata["BandstopFilter"] = [
                self.inputs.band_stop_min,
                self.inputs.band_stop_max,
            ]
            motion_metadata["BandstopFilterOrder"] = self.inputs.motion_filter_order

        self._results["motion_metadata"] = motion_metadata

        outliers_metadata = {
            "framewise_displacement": {
                "Description": "Outlier time series based on framewise displacement.",
                "Levels": {
                    "0": "Non-outlier volume",
                    "1": "Outlier volume",
                },
                "Threshold": self.inputs.fd_thresh,
            }
        }
        self._results["temporal_mask_metadata"] = outliers_metadata

        return runtime
