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

from xcp_d.utils.confounds import _infer_dummy_scans, load_motion
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
    confounds_file = File(
        exists=True,
        mandatory=True,
        desc="TSV file with selected confounds for denoising.",
    )
    fmriprep_confounds_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv. Used for motion-based censoring.",
    )


class _RemoveDummyVolumesOutputSpec(TraitedSpec):
    confounds_file_dropped_TR = File(
        exists=True,
        mandatory=True,
        desc="TSV file with selected confounds for denoising, after removing TRs.",
    )

    fmriprep_confounds_file_dropped_TR = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv after removing TRs. Used for motion-based censoring.",
    )

    bold_file_dropped_TR = File(
        exists=True,
        mandatory=True,
        desc="bold or cifti with volumes dropped",
    )
    dummy_scans = traits.Int(desc="Number of volumes dropped.")


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
            suffix="_fmriprepDropped",
            use_ext=True,
        )
        self._results["confounds_file_dropped_TR"] = fname_presuffix(
            self.inputs.bold_file,
            suffix="_selected_confounds_dropped.tsv",
            newpath=os.getcwd(),
            use_ext=False,
        )

        # Remove the dummy volumes
        dropped_image = _drop_dummy_scans(self.inputs.bold_file, dummy_scans=dummy_scans)
        dropped_image.to_filename(self._results["bold_file_dropped_TR"])

        # Drop the first N rows from the pandas dataframe
        fmriprep_confounds_df = pd.read_table(self.inputs.fmriprep_confounds_file)
        dropped_fmriprep_confounds_df = fmriprep_confounds_df.drop(np.arange(dummy_scans))

        # Drop the first N rows from the confounds file
        confounds_df = pd.read_table(self.inputs.confounds_file)
        confounds_tsv_dropped = confounds_df.drop(np.arange(dummy_scans))

        # Save out results
        dropped_fmriprep_confounds_df.to_csv(
            self._results["fmriprep_confounds_file_dropped_TR"],
            sep="\t",
            index=False,
        )
        confounds_tsv_dropped.to_csv(
            self._results["confounds_file_dropped_TR"],
            sep="\t",
            index=False,
        )

        return runtime


class _FlagMotionOutliersInputSpec(BaseInterfaceInputSpec):
    fd_thresh = traits.Float(
        mandatory=False,
        default_value=0.2,
        desc="Framewise displacement threshold. All values above this will be dropped.",
    )
    fmriprep_confounds_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv. Used for flagging high-motion volumes.",
    )
    head_radius = traits.Float(mandatory=False, default_value=50, desc="Head radius in mm ")
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
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
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


class _FlagMotionOutliersOutputSpec(TraitedSpec):
    filtered_motion = File(
        exists=True,
        mandatory=True,
        desc=(
            "Framewise displacement timeseries. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )
    filtered_motion_metadata = traits.Dict(
        desc="Metadata associated with the filtered_motion output.",
    )
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc=(
            "Temporal mask; all values above fd_thresh set to 1. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )
    temporal_mask_metadata = traits.Dict(
        desc="Metadata associated with the temporal_mask output.",
    )


class FlagMotionOutliers(SimpleInterface):
    """Generate a temporal mask based on recalculated FD.

    Takes in confound files and information about filtering-
    including band stop values and motion filter type.
    Then proceeds to create a motion-filtered confounds matrix and recalculates FD from
    filtered motion parameters.
    Finally generates temporal mask with volumes above FD threshold set to 1.
    Outputs temporal mask and framewise displacement timeseries.
    """

    input_spec = _FlagMotionOutliersInputSpec
    output_spec = _FlagMotionOutliersOutputSpec

    def _run_interface(self, runtime):
        # Read in fmriprep confounds tsv to calculate FD
        fmriprep_confounds_df = pd.read_table(self.inputs.fmriprep_confounds_file)
        motion_df = load_motion(
            fmriprep_confounds_df.copy(),
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            band_stop_min=self.inputs.band_stop_min,
            band_stop_max=self.inputs.band_stop_max,
        )

        fd_timeseries = compute_fd(
            confound=motion_df,
            head_radius=self.inputs.head_radius,
        )
        motion_df["framewise_displacement"] = fd_timeseries

        # Generate temporal mask with all timepoints have FD over threshold
        # set to 1 and then dropped.
        outlier_mask = np.zeros(len(fd_timeseries), dtype=int)
        if self.inputs.fd_thresh > 0:
            outlier_mask[fd_timeseries > self.inputs.fd_thresh] = 1
        else:
            LOGGER.info(f"FD threshold set to {self.inputs.fd_thresh}. Censoring is disabled.")

        # get the output
        self._results["temporal_mask"] = fname_presuffix(
            "desc-fd_outliers.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        self._results["filtered_motion"] = fname_presuffix(
            "desc-filtered_motion.tsv",
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

        self._results["filtered_motion_metadata"] = motion_metadata

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

        motion_df.to_csv(
            self._results["filtered_motion"],
            index=False,
            header=True,
            sep="\t",
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
        mandatory=True,
        desc="Censored bold file",
    )
    censored_confounds = File(
        exists=True,
        mandatory=True,
        desc="confounds_file censored",
    )
    censored_motion = File(
        exists=True,
        mandatory=True,
        desc=(
            "Framewise displacement timeseries. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )


class Censor(SimpleInterface):
    """Apply temporal mask to data."""

    input_spec = _CensorInputSpec
    output_spec = _CensorOutputSpec

    def _run_interface(self, runtime):
        # Read in temporal mask
        temporal_mask = pd.read_table(self.inputs.temporal_mask)
        temporal_mask = temporal_mask["framewise_displacement"].to_numpy()

        if np.sum(temporal_mask) == 0:  # No censoring needed
            self._results["censored_denoised_bold"] = self.inputs.in_file
            return runtime

        # Read in other files
        bold_img_interp = nb.load(self.inputs.in_file)
        bold_data_interp = bold_img_interp.get_fdata()

        if bold_img_interp.ndim > 2:  # If Nifti
            bold_data_censored = bold_data_interp[:, :, :, temporal_mask == 0]
        else:
            bold_data_censored = bold_data_interp[temporal_mask == 0, :]

        # Turn censored bold into image
        if nb.load(self.inputs.in_file).ndim > 2:
            # If it's a Nifti image
            bold_img_censored = nb.Nifti1Image(
                bold_data_censored,
                affine=bold_img_interp.affine,
                header=bold_img_interp.header,
            )
        else:
            # If it's a Cifti image
            time_axis, brain_model_axis = [
                bold_img_interp.header.get_axis(i) for i in range(bold_img_interp.ndim)
            ]
            new_total_volumes = bold_data_censored.shape[0]
            censored_time_axis = time_axis[:new_total_volumes]
            # Note: not an error. A time axis cannot be accessed with irregularly
            # spaced values. Since we use the temporal_mask for marking the volumes removed,
            # the time axis also is not used further in XCP.
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
