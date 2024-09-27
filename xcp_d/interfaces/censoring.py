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
    isdefined,
    traits,
)

from xcp_d.utils.confounds import _infer_dummy_scans, _modify_motion_filter, load_motion
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
    confounds_tsv = traits.Either(
        File(exists=True),
        None,
        mandatory=False,
        desc=(
            "TSV file with selected confounds for denoising. "
            "May be None if denoising is disabled."
        ),
    )
    confounds_images = traits.Either(
        traits.List(File(exists=True)),
        None,
        mandatory=False,
        desc="List of images with confounds. May be None if denoising is disabled.",
    )
    motion_file = File(
        exists=True,
        mandatory=True,
        desc="TSV file with motion regressors. Used for motion-based censoring.",
    )
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc="Temporal mask file.",
    )


class _RemoveDummyVolumesOutputSpec(TraitedSpec):
    bold_file_dropped_TR = File(
        exists=True,
        desc="bold or cifti with volumes dropped",
    )
    dummy_scans = traits.Int(desc="Number of volumes dropped.")
    confounds_tsv_dropped_TR = traits.Either(
        File(exists=True),
        None,
        desc=(
            "TSV file with selected confounds for denoising. "
            "May be None if denoising is disabled."
        ),
    )
    confounds_images_dropped_TR = traits.Either(
        traits.List(File(exists=True)),
        None,
        desc="List of images with confounds. May be None if denoising is disabled.",
    )
    motion_file_dropped_TR = File(
        exists=True,
        desc="TSV file with motion parameters.",
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
            confounds_file=self.inputs.motion_file,
        )

        self._results["dummy_scans"] = dummy_scans

        # Check if we need to do anything
        if dummy_scans == 0:
            # write the output out
            self._results["bold_file_dropped_TR"] = self.inputs.bold_file
            self._results["confounds_tsv_dropped_TR"] = self.inputs.confounds_tsv
            self._results["confounds_images_dropped_TR"] = self.inputs.confounds_images
            self._results["motion_file_dropped_TR"] = self.inputs.motion_file
            self._results["temporal_mask_dropped_TR"] = self.inputs.temporal_mask

            if (
                isdefined(self.inputs.confounds_images)
                and self.inputs.confounds_images is not None
            ):
                self._results["confounds_images_dropped_TR"] = self.inputs.confounds_images

            if isdefined(self.inputs.confounds_tsv) and self.inputs.confounds_tsv is not None:
                self._results["confounds_tsv_dropped_TR"] = self.inputs.confounds_tsv

            return runtime

        # get the file names to output to
        self._results["bold_file_dropped_TR"] = fname_presuffix(
            self.inputs.bold_file,
            newpath=runtime.cwd,
            suffix="_dropped",
            use_ext=True,
        )
        self._results["motion_file_dropped_TR"] = fname_presuffix(
            self.inputs.motion_file,
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
        if isdefined(self.inputs.confounds_tsv) and self.inputs.confounds_tsv is not None:
            self._results["confounds_tsv_dropped_TR"] = fname_presuffix(
                self.inputs.bold_file,
                suffix="_confounds_dropped.tsv",
                newpath=os.getcwd(),
                use_ext=False,
            )
            confounds_df = pd.read_table(self.inputs.confounds_tsv)
            confounds_df_dropped = confounds_df.drop(np.arange(dummy_scans))
            confounds_df_dropped.to_csv(
                self._results["confounds_tsv_dropped_TR"],
                sep="\t",
                index=False,
            )

        if isdefined(self.inputs.confounds_images) and self.inputs.confounds_images is not None:
            self._results["confounds_images_dropped_TR"] = []
            for i_file, confound_file in enumerate(self.inputs.confounds_images):
                confound_file_dropped = fname_presuffix(
                    confound_file,
                    suffix=f"_conf{i_file}_dropped",
                    newpath=os.getcwd(),
                    use_ext=True,
                )
                dropped_confounds_image = _drop_dummy_scans(confound_file, dummy_scans=dummy_scans)
                dropped_confounds_image.to_filename(confound_file_dropped)
                self._results["confounds_images_dropped_TR"].append(confound_file_dropped)

        # Remove the dummy volumes
        dropped_image = _drop_dummy_scans(self.inputs.bold_file, dummy_scans=dummy_scans)
        dropped_image.to_filename(self._results["bold_file_dropped_TR"])

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
    column = traits.Str(
        default="framewise_displacement",
        usedefault=True,
        mandatory=False,
        desc="Column name in the temporal mask to use for censoring.",
    )


class _CensorOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Censored BOLD file",
    )


class Censor(SimpleInterface):
    """Apply temporal mask to data.

    If a column other than "framewise_displacement" is used, then this assumes that the outliers
    flagged by "framewise_displacement" have already been removed from the BOLD file, and it
    reduces the data further.
    """

    input_spec = _CensorInputSpec
    output_spec = _CensorOutputSpec

    def _run_interface(self, runtime):
        # Read in temporal mask
        censoring_df = pd.read_table(self.inputs.temporal_mask)
        img = nb.load(self.inputs.in_file)

        if self.inputs.column not in censoring_df.columns:
            raise ValueError(
                f"Column '{self.inputs.column}' not found in temporal mask file "
                f"({self.inputs.temporal_mask})."
            )

        # Drop the high-motion volumes, because the image is already censored
        if self.inputs.column != "framewise_displacement":
            censoring_df = censoring_df.loc[censoring_df["framewise_displacement"] == 0]
            censoring_df.reset_index(drop=True, inplace=True)

        motion_outliers = (censoring_df[self.inputs.column] != 0).index.values

        if motion_outliers.size == 0:  # No censoring needed
            self._results["censored_denoised_bold"] = self.inputs.in_file
            return runtime

        # Read in other files
        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()

        is_nifti = img.ndim > 2
        if is_nifti:
            if censoring_df.shape[0] != img.shape[3]:
                raise ValueError(
                    f"Number of volumes in the temporal mask ({censoring_df.shape[0]}) "
                    f"does not match the NIfTI ({img.shape[3]})."
                )

            data_censored = data[:, :, :, motion_outliers]

            img_censored = nb.Nifti1Image(
                data_censored,
                affine=img.affine,
                header=img.header,
            )
        else:
            if censoring_df.shape[0] != img.shape[0]:
                raise ValueError(
                    f"Number of volumes in the temporal mask ({censoring_df.shape[0]}) "
                    f"does not match the CIFTI ({img.shape[0]})."
                )

            data_censored = data[motion_outliers, :]

            time_axis, brain_model_axis = [img.header.get_axis(i) for i in range(img.ndim)]
            new_total_volumes = data_censored.shape[0]
            censored_time_axis = time_axis[:new_total_volumes]
            # Note: not an error. A time axis cannot be accessed with irregularly spaced values.
            # Since we use the temporal_mask for marking the volumes removed,
            # the time axis also is not used further in XCP-D.
            censored_header = nb.cifti2.Cifti2Header.from_axes(
                (censored_time_axis, brain_model_axis)
            )
            img_censored = nb.Cifti2Image(
                data_censored,
                header=censored_header,
                nifti_header=img.nifti_header,
            )

        # get the output
        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_censored",
            newpath=runtime.cwd,
            use_ext=True,
        )

        img_censored.to_filename(self._results["out_file"])
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


class _ProcessMotionInputSpec(BaseInterfaceInputSpec):
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
    fd_thresh = traits.Float(
        mandatory=False,
        default_value=0.3,
        desc="Framewise displacement threshold. All values above this will be dropped.",
    )
    head_radius = traits.Float(mandatory=False, default_value=50, desc="Head radius in mm ")
    motion_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv.",
    )
    motion_json = File(
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


class _ProcessMotionOutputSpec(TraitedSpec):
    motion_file = File(
        exists=True,
        desc="The filtered motion parameters.",
    )
    motion_metadata = traits.Dict(desc="Metadata associated with the motion_file output.")
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


class ProcessMotion(SimpleInterface):
    """Load and filter motion regressors in order to generate a temporal mask."""

    input_spec = _ProcessMotionInputSpec
    output_spec = _ProcessMotionOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.motion_json, "r") as f:
            motion_metadata = json.load(f)

        band_stop_min_adjusted, band_stop_max_adjusted, _ = _modify_motion_filter(
            motion_filter_type=self.inputs.motion_filter_type,
            band_stop_min=self.inputs.band_stop_min,
            band_stop_max=self.inputs.band_stop_max,
            TR=self.inputs.TR,
        )

        motion_df = load_motion(
            self.inputs.motion_file,
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            band_stop_min=band_stop_min_adjusted,
            band_stop_max=band_stop_max_adjusted,
        )

        # Add in framewise displacement
        motion_df["framewise_displacement"] = compute_fd(
            confound=motion_df,
            head_radius=self.inputs.head_radius,
            filtered=False,
        )
        fd_timeseries = motion_df["framewise_displacement"].to_numpy()
        if self.inputs.motion_filter_type:
            motion_df["framewise_displacement_filtered"] = compute_fd(
                confound=motion_df,
                head_radius=self.inputs.head_radius,
                filtered=True,
            )
            fd_timeseries = motion_df["framewise_displacement_filtered"].to_numpy()

        # Compile motion metadata from confounds metadata, adding in filtering info
        # First drop any columns that are not motion parameters
        orig_motion_df = pd.read_table(self.inputs.motion_file)
        orig_motion_cols = orig_motion_df.columns.tolist()
        cols_to_drop = sorted(list(set(orig_motion_cols) - set(motion_df.columns.tolist())))
        motion_metadata = {k: v for k, v in motion_metadata.items() if k not in cols_to_drop}
        for col in motion_df.columns.tolist():
            col_metadata = motion_metadata.get(col, {})
            if col.startswith("framewise_displacement"):
                col_metadata["Description"] = (
                    "Framewise displacement calculated according to Power et al. (2012)."
                )
                col_metadata["Units"] = "mm"
                col_metadata["HeadRadius"] = self.inputs.head_radius

            if self.inputs.motion_filter_type == "lp" and col.endswith("_filtered"):
                filters = col_metadata.get("SoftwareFilters", {})
                filters["Butterworth low-pass filter"] = {
                    "cutoff": band_stop_min_adjusted / 60,
                    "order": self.inputs.motion_filter_order,
                    "cutoff units": "Hz",
                    "function": "scipy.signal.filtfilt",
                }
                col_metadata["SoftwareFilters"] = filters

            elif self.inputs.motion_filter_type == "notch" and col.endswith("_filtered"):
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

            motion_metadata[col] = col_metadata

        self._results["motion_metadata"] = motion_metadata

        # Store the filtered motion parameters
        self._results["motion_file"] = fname_presuffix(
            self.inputs.motion_file,
            suffix="_motion",
            newpath=runtime.cwd,
            use_ext=True,
        )
        motion_df.to_csv(self._results["motion_file"], sep="\t", index=False)

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


class _GenerateConfoundsInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="Preprocessed BOLD file",
    )
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
    confounds_files = traits.Dict(
        mandatory=True,
        desc=(
            "Dictionary of confound names and paths to corresponding files. "
            "Keys are confound names, values are dictionaries with keys 'file' and 'metadata'."
        ),
    )
    confounds_config = traits.Dict(
        mandatory=True,
        desc="Configuration file for confounds.",
    )
    dataset_links = traits.Dict(
        mandatory=True,
        desc="Dataset links for the XCP-D run.",
    )
    out_dir = traits.Str(
        mandatory=True,
        desc=(
            "Output directory for the XCP-D run. "
            "Not used to write out any files- just used for dataset links."
        ),
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
    confounds_tsv = File(
        exists=True,
        desc="The aggregated confounds in a tabular file.",
    )
    confounds_images = traits.List(
        File(exists=True),
        desc="The aggregated confounds in image files.",
    )
    confounds_metadata = traits.Dict(desc="Metadata associated with the confounds output.")


class GenerateConfounds(SimpleInterface):
    """Load confounds according to a configuration file.

    This function basically checks that each confounds file has the right number of volumes,
    selects the requisite columns from each input tabular file, and puts those columns into a
    single tabular file.

    It also applies the motion filter to motion-based regressors, if any are detected.
    NOTE: This interface identifies motion-based regressors based on their names.
    If the names of the motion-based regressors are not as expected,
    the motion filter will not be applied.

    Parameters
    ----------
    confounds_files : dict
        Dictionary of confound names and paths to corresponding files.
        Keys are confound names, values are dictionaries with keys "file" and "metadata".
    confounds_config : dict
        Configuration file for confounds.
    n_volumes : int
        Number of volumes in the fMRI data.

    Returns
    -------
    confounds_tsv : str or None
        Path to the TSV file containing combined tabular confounds.
        None if no tabular confounds are present.
    confounds_images : list of str
        List of paths to the voxelwise confounds images.
    """

    input_spec = _GenerateConfoundsInputSpec
    output_spec = _GenerateConfoundsOutputSpec

    def _run_interface(self, runtime):
        import re

        import nibabel as nb
        import pandas as pd

        from xcp_d.utils.bids import make_bids_uri
        from xcp_d.utils.confounds import filter_motion, volterra

        in_img = nb.load(self.inputs.in_file)
        if in_img.ndim == 2:  # CIFTI
            n_volumes = in_img.shape[0]
        else:  # NIfTI
            n_volumes = in_img.shape[3]

        new_confound_df = pd.DataFrame(index=np.arange(n_volumes))

        confounds_images = []
        confounds_metadata = {}
        confound_files = []
        for confound_name, confound_info in self.inputs.confounds_files.items():
            confound_file = confound_info["file"]
            confound_files.append(confound_file)
            confound_metadata = confound_info["metadata"]
            confound_params = self.inputs.confounds_config["confounds"][confound_name]
            if "columns" in confound_params:  # Tabular confounds
                confound_df = pd.read_table(confound_file)
                if confound_df.shape[0] != n_volumes:
                    raise ValueError(
                        f"Number of volumes in confounds file ({confound_df.shape[0]}) "
                        f"does not match number of volumes in the fMRI data ({n_volumes})."
                    )

                available_columns = confound_df.columns.tolist()
                required_columns = confound_params["columns"]
                for column in required_columns:
                    if column.startswith("^"):
                        # Regular expression
                        found_columns = [
                            col_name
                            for col_name in available_columns
                            if re.match(column, col_name, re.IGNORECASE)
                        ]
                        if not found_columns:
                            raise ValueError(
                                f"No columns found matching regular expression '{column}'"
                            )

                        for found_column in found_columns:
                            if found_column in new_confound_df:
                                raise ValueError(
                                    f"Duplicate column name ({found_column}) in confounds "
                                    "configuration."
                                )

                            new_confound_df[found_column] = confound_df[found_column]

                            # Replace NaNs in new column with zeros
                            new_confound_df.fillna({found_column: 0}, inplace=True)

                            confounds_metadata[found_column] = confounds_metadata.get(
                                found_column, {}
                            )
                            confounds_metadata[found_column]["Sources"] = make_bids_uri(
                                in_files=[confound_file],
                                dataset_links=self.inputs.dataset_links,
                                out_dir=self.inputs.out_dir,
                            )
                    else:
                        if column not in confound_df.columns:
                            raise ValueError(f"Column '{column}' not found in confounds file.")

                        if column in new_confound_df:
                            raise ValueError(
                                f"Duplicate column name ({column}) in confounds configuration."
                            )

                        new_confound_df[column] = confound_df[column]

                        # Replace NaNs in new column with zeros
                        new_confound_df.fillna({column: 0}, inplace=True)

                        confounds_metadata[column] = confounds_metadata.get(column, {})
                        confounds_metadata[column]["Sources"] = make_bids_uri(
                            in_files=[confound_file],
                            dataset_links=self.inputs.dataset_links,
                            out_dir=self.inputs.out_dir,
                        )

                # Collect column metadata
                for column in new_confound_df.columns:
                    if column in confound_metadata:
                        confounds_metadata[column] = confound_metadata[column]
                    else:
                        confounds_metadata[column] = {}

            else:  # Voxelwise confounds
                confound_img = nb.load(confound_file)
                if confound_img.ndim == 2:  # CIFTI
                    n_volumes_check = confound_img.shape[0]
                else:  # NIfTI
                    n_volumes_check = confound_img.shape[3]

                if n_volumes_check != n_volumes:
                    raise ValueError(
                        f"Number of volumes in confounds image ({n_volumes_check}) "
                        f"does not match number of volumes in the fMRI data ({n_volumes})."
                    )

                confounds_images.append(confound_file)

                # Collect image metadata
                new_confound_df.loc[:, confound_name] = np.nan  # fill with NaNs as a placeholder
                confounds_metadata[confound_name] = confound_metadata
                confounds_metadata[confound_name]["Sources"] = make_bids_uri(
                    in_files=[confound_file],
                    dataset_links=self.inputs.dataset_links,
                    out_dir=self.inputs.out_dir,
                )
                confounds_metadata[confound_name]["Description"] = (
                    "A placeholder column representing a voxel-wise confound. "
                    "The actual confound data are stored in an imaging file."
                )

        # This actually gets overwritten in init_postproc_derivatives_wf.
        confounds_metadata["Sources"] = make_bids_uri(
            in_files=confound_files,
            dataset_links=self.inputs.dataset_links,
            out_dir=self.inputs.out_dir,
        )

        if self.inputs.motion_filter_type:
            # Filter the motion parameters
            # 1. Pop out the 6 basic motion parameters
            # 2. Filter them
            # 3. Calculate the Volterra expansion of the filtered parameters
            # 4. For each selected motion confound, remove that column and replace with the
            #    filtered version. Include `_filtered` in the new column name.
            motion_params = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
            motion_based_params = [
                c for c in new_confound_df.columns if any(c.startswith(p) for p in motion_params)
            ]
            # Motion-based regressors detected
            if len(motion_based_params):
                # Check the motion filter parameters
                band_stop_min_adjusted, band_stop_max_adjusted, _ = _modify_motion_filter(
                    motion_filter_type=self.inputs.motion_filter_type,
                    band_stop_min=self.inputs.band_stop_min,
                    band_stop_max=self.inputs.band_stop_max,
                    TR=self.inputs.TR,
                )

                # Filter the base motion parameters and calculate the Volterra expansion
                base_motion_columns = [c for c in new_confound_df.columns if c in motion_params]
                motion_df = new_confound_df[base_motion_columns]
                motion_df.loc[:, :] = filter_motion(
                    data=motion_df.to_numpy(),
                    TR=self.inputs.TR,
                    motion_filter_type=self.inputs.motion_filter_type,
                    band_stop_min=band_stop_min_adjusted,
                    band_stop_max=band_stop_max_adjusted,
                    motion_filter_order=self.inputs.motion_filter_order,
                )
                motion_df = volterra(motion_df)
                motion_df.fillna(0, inplace=True)

                # Patch in the filtered motion parameters to the confounds DataFrame
                overlapping_columns = [
                    c for c in new_confound_df.columns if c in motion_df.columns
                ]
                motion_unfiltered = [
                    c for c in motion_based_params if c not in overlapping_columns
                ]
                if motion_unfiltered:
                    raise ValueError(
                        f"Motion-based regressors {motion_unfiltered} were not filtered."
                    )

                # Select the relevant filtered motion parameter columns
                motion_df = motion_df[overlapping_columns]
                motion_df.columns = [f"{c}_filtered" for c in motion_df.columns]

                # Replace the original motion columns with the filtered versions
                new_confound_df.drop(columns=overlapping_columns, inplace=True)
                new_confound_df = pd.concat([new_confound_df, motion_df], axis=1)

                # Replace the original motion metadata with the filtered versions
                for column in overlapping_columns:
                    col_metadata = confounds_metadata[column]

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

                    confounds_metadata[f"{column}_filtered"] = col_metadata
                    confounds_metadata.pop(column, None)

        self._results["confounds_tsv"] = fname_presuffix(
            "desc-confounds_timeseries.tsv",
            newpath=runtime.cwd,
            use_ext=True,
        )
        new_confound_df.to_csv(self._results["confounds_tsv"], sep="\t", index=False)
        self._results["confounds_images"] = confounds_images
        self._results["confounds_metadata"] = confounds_metadata
        return runtime
