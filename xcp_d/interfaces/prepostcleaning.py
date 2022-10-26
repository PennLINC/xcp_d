"""Interfaces for the post-processing workflows."""
import os

import nibabel as nb
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from xcp_d.utils.confounds import load_motion
from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.modified_data import compute_fd, generate_mask, interpolate_masked_data
from xcp_d.utils.write_save import read_ndata, write_ndata


class _RemoveTRInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True,
                     mandatory=True,
                     desc="Either cifti or nifti ")
    initial_volumes_to_drop = traits.Int(mandatory=True,
                                         desc="Number of volumes to drop from the beginning,"
                                              "calculated in an earlier workflow from dummytime "
                                              "and repetition time.")
    fmriprep_confounds_file = File(exists=True,
                                   mandatory=False,
                                   desc="fmriprep confounds tsv")
    custom_confounds = traits.Either(traits.Undefined,
                                     File,
                                     desc="Name of custom confounds file, or True",
                                     exists=False,
                                     mandatory=False)


class _RemoveTROutputSpec(TraitedSpec):
    fmriprep_confounds_file_dropped_TR = File(exists=True,
                                              mandatory=True,
                                              desc="fmriprep confounds tsv after removing TRs,")

    bold_file_dropped_TR = File(exists=True,
                                mandatory=True,
                                desc="bold or cifti with volumes dropped")

    custom_confounds_dropped = File(exists=False,
                                    mandatory=False,
                                    desc="custom_confounds_tsv dropped")


class RemoveTR(SimpleInterface):
    """Removes initial volumes from a nifti or cifti file.

    A bold file and its corresponding confounds TSV (fmriprep format)
    are adjusted to remove the first n seconds of data.

    If 0, the bold file and confounds are returned as-is. If dummytime
    is larger than the repetition time, the corresponding rows are removed
    from the confounds TSV and the initial volumes are removed from the
    nifti or cifti file.

    If the dummy time is less than the repetition time, it will
    be rounded up. (i.e. dummytime=3, TR=2 will remove the first 2 volumes).

    The number of volumes to be removed has been calculated in a previous
    workflow.
    """

    input_spec = _RemoveTRInputSpec
    output_spec = _RemoveTROutputSpec

    def _run_interface(self, runtime):
        volumes_to_drop = self.inputs.initial_volumes_to_drop
        # Check if we need to do anything
        if self.inputs.initial_volumes_to_drop == 0:
            # write the output out
            self._results['bold_file_dropped_TR'] = self.inputs.bold_file
            self._results['fmriprep_confounds_file_dropped'
                          '_TR'] = self.inputs.fmriprep_confounds_file
            return runtime

        # get the file names to output to
        dropped_bold_file = fname_presuffix(
            self.inputs.bold_file,
            newpath=runtime.cwd,
            suffix="_dropped",
            use_ext=True)
        dropped_confounds_file = fname_presuffix(
            self.inputs.fmriprep_confounds_file,
            newpath=runtime.cwd,
            suffix="_dropped",
            use_ext=True)

        # read the bold file
        bold_image = nb.load(self.inputs.bold_file)
        data = bold_image.get_fdata()

        # If it's a Cifti Image:
        if bold_image.ndim == 2:
            dropped_data = data[volumes_to_drop:, ...]   # time series is the first element
            time_axis, brain_model_axis = [
                bold_image.header.get_axis(i) for i in range(bold_image.ndim)]
            new_total_volumes = dropped_data.shape[0]
            dropped_time_axis = time_axis[:new_total_volumes]
            dropped_header = nb.cifti2.Cifti2Header.from_axes(
                (dropped_time_axis, brain_model_axis))
            dropped_image = nb.Cifti2Image(
                dropped_data,
                header=dropped_header,
                nifti_header=bold_image.nifti_header)

        # If it's a Nifti Image:
        else:
            dropped_data = data[..., volumes_to_drop:]
            dropped_image = nb.Nifti1Image(
                dropped_data,
                affine=bold_image.affine,
                header=bold_image.header)

        # Write the file
        dropped_image.to_filename(dropped_bold_file)

        # Drop the first N rows from the pandas dataframe
        confounds_df = pd.read_csv(self.inputs.fmriprep_confounds_file, sep="\t")
        dropped_confounds_df = confounds_df.drop(np.arange(volumes_to_drop))

        # Drop the first N rows from the custom confounds file, if provided:
        if self.inputs.custom_confounds:
            if os.path.exists(self.inputs.custom_confounds):
                custom_confounds_tsv_undropped = pd.read_table(
                    self.inputs.custom_confounds, header=None)
                custom_confounds_tsv_dropped = custom_confounds_tsv_undropped.drop(
                    np.arange(volumes_to_drop))
            else:
                print("No custom confounds were found or had their volumes dropped")
        else:
            print("No custom confounds were found or had their volumes dropped")

        # Save out results
        dropped_confounds_df.to_csv(dropped_confounds_file, sep="\t", index=False)
        # Write to output node
        self._results['bold_file_dropped_TR'] = dropped_bold_file
        self._results['fmriprep_confounds_file_dropped_TR'] = dropped_confounds_file

        if self.inputs.custom_confounds:
            if os.path.exists(self.inputs.custom_confounds):
                self._results['custom_confounds_dropped'] = fname_presuffix(
                    self.inputs.bold_file,
                    suffix='_custom_confounds_dropped.tsv',
                    newpath=os.getcwd(),
                    use_ext=False)

                custom_confounds_tsv_dropped.to_csv(self._results['custom_confounds_dropped'],
                                                    index=False,
                                                    header=False,
                                                    sep="\t")  # Assuming input is tab separated!

        return runtime


class _CensorScrubInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc=" Partially processed bold or nifti")
    fd_thresh = traits.Float(
        exists=True,
        mandatory=False,
        default_value=0.2,
        desc="Framewise displacement threshold. All values above this will be dropped.",
    )
    initial_volumes_to_drop = traits.Either(
        traits.Int,
        "auto",
        exists=False,
        mandatory=True,
        desc="Number of volumes to remove from the beginning of the BOLD run.",
    )
    fmriprep_confounds_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv.",
    )
    head_radius = traits.Float(
        exists=True,
        mandatory=False,
        default_value=50,
        desc="Head radius in mm.",
    )
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
    motion_filter_type = traits.Either(
        None,
        traits.Str,
        exists=False,
        mandatory=True,
    )
    motion_filter_order = traits.Int(exists=False, mandatory=True)
    band_stop_min = traits.Either(
        None,
        traits.Float,
        exists=True,
        mandatory=True,
        desc="Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )
    band_stop_max = traits.Either(
        None,
        traits.Float,
        exists=True,
        mandatory=True,
        desc="Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )


class _CensorScrubOutputSpec(TraitedSpec):
    tmask = File(exists=True, mandatory=True,
                 desc="Temporal mask; all values above fd_thresh set to 1")
    filtered_motion = File(
        exists=True,
        mandatory=True,
        desc=(
            "Framewise displacement timeseries. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )


class CensorScrub(SimpleInterface):
    """Generate a temporal mask based on recalculated FD.

    Takes in confound files, bold file to be censored, and information about filtering-
    including band stop values and motion filter type.
    Then proceeds to create a motion-filtered confounds matrix and recalculates FD from
    filtered motion parameters.
    Finally generates temporal mask with volumes above FD threshold set to 1,
    then dropped from both confounds file and bolds file.
    Outputs temporal mask, framewise displacement timeseries and censored bold files.
    """

    input_spec = _CensorScrubInputSpec
    output_spec = _CensorScrubOutputSpec

    def _run_interface(self, runtime):

        # Read in fmriprep confounds tsv to calculate FD
        fmriprep_confounds_df = pd.read_table(self.inputs.fmriprep_confounds_file)
        initial_volumes_to_drop = self.inputs.initial_volumes_to_drop

        if initial_volumes_to_drop == "auto":
            nss_cols = [
                c for c in fmriprep_confounds_df.columns
                if c.startswith("non_steady_state_outlier")
            ]
            initial_volumes_df = fmriprep_confounds_df[nss_cols]
        else:
            initial_volumes_columns = [f"dummy_volume{i}" for i in range(initial_volumes_to_drop)]
            initial_volumes_array = np.vstack(
                (
                    np.eye(initial_volumes_to_drop),
                    np.zeros(
                        (
                            fmriprep_confounds_df.shape[0] - initial_volumes_to_drop,
                            initial_volumes_to_drop,
                        ),
                    ),
                ),
            )
            initial_volumes_df = pd.DataFrame(
                data=initial_volumes_array,
                columns=initial_volumes_columns,
            )

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
        tmask = generate_mask(
            fd_res=fd_timeseries,
            fd_thresh=self.inputs.fd_thresh,
        )
        tmask_idx = np.where(tmask)[0]
        one_hot_outliers_columns = [
            f"framewise_displacement_outlier{i}" for i in range(tmask_idx.size)
        ]
        one_hot_outliers = np.zeros((tmask_idx.size, tmask_idx.max() + 1))
        one_hot_outliers[np.arange(tmask_idx.size), tmask_idx] = 1
        one_hot_outliers = np.vstack(
            (
                one_hot_outliers,
                np.zeros(
                    (
                        fmriprep_confounds_df.shape[0] - tmask_idx.max(),
                        one_hot_outliers.shape[1],
                    ),
                )
            )
        )
        one_hot_outliers_df = pd.DataFrame(
            data=one_hot_outliers,
            columns=one_hot_outliers_columns,
        )
        outliers_df = pd.concat((initial_volumes_df, one_hot_outliers_df), axis=1)

        self._results["tmask"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_desc-fd_outliers.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["filtered_motion"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_desc-filtered_motion.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )

        outliers_df.to_csv(
            self._results["tmask"],
            index=False,
            header=True,
            sep="\t",
        )

        motion_df.to_csv(
            self._results["filtered_motion"],
            index=False,
            header=True,
            sep="\t",
        )

        return runtime


class _InterpolateInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc=" censored or clean bold")
    bold_file = File(exists=True,
                     mandatory=True,
                     desc=" censored or clean bold")
    tmask = File(exists=True, mandatory=True, desc="temporal mask")
    mask_file = File(exists=False, mandatory=False, desc="required for nifti")
    TR = traits.Float(exists=True,
                      mandatory=True,
                      desc="repetition time in TR")


class _InterpolateOutputSpec(TraitedSpec):
    bold_interpolated = File(exists=True,
                             mandatory=True,
                             desc=" fmriprep censored")


class Interpolate(SimpleInterface):
    """Interpolates scrubbed/regressed BOLD data based on temporal mask.

    Interpolation takes in the scrubbed/regressed bold file and temporal mask,
    subs in the scrubbed values with 0, and then uses scipy's
    interpolate functionality to interpolate values into these 0s.
    It outputs the interpolated file.
    """

    input_spec = _InterpolateInputSpec
    output_spec = _InterpolateOutputSpec

    def _run_interface(self, runtime):
        # Read in regressed bold data and temporal mask
        # from censorscrub
        bold_data = read_ndata(datafile=self.inputs.in_file,
                               maskfile=self.inputs.mask_file)

        tmask_df = pd.read_table(self.inputs.tmask)
        tmask_arr = tmask_df["framewise_displacement"].values

        # check if any volumes were censored - if they were,
        # put 0s in their place.
        if bold_data.shape[1] != len(tmask_arr):
            data_with_zeros = np.zeros([bold_data.shape[0], len(tmask_arr)])
            data_with_zeros[:, tmask_arr == 0] = bold_data
        else:
            data_with_zeros = bold_data

        # interpolate the data using scipy's interpolation functionality
        interpolated_data = interpolate_masked_data(
            bold_data=data_with_zeros,
            tmask=tmask_arr,
            TR=self.inputs.TR,
        )

        # save out results
        self._results['bold_interpolated'] = fname_presuffix(
            self.inputs.in_file,
            newpath=os.getcwd(),
            use_ext=True,
        )

        write_ndata(
            data_matrix=interpolated_data,
            template=self.inputs.bold_file,
            mask=self.inputs.mask_file,
            TR=self.inputs.TR,
            filename=self._results['bold_interpolated'],
        )

        return runtime
