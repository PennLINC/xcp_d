import numpy as np
import os
import pandas as pd
import nibabel as nb
from ..utils import (read_ndata, write_ndata, compute_FD,
                     generate_mask, interpolate_masked_data)
from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec, File,
                                    SimpleInterface)
from ..utils.filemanip import fname_presuffix


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

    The  number of volumes to be removed has been calculated in a previous
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
            custom_confounds_tsv_undropped = pd.read_table(
                self.inputs.custom_confounds, header=None)
            custom_confounds_tsv_dropped = custom_confounds_tsv_undropped.drop(
                np.arange(volumes_to_drop))

        # Save out results
        dropped_confounds_df.to_csv(dropped_confounds_file, sep="\t", index=False)
        # Write to output node
        self._results['bold_file_dropped_TR'] = dropped_bold_file
        self._results['fmriprep_confounds_file_dropped_TR'] = dropped_confounds_file
        if self.inputs.custom_confounds:
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
    fd_thresh = traits.Float(exists=True, mandatory=False, default_value=0.2, desc="Framewise displacement threshold. All"
                             "values above this will be dropped.")
    custom_confounds = traits.Either(traits.Undefined,
                                     File,
                                     desc="Name of custom confounds file, or True",
                                     exists=False,
                                     mandatory=False)
    fmriprep_confounds_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv after removing dummy time, if any")
    head_radius = traits.Float(exists=True,
                               mandatory=False,
                               default_value=50,
                               desc="Head radius in mm ")
    motion_filter_type = traits.Str(exists=False, mandatory=True)
    motion_filter_order = traits.Int(exists=False, mandatory=True)
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
    low_freq = traits.Float(
        exists=True,
        mandatory=True,
        desc="Low frequency band for Notch filter in breaths per min (bpm)")
    high_freq = traits.Float(
        exists=True,
        mandatory=True,
        desc="High frequency band for Notch filter in breaths per min (bpm)")


class _CensorScrubOutputSpec(TraitedSpec):
    bold_censored = File(exists=True,
                         manadatory=True,
                         desc="FD-censored bold file")

    fmriprep_confounds_censored = File(exists=True,
                                       mandatory=True,
                                       desc="fmriprep_confounds_tsv censored")
    custom_confounds_censored = File(exists=False,
                                     mandatory=False,
                                     desc="custom_confounds_tsv censored")
    tmask = File(exists=True, mandatory=True,
                 desc="Temporal mask; all values above fd_thresh set to 1")
    fd_timeseries = File(exists=True, mandatory=True, desc="Framewise displacement timeseries")


class CensorScrub(SimpleInterface):
    r"""
    Takes in confound files, bold file to be censored, and information about filtering 
    - including band stop values and motion filter type. Then proceeds to create a 
    motion-filtered confounds matrix and recalculates FDfrom filtered motion parameters. 
    Finally generates temporal mask with volumes above fd threshold set to 1, then dropped
    from both confounds file and bolds file. Outputs temporal mask, framewise displacement
    timeseries and censored bold files. 
    """
    input_spec = _CensorScrubInputSpec
    output_spec = _CensorScrubOutputSpec

    def _run_interface(self, runtime):
        from ..utils.confounds import (load_motion)

        # Read in fmriprep confounds tsv to calculate FD
        fmriprep_confounds_tsv_uncensored = pd.read_table(self.inputs.fmriprep_confounds_file)
        motion_confounds = load_motion(
            fmriprep_confounds_tsv_uncensored.copy(),
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            freqband=[self.inputs.low_freq, self.inputs.high_freq])
        motion_df = pd.DataFrame(data=motion_confounds.values,
                                 columns=[
                                     "rot_x", "rot_y", "rot_z", "trans_x",
                                     "trans_y", "trans_z"
                                 ])
        fd_timeseries_uncensored = compute_FD(confound=motion_df,
                                              head_radius=self.inputs.head_radius)

        # Read in custom confounds file (if any) and bold file to be censored
        bold_file_uncensored = nb.load(self.inputs.in_file).get_fdata()
        if self.inputs.custom_confounds:
            custom_confounds_tsv_uncensored = pd.read_csv(
                self.inputs.custom_confounds, header=None)
        # Generate temporal mask with all timepoints have FD over threshold
        # set to 1 and then dropped.
        tmask = generate_mask(fd_res=fd_timeseries_uncensored,
                              fd_thresh=self.inputs.fd_thresh)
        if np.sum(tmask) > 0:  # If any FD values exceed the threshold
            if nb.load(self.inputs.in_file).ndim > 2:  # If Nifti
                bold_file_censored = bold_file_uncensored[:, :, :, tmask == 0]
            else:
                bold_file_censored = bold_file_uncensored[tmask == 0, :]
            fmriprep_confounds_tsv_censored = fmriprep_confounds_tsv_uncensored.drop(
                fmriprep_confounds_tsv_uncensored.index[np.where(tmask == 1)])
            if self.inputs.custom_confounds:  # If custom regressors are present
                custom_confounds_tsv_censored = custom_confounds_tsv_uncensored.drop(
                    custom_confounds_tsv_uncensored.index[np.where(tmask == 1)])
        else:  # No censoring needed
            bold_file_censored = bold_file_uncensored
            fmriprep_confounds_tsv_censored = fmriprep_confounds_tsv_uncensored
            if self.inputs.custom_confounds:
                custom_confounds_tsv_censored = custom_confounds_tsv_uncensored

        #  Turn censored bold into nifti image
        if nb.load(self.inputs.in_file).ndim > 2:
            bold_file_censored = nb.Nifti1Image(bold_file_censored,
                                                affine=nb.load(self.inputs.in_file).affine,
                                                header=nb.load(self.inputs.in_file).header)
        else:
            # If it's a Cifti Image:
            original_image = nb.load(self.inputs.in_file)
            time_axis, brain_model_axis = [
                original_image.header.get_axis(i) for i in range(original_image.ndim)]
            new_total_volumes = bold_file_censored.shape[0]
            censored_time_axis = time_axis[:new_total_volumes]
            # Note: not an error. A time axis cannot be accessed with irregularly
            # spaced values. Since we use the tmask for marking the volumes removed,
            # the time axis also is not used further in XCP.
            censored_header = nb.cifti2.Cifti2Header.from_axes(
                (censored_time_axis, brain_model_axis))
            bold_file_censored = nb.Cifti2Image(
                bold_file_censored,
                header=censored_header,
                nifti_header=original_image.nifti_header)

        # get the output
        self._results['bold_censored'] = fname_presuffix(self.inputs.in_file,
                                                         suffix='_censored',
                                                         newpath=os.getcwd(),
                                                         use_ext=True)
        self._results['fmriprep_confounds_censored'] = fname_presuffix(
            self.inputs.in_file,
            suffix='_fmriprep_confounds_censored.tsv',
            newpath=os.getcwd(),
            use_ext=False)
        if self.inputs.custom_confounds:
            self._results['custom_confounds_censored'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_custom_confounds_censored.tsv',
                newpath=os.getcwd(),
                use_ext=False)
        self._results['tmask'] = fname_presuffix(self.inputs.in_file,
                                                 suffix='_temporal_mask.tsv',
                                                 newpath=os.getcwd(),
                                                 use_ext=False)
        self._results['fd_timeseries'] = fname_presuffix(
            self.inputs.in_file,
            suffix='_fd_timeseries.tsv',
            newpath=os.getcwd(),
            use_ext=False)

        bold_file_censored.to_filename(self._results['bold_censored'])

        fmriprep_confounds_tsv_censored.to_csv(self._results['fmriprep_confounds_censored'],
                                               index=False,
                                               header=True,
                                               sep="\t")
        np.savetxt(self._results['tmask'], tmask, fmt="%d", delimiter=',')
        np.savetxt(self._results['fd_timeseries'],
                   fd_timeseries_uncensored,
                   fmt="%1.4f",
                   delimiter=',')
        if self.inputs.custom_confounds:
            custom_confounds_tsv_censored.to_csv(self._results['custom_confounds_censored'],
                                                 index=False,
                                                 header=False,
                                                 sep="\t")  # Assuming input is tab separated!
        return runtime


# interpolation


class _interpolateInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc=" censored or clean bold")
    bold_file = File(exists=True,
                     mandatory=True,
                     desc=" censored or clean bold")
    tmask = File(exists=True, mandatory=True, desc="temporal mask")
    mask_file = File(exists=False, mandatory=False, desc="required for nifti")
    TR = traits.Float(exists=True,
                      mandatory=True,
                      desc="repetition time in TR")


class _interpolateOutputSpec(TraitedSpec):
    bold_interpolated = File(exists=True,
                             manadatory=True,
                             desc=" fmriprep censored")


class interpolate(SimpleInterface):
    """
    Interpolation takes in the scrubbed/regressed bold file and temporal mask,
    subs in the scrubbed values with 0, and then uses scipy's
    interpolate functionality to interpolate values into these 0s.
    It outputs the interpolated file. 
    """
    input_spec = _interpolateInputSpec
    output_spec = _interpolateOutputSpec

    def _run_interface(self, runtime):
        # Read in regressed bold data and temporal mask
        # from censorscrub
        bold_data = read_ndata(datafile=self.inputs.in_file,
                               maskfile=self.inputs.mask_file)

        tmask = np.loadtxt(self.inputs.tmask)
        # check if any volumes were censored - if they were,
        # put 0s in their place.
        if bold_data.shape[1] != len(tmask):
            data_with_zeros = np.zeros([bold_data.shape[0], len(tmask)])
            data_with_zeros[:, tmask == 0] = bold_data
        else:
            data_with_zeros = bold_data
        # interpolate the data using scipy's interpolation functionality
        interpolated_data = interpolate_masked_data(bold_data=data_with_zeros,
                                                    tmask=tmask,
                                                    TR=self.inputs.TR)
        # save out results 
        self._results['bold_interpolated'] = fname_presuffix(
            self.inputs.in_file, newpath=os.getcwd(), use_ext=True)

        write_ndata(data_matrix=interpolated_data,
                    template=self.inputs.bold_file,
                    mask=self.inputs.mask_file,
                    TR=self.inputs.TR,
                    filename=self._results['bold_interpolated'])

        return runtime
