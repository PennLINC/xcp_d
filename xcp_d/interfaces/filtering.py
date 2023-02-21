# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling filtering."""
import pandas as pd
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.utils import butter_bandpass
from xcp_d.utils.write_save import read_ndata, write_ndata

LOGGER = logging.getLogger("nipype.interface")


class _FilteringDataInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="Bold file")
    temporal_mask = File(exists=True, mandatory=True, desc="Temporal mask file")
    mask_metadata = traits.Dict(
        desc="Metadata associated with the temporal_mask output.",
    )
    TR = traits.Float(mandatory=True, desc="Repetition time")
    filter_order = traits.Int(mandatory=True, default_value=2, desc="Filter order")
    lowpass = traits.Float(mandatory=True, default_value=0.10, desc="Lowpass filter in Hz")
    highpass = traits.Float(mandatory=True, default_value=0.01, desc="Highpass filter in Hz")
    mask = File(exists=True, mandatory=False, desc="Brain mask for nifti file")
    bandpass_filter = traits.Bool(mandatory=True, desc="To apply bandpass or not")


class _FilteringDataOutputSpec(TraitedSpec):
    filtered_file = File(exists=True, mandatory=True, desc="Filtered file")
    filtered_mask = File(exists=True, mandatory=True, desc="Filtered temporal mask")
    mask_metadata = traits.Dict(
        desc="Metadata associated with the filtered_mask output.",
    )


class FilteringData(SimpleInterface):
    """Filter the data with scipy.signal."""

    input_spec = _FilteringDataInputSpec
    output_spec = _FilteringDataOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.bandpass_filter:
            LOGGER.debug("Not running bandpass filter.")
            self._results["filtered_file"] = self.inputs.in_file
            self._results["filtered_mask"] = self.inputs.temporal_mask
            self._results["tmask_metadata"] = self.inputs.tmask_metadata
            return runtime

        # get the nifti/cifti into  matrix
        data_matrix = read_ndata(datafile=self.inputs.in_file, maskfile=self.inputs.mask)
        temporal_mask = pd.read_table(self.inputs.temporal_mask)
        outliers_metadata = self.inputs.tmask_metadata

        filt_data = butter_bandpass(
            data=data_matrix.T,
            fs=1 / self.inputs.TR,
            lowpass=self.inputs.lowpass,
            highpass=self.inputs.highpass,
            order=self.inputs.filter_order,
        ).T
        temporal_mask["filtered_outliers"] = butter_bandpass(
            temporal_mask["framewise_displacement"].to_numpy()
        )
        outliers_metadata["filtered_outliers"] = {
            "Description": (
                "Outlier timeseries, bandpass filtered to identify volumes that have been "
                "contaminated by interpolated data"
            ),
            "Threshold": outliers_metadata["framewise_displacement"]["Threshold"],
        }
        self._results["tmask_metadata"] = outliers_metadata

        # write out the data
        if self.inputs.in_file.endswith(".dtseries.nii"):
            suffix = "_filtered.dtseries.nii"
        elif self.inputs.in_file.endswith(".nii.gz"):
            suffix = "_filtered.nii.gz"

        # write the output out
        self._results["filtered_file"] = fname_presuffix(
            self.inputs.in_file,
            suffix=suffix,
            newpath=runtime.cwd,
            use_ext=False,
        )
        write_ndata(
            data_matrix=filt_data,
            template=self.inputs.in_file,
            filename=self._results["filtered_file"],
            mask=self.inputs.mask,
        )
        self._results["filtered_mask"] = fname_presuffix(
            self.inputs.temporal_mask,
            suffix="_filtered",
            newpath=runtime.cwd,
            use_ext=True,
        )
        temporal_mask.to_csv(self._results["filtered_mask"], sep="\t", index=False)

        return runtime
