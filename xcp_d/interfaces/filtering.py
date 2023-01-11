# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Handling filtering."""
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
    TR = traits.Float(mandatory=True, desc="Repetition time")
    filter_order = traits.Int(mandatory=True, default_value=2, desc="Filter order")
    lowpass = traits.Float(mandatory=True, default_value=0.10, desc="Lowpass filter in Hz")
    highpass = traits.Float(mandatory=True, default_value=0.01, desc="Highpass filter in Hz")
    mask = File(exists=True, mandatory=False, desc="Brain mask for nifti file")
    bandpass_filter = traits.Bool(mandatory=True, desc="To apply bandpass or not")


class _FilteringDataOutputSpec(TraitedSpec):
    filtered_file = File(exists=True, mandatory=True, desc="Filtered file")


class FilteringData(SimpleInterface):
    """Filter the data with scipy.signal."""

    input_spec = _FilteringDataInputSpec
    output_spec = _FilteringDataOutputSpec

    def _run_interface(self, runtime):

        # get the nifti/cifti into  matrix
        data_matrix = read_ndata(datafile=self.inputs.in_file, maskfile=self.inputs.mask)
        # filter the data
        if self.inputs.bandpass_filter:
            filt_data = butter_bandpass(
                data=data_matrix,
                fs=1 / self.inputs.TR,
                lowpass=self.inputs.lowpass,
                highpass=self.inputs.highpass,
                order=self.inputs.filter_order,
            )
        else:
            filt_data = data_matrix  # no filtering!

        # writeout the data
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
        self._results["filtered_file"] = write_ndata(
            data_matrix=filt_data,
            template=self.inputs.in_file,
            filename=self._results["filtered_file"],
            mask=self.inputs.mask,
        )
        return runtime
