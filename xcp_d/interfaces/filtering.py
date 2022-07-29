# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import numpy as np
from scipy.signal import butter, filtfilt
from nipype import logging
from ..utils.filemanip import fname_presuffix
from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec, File,
                                    SimpleInterface)
from ..utils import (read_ndata, write_ndata)

LOGGER = logging.getLogger('nipype.interface')


class _filterdataInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Interpolated file : either cifti or nifti file")
    TR = traits.Float(exists=True, mandatory=True, desc="Repetition time")
    filter_order = traits.Int(exists=True,
                              mandatory=True,
                              default_value=2,
                              desc="Filter order")
    lowpass = traits.Float(exists=True,
                           mandatory=True,
                           default_value=0.10,
                           desc="Lowpass filter frequency in Hz")
    highpass = traits.Float(exists=True,
                            mandatory=True,
                            default_value=0.01,
                            desc="Highpass filter frequency in Hz")
    mask = File(exists=False,
                mandatory=False,
                desc="Brain mask for nifti file")
    bandpass_filter = traits.Bool(exists=False,
                                  mandatory=True,
                                  desc="Apply bandpass(if true) or not")


class _filterdataOutputSpec(TraitedSpec):
    filtered_file = File(exists=True, manadatory=True, desc="Filtered file")


class FilteringData(SimpleInterface):
    """
    Put the data through a Butterworth filter.
    """

    input_spec = _filterdataInputSpec
    output_spec = _filterdataOutputSpec

    def _run_interface(self, runtime):

        # Get the nifti/cifti into  matrix format
        data_matrix = read_ndata(datafile=self.inputs.in_file,
                                 maskfile=self.inputs.mask)
        # Filter the data if set to True
        if self.inputs.bandpass_filter:
            filtered_data = butter_bandpass(data=data_matrix,
                                        fs=1 / self.inputs.TR,
                                        lowpass=self.inputs.lowpass,
                                        highpass=self.inputs.highpass,
                                        order=self.inputs.filter_order)
        else:
            filtered_data = data_matrix  # No filtering!

        # Write out the data
        if self.inputs.in_file.endswith('.dtseries.nii'):
            suffix = '_filtered.dtseries.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):
            suffix = '_filtered.nii.gz'

        # write the output out
        self._results['filtered_file'] = fname_presuffix(
            self.inputs.in_file,
            suffix=suffix,
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results['filtered_file'] = write_ndata(
            data_matrix=filtered_data,
            template=self.inputs.in_file,
            filename=self._results['filtered_file'],
            mask=self.inputs.mask)
        return runtime


def butter_bandpass(data, fs, lowpass, highpass, order=2):
    '''
    Data : Voxels by ntimepoints 
    fs : sampling frequency,=1/TR(s)
    lowpass frequency
    highpass frequency
    Returns filtered data
    '''

    nyq = 0.5 * fs
    lowcut = np.float(highpass) / nyq
    highcut = np.float(lowpass) / nyq

    b, a = butter(order / 2, [lowcut, highcut], btype='band')

    # Pad the data with zeros to avoid filter artifacts
    dimension_shape_1 = np.int(data.shape[1] / 4)
    padded_data = np.hstack((data[:, 0:dimension_shape_1], data, data[:, 0:dimension_shape_1]))

    # Get the mean of the data
    mean_data = np.mean(data, axis=1)

    filtered_data = np.zeros_like(padded_data)

    # Filter once first
    for i in range(padded_data.shape[0]):
        filtered_data[i, :] = filtfilt(b, a, padded_data[i, :])

    dimension_shape_2 = padded_data.shape[1]

    # Add mean back
    mean_datag = np.outer(mean_data, np.ones(filtered_data.shape[1]))

    filtered_data = np.add(mean_datag, filtered_data)

    return filtered_data[:, dimension_shape_1:(dimension_shape_2 - dimension_shape_1)]
