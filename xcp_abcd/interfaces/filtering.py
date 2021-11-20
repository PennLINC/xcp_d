# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling filtering.
    .. testsetup::
    # will comeback
"""
import numpy as np
import nibabel as nb
import pandas as pd
import sys 
from scipy.signal import butter, filtfilt
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, SimpleInterface)
from ..utils import(read_ndata, write_ndata)

LOGGER = logging.getLogger('nipype.interface') 


class _filterdataInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="Input file : either cifti or nifti file")
    tr = traits.Float(exists=True,mandatory=True, desc="repetition time")
    filter_order = traits.Int(exists=True,mandatory=True,default_value=2,desc="filter order")
    lowpass = traits.Float(exists=True,mandatory=True, 
                            default_value=0.10,desc="lowpass filter in Hz")
    highpass = traits.Float(exists=True,mandatory=True, 
                            default_value=0.01,desc="highpass filter in Hz")
    mask = File(exists=False, mandatory=False,
                          desc=" brain mask for nifti file")



class _filterdataOutputSpec(TraitedSpec):
    filt_file = File(exists=True, manadatory=True,
                                  desc=" filtered file")


class FilteringData(SimpleInterface):
    r"""filter the data.
    the filtering was setup with scipy signal
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> filt=FilteringData()
    >>> filt.inputs.in_file = reg._results['res_file']
    >>> filt.inputs.tr = 3
    >>> filt.inputs.lowpass = 0.08
    >>> filt.inputs.highpass = 0.01
    >>> filt.run()
    .. testcleanup::
    >>> tmpdir.cleanup()    
    """

    input_spec = _filterdataInputSpec
    output_spec = _filterdataOutputSpec

    def _run_interface(self, runtime):
        
        # get the nifti/cifti into  matrix
        data_matrix = read_ndata(datafile=self.inputs.in_file,
                           maskfile=self.inputs.mask)
        # filter the data 
        filt_data = butter_bandpass(data=data_matrix,fs=1/self.inputs.tr,
                      lowpass=self.inputs.lowpass,highpass=self.inputs.highpass,
                      order=self.inputs.filter_order)

        # writeout the data
        if self.inputs.in_file.endswith('.dtseries.nii'):
            suffix='_filtered.dtseries.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):
            suffix='_filtered.nii.gz'

        #write the output out
        self._results['filt_file'] = fname_presuffix(
                self.inputs.in_file,
                suffix=suffix, newpath=runtime.cwd,
                use_ext=False,)
        self._results['filt_file'] = write_ndata(data_matrix=filt_data, template=self.inputs.in_file, 
                filename=self._results['filt_file'],mask=self.inputs.mask)
        return runtime

def butter_bandpass(data,fs,lowpass,highpass,order=2):
    '''
    data : voxels/vertices by timepoints dimension
    fs : sampling frequency,=1/TR(s)
    lowpass frequency
    highpass frequency 
    '''
    
    nyq = 0.5 * fs
    lowcut = np.float(highpass) / nyq
    highcut = np.float(lowpass) / nyq
    
    b, a = butter(order/2, [lowcut, highcut], btype='band')
    #mean_data=np.mean(data,axis=1)
    y=np.zeros_like(data)
    #filter_dir = np.floor(order/2)

    # filter once first 
    for i in range(data.shape[0]):
        y[i,:] = filtfilt(b, a, data[i,:])
    
    # filter more if order is greater than 2,
    # then filter morei 
                
    #add mean back 
    #mean_datag = np.outer(mean_data, np.ones(data.shape[1]))
    return y 