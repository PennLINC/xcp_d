# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling regression.
    .. testsetup::
    # will comeback
"""

import os
import sys 
import re
import shutil
import nibabel as nb
import numpy as np
import pandas as pd
from nipype import logging
from sklearn.linear_model import LinearRegression
from nilearn.signal import clean 
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)
from ..utils import(read_ndata, write_ndata,despikedatacifti)

LOGGER = logging.getLogger('nipype.interface') 


class _regressInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="Input file either cifti or nifti file ")
    confounds = File(exists=True, mandatory=True,
                          desc=" confound regressors selected from fmriprep's confound matrix.")
    tr = traits.Float(exists=True,mandatory=True, desc="repetition time")
    custom_conf = File(exists=False, mandatory=False,
                          desc=" custom regressors like task or respiratory with the same length as in_file")
    mask = File(exists=False, mandatory=False,
                          desc=" brain mask nifti file")
    

class _regressOutputSpec(TraitedSpec):
    res_file = File(exists=True, manadatory=True,
                                  desc=" residual file after regression")


class regress(SimpleInterface):
    r"""
    regress the nuissance regressors from cifti or nifti.
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> reg = regress()
    >>> reg.inputs.in_file = datafile
    >>> reg.inputs.confounds = confoundfile # selected with ConfoundMatrix() or custom
    >>> reg.inputs.tr = 3
    >>> reg.run()
    
    
    """

    input_spec = _regressInputSpec
    output_spec = _regressOutputSpec

    def _run_interface(self, runtime):
        
        # get the confound matrix 
        confound = pd.read_csv(self.inputs.confounds,header=None).to_numpy().T
        if self.inputs.custom_conf:
            confound_custom = pd.read_csv(self.inputs.custom_conf,
                                header=None).to_numpy().T
            confound = np.hstack((confound, confound_custom))
        
        # get the nifti/cifti  matrix
        data_matrix = read_ndata(datafile=self.inputs.in_file,
                           maskfile=self.inputs.mask)
        # demean and detrend the data 
        dd_data = demean_detrend_data(data=data_matrix,TR=self.inputs.tr,order=1)
        # regress the confound regressors from data
        resid_data = linear_regression(data=dd_data, confound=confound)
        
        # writeout the data
        if self.inputs.in_file.endswith('.dtseries.nii'):
            suffix='_residualized.dtseries.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):
            suffix='_residualized.nii.gz'

        #write the output out
        self._results['res_file'] = fname_presuffix(
                self.inputs.in_file,
                suffix=suffix, newpath=runtime.cwd,
                use_ext=False,)
        self._results['res_file'] = write_ndata(data_matrix=resid_data, template=self.inputs.in_file, 
                filename=self._results['res_file'],mask=self.inputs.mask)
        return runtime






def linear_regression(data,confound):
    
    '''
     data :
       numpy ndarray- vertices by timepoints
     confound: 
       nuissance regressors reg by timepoints
     return: 
        residual matrix 
    '''
    regr = LinearRegression()
    regr.fit(confound.T,data.T)
    y_pred = regr.predict(confound.T)
    return data - y_pred.T

def demean_detrend_data(data,TR,order=1):
    '''
    data should be voxels/vertices by timepoints dimension
    order=1
    # order of polynomial detrend is usually obtained from 
    # order = floor(1 + TR*nVOLS / 150)
    TR= repetition time
    this can be use for both confound and bold 
    '''
    
    # demean the data first, check if it has been demean
    if abs(np.mean(data)) > 1e-8:
        mean_data =np.mean(data,axis=1)
        means_expanded = np.outer(mean_data, np.ones(data.shape[1]))
        demeand = data - means_expanded
    else:
        demeand=data

    x = np.linspace(0,(data.shape[1]-1)*TR,num=data.shape[1])
    predicted=np.zeros_like(demeand)
    for j in range(demeand.shape[0]):
        model = np.polyfit(x,demeand[j,:],order)
        predicted[j,:] = np.polyval(model, x) 
    return demeand - predicted

class _ciftidespikeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc=" cifti  file ")
    tr = traits.Float(exists=True,mandatory=True, desc="repetition time")

class _ciftidespikeOutputSpec(TraitedSpec):
    des_file = File(exists=True, manadatory=True,
                                  desc=" despike cifti")


class ciftidespike(SimpleInterface):
    r"""
    regress the nuissance regressors from cifti or nifti.
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> reg = ciftidespike()
    >>> reg.inputs.in_file = datafile
    >>> reg.inputs.tr = 3
    >>> reg.run()
    
    
    """

    input_spec = _ciftidespikeInputSpec
    output_spec = _ciftidespikeOutputSpec

    def _run_interface(self, runtime):

        #write the output out
        self._results['des_file'] = fname_presuffix(
                'ciftidepike',
                suffix='.dtseries.nii', newpath=runtime.cwd,
                use_ext=False,)
        self._results['des_file'] = despikedatacifti(cifti=self.inputs.in_file,
                                    tr=self.inputs.tr,basedir=runtime.cwd)
        return runtime
