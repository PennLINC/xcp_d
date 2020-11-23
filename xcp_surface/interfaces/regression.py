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
from utils import(read_ndata, write_ndata)

LOGGER = logging.getLogger('nipype.interface') 


class _regressInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="Input file ")
    confounds = File(exists=True, mandatory=True,
                          desc=" counfound regressors selected from fmriprep.")
    tr = traits.Float(exists=True,mandatory=True, desc="repetition time")
    customs_conf = File(exists=False, mandatory=False,
                          desc=" custom regressors like task or respiratory")
    mask = File(exists=False, mandatory=False,
                          desc=" mask for nifti file")


class _regressOutputSpec(TraitedSpec):
    res_file = File(exists=True, manadatory=True,
                                  desc=" residual file after regression")


class regress(SimpleInterface):
    """regress the regressors from cifti or nifti."""

    input_spec = _regressInputSpec
    output_spec = _regressOutputSpec

    def _run_interface(self, runtime):
        
        # get the confound matrix 
        confound = pd.read_csv(self.inputs.confounds,sep='\t',index=None).to_numpy()
        if self.inputs.customs_conf:
            confound_custom = pd.read_csv(self.inputs.customs_conf,
                                sep='\t',index=None).to_numpy()
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
    if np.mean(data) > 0.00000000001:
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


