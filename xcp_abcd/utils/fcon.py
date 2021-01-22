# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
nifti functional connectivity
"""
from nilearn.input_data import NiftiLabelsMasker
import numpy as np 

def extract_timeseries_funct(in_file,
                             atlas,
                             timeseries,
                             fconmatrix):
    
    masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
    time_series = masker.fit_transform(in_file)
    correlation_matrices = np.corrcoef(time_series.T)
    
    np.savetxt(fconmatrix, correlation_matrices, delimiter=",")
    np.savetxt(timeseries, time_series, delimiter=",")

    return timeseries, fconmatrix


