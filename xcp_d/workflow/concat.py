# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import fnmatch
import glob
from nipype import logging
from ..interfaces.connectivity import ApplyTransformsx
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from ..utils import bid_derivative
from ..utils import (make_dcan_df, collect_data, get_customfile, select_cifti_bold,
                     select_registrationfile, extract_t1w_seg)
import h5py
from natsort import natsorted

LOGGER = logging.getLogger('nipype.workflow')


def init_concat_wf(cifti,
                   custom_confounds,                    
                   layout, 
                   fmri_dir, 
                   output_dir,  
                   omp_nthreads,
                   name='concat_wf'):
    """
    This workflow organizes the concatenation workflow.
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflow.concat import init_concat_wf
            wf = init_concat_wf(cifti,
                   custom_confounds,                    
                   layout, 
                   fmri_dir, 
                   output_dir,  
                   omp_nthreads,
                   name='concat_wf'):
    Parameters
    ----------
    cifti : bool
        To postprocessed cifti files instead of nifti
    custom_confounds: str
        path to custom nuisance regressors
    fmri_dir : Path
        fmriprep/nibabies output directory
    omp_nthreads : int
        Maximum number of threads an individual process may use
    output_dir : str
        Directory in which to save xcp_d output
    omp_nthreads : int
        Maximum number of threads an individual process may use

    Inputs
    ------
    processed_bold
        list of clean bold after regression and filtering
    smoothed_bold
        list of smoothed clean bold timeseries
    confounds
        list of fmriprep confound files
    custom_confounds
        list of input custom confound files
    filtered_confounds
        list of motion-filtered confound files 
        (no frame censoring applied)
    filtered_custom_confounds            
        list of motion-filtered custom confound files 
        (no frame censoring applied)
    xxxxxxxxxxxxxxxxxxx

    Outputs
    ------
    concat_processed_bold
        list of clean bold after regression and filtering
    concat_smoothed_bold
        list of smoothed clean bold timeseries
    concat_fd_unfiltered
        list of pre-motion-filtered framewise displacement tsv files,
        concatenated per task
    concat_fd
        list of motion-filtered framewise displacement tsv files,
        concatenated per task
    concat_confounds
        list of fmriprep confound tsv files
    concat_filtered_confounds
        list of motion-filtered confound tsv files 
        (no frame censoring applied)
    concat_custom_confounds
        list of input custom confound tsv files
    concat_filtered_custom_confounds            
        list of motion-filtered custom confound tsv files 
        (no frame censoring applied)
    concat_dcan_motion
        concatenated per-task hdf5 motion data file, derived from fmriprep confounds,
        with fields corresponding to DCAN pipeline's
        "_power2014_FD_only.mat", concatenated within task
    concat_filtered_dcan_motion
        concatenated per-task hdf5 motion data file, derived from fmriprep confounds,
        with fields corresponding to DCAN pipeline's
        "_power2014_FD_only.mat", concatenated within task, 
        after applying motion filter 
    concat_custom_dcan_motion
        concatenated per-task hdf5 motion data file, derived from custom confounds,
        with fields corresponding to DCAN pipeline's
        "_power2014_FD_only.mat", concatenated within task
    concat_custom_filtered_dcan_motion
        concatenated per-task hdf5 motion data file, derived from custom confounds,
        with fields corresponding to DCAN pipeline's
        "_power2014_FD_only.mat", concatenated within task, 
        after applying motion filter 
"""
    workflow = Workflow(name=name)
    return workflow
