# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from nipype import logging
from sklearn.linear_model import LinearRegression
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (traits, TraitedSpec,
                                    BaseInterfaceInputSpec, File,
                                    SimpleInterface)
from ..utils import (read_ndata, write_ndata, despikedatacifti, load_confound_matrix)
from os.path import exists
LOGGER = logging.getLogger('nipype.interface')


class _regressInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="The bold file to be regressed")
    confounds = File(
        exists=True,
        mandatory=True,
        desc="The fMRIPrep confounds tsv after censoring")
    TR = traits.Float(exists=True, mandatory=True, desc="Repetition time")
    mask = File(exists=False, mandatory=False, desc="Brain mask for nifti files")
    original_file = traits.Str(exists=True, mandatory=False,
                               desc="Name of original bold file- helps load in the confounds"
                               "file down the line using the original path name")
    custom_confounds = traits.Either(traits.Undefined,
                                     File,
                                     desc="Name of custom confounds file, or True",
                                     exists=False,
                                     mandatory=False)


class _regressOutputSpec(TraitedSpec):
    res_file = File(exists=True,
                    mandatory=True,
                    desc="Residual file after regression")


class regress(SimpleInterface):
    r"""
    Takes in the confound tsv, turns it to a matrix and expands it. Custom
    confounds are added in during this step if present.

    Then reads in the bold file, does demeaning and a linear detrend.

    Finally, uses sklearns's Linear  Regression to regress out the confounds from
    the bold files and returns the residual image.
    """

    input_spec = _regressInputSpec
    output_spec = _regressOutputSpec

    def _run_interface(self, runtime):

        # Get the confound matrix
        # Do we have custom confounds?
        if self.inputs.custom_confounds and exists(self.inputs.custom_confounds):
            confound = load_confound_matrix(original_file=self.inputs.original_file,
                                            datafile=self.inputs.in_file,
                                            custom_confounds=self.inputs.custom_confounds,
                                            confound_tsv=self.inputs.confounds)
        else:  # No custom confounds
            confound = load_confound_matrix(original_file=self.inputs.original_file,
                                            datafile=self.inputs.in_file,
                                            confound_tsv=self.inputs.confounds)

        confound = confound.to_numpy().T  # Transpose confounds matrix to line up with bold matrix

        # Get the nifti/cifti matrix
        bold_matrix = read_ndata(datafile=self.inputs.in_file,
                                 maskfile=self.inputs.mask)

        # Demean and detrend the data
        demeaned_detrended_data = demean_detrend_data(data=bold_matrix,
                                                      TR=self.inputs.TR)

        # Regress out the confounds via linear regression from sklearn
        residualized_data = linear_regression(data=demeaned_detrended_data, confound=confound)

        # Write out the data
        if self.inputs.in_file.endswith('.dtseries.nii'):  # If cifti
            suffix = '_residualized.dtseries.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):  # If nifti
            suffix = '_residualized.nii.gz'

        # write the output out
        self._results['res_file'] = fname_presuffix(
            self.inputs.in_file,
            suffix=suffix,
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results['res_file'] = write_ndata(
            data_matrix=residualized_data,
            template=self.inputs.in_file,
            filename=self._results['res_file'],
            mask=self.inputs.mask)
        return runtime


def linear_regression(data, confound):
    '''
     data :
       numpy ndarray- vertices by timepoints for bold file
     confound:
       nuissance regressors - vertices by timepoints for confounds matrix
     returns:
        residual matrix after regression
    '''
    regression = LinearRegression(n_jobs=1)
    regression.fit(confound.T, data.T)
    y_predicted = regression.predict(confound.T)

    return data - y_predicted.T


def demean_detrend_data(data, TR):
    '''
    data:
        numpy ndarray- vertices by timepoints for bold file
    TR:
        Repetition time

    Returns demeaned and detrended data
    '''
    order = 1  # For linear detrending
    # Demean the data
    mean = np.mean(data, axis=1)  # Get the mean of each voxel across timepoints
    # Replace each timepoint with the average
    mean_data = np.outer(mean, np.ones(data.shape[1]))
    demeaned = data - mean_data  # The demeaned data has the average across timepoints subtracted
    # out

    # Create an array of false slice times with the same number of timepoints
    evenly_spaced_slice_times = np.linspace(0, (data.shape[1] - 1) * TR, num=data.shape[1])
    # An array of zeros with the same shape as the bold file
    predicted_values = np.zeros_like(demeaned)
    for voxel in range(demeaned.shape[0]):  # Looping through each voxel
        # Create a linear model using the slice times array and the timepoints for each voxel
        model = np.polyfit(evenly_spaced_slice_times, demeaned[voxel, :], order)
        # Generate predicted values the values using the array slice times and model from the
        # previous step
        predicted_values[voxel, :] = np.polyval(model, evenly_spaced_slice_times)
    return demeaned - predicted_values  # Subtract these predicted values from the demeaned data


class _ciftidespikeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc=" cifti  file ")
    tr = traits.Float(exists=True, mandatory=True, desc="repetition time")


class _ciftidespikeOutputSpec(TraitedSpec):
    des_file = File(exists=True, manadatory=True, desc=" despike cifti")


class ciftidespike(SimpleInterface):
    r"""


    """

    input_spec = _ciftidespikeInputSpec
    output_spec = _ciftidespikeOutputSpec

    def _run_interface(self, runtime):

        # write the output out
        self._results['des_file'] = fname_presuffix(
            'ciftidepike',
            suffix='.dtseries.nii',
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results['des_file'] = despikedatacifti(cifti=self.inputs.in_file,
                                                     tr=self.inputs.tr,
                                                     basedir=runtime.cwd)
        return runtime
