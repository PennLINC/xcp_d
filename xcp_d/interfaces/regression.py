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

LOGGER = logging.getLogger('nipype.interface')


class _regressInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="The bold file to be regressed")
    confounds = File(
        exists=True,
        mandatory=True,
        desc="The fMRIPrep confounds tsv after censoring")
    TR = traits.Float(exists=True, mandatory=True, desc="repetition time")
    mask = File(exists=False, mandatory=False, desc="brain mask nifti file")
    motion_filter_type = traits.Str(exists=False, mandatory=True)
    motion_filter_order = traits.Int(exists=False, mandatory=True)
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
                    manadatory=True,
                    desc=" residual file after regression")


class regress(SimpleInterface):
    r"""
    #TODO: Detailed docstring
    """

    input_spec = _regressInputSpec
    output_spec = _regressOutputSpec

    def _run_interface(self, runtime):

        # get the confound matrix
        confound = load_confound_matrix(original_file=self.inputs.original_file,
                                        datafile=self.inputs.in_file,
                                        custom_confounds=self.inputs.custom_confounds,
                                        confound_tsv=self.inputs.confounds)
        confound = confound.to_numpy().T
        # if self.inputs.custom_confounds:
        #     confound_custom = pd.read_table(self.inputs.custom_confounds,
        #                         header=None,delimiter=' ')
        #     confound = pd.concat((confound.T, confound_custom.T)).to_numpy()
        #     confound = np.nan_to_num(confound)
        # else:
        #     confound = confound.to_numpy().T

        # get the nifti/cifti  matrix
        data_matrix = read_ndata(datafile=self.inputs.in_file,
                                 maskfile=self.inputs.mask)
        # demean and detrend the data
        #
        # use afni order
        orderx = np.floor(1 + data_matrix.shape[1] * self.inputs.TR / 150)
        dd_data = demean_detrend_data(data=data_matrix,
                                      TR=self.inputs.TR,
                                      order=orderx)
        # confound = demean_detrend_data(data=confound,TR=self.inputs.tr,order=orderx)
        # regress the confound regressors from data
        resid_data = linear_regression(data=dd_data, confound=confound)

        # writeout the data
        if self.inputs.in_file.endswith('.dtseries.nii'):
            suffix = '_residualized.dtseries.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):
            suffix = '_residualized.nii.gz'

        # write the output out
        self._results['res_file'] = fname_presuffix(
            self.inputs.in_file,
            suffix=suffix,
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results['res_file'] = write_ndata(
            data_matrix=resid_data,
            template=self.inputs.in_file,
            filename=self._results['res_file'],
            mask=self.inputs.mask)
        return runtime


def linear_regression(data, confound):
    '''
     data :
       numpy ndarray- vertices by timepoints
     confound:
       nuissance regressors reg by timepoints
     return:
        residual matrix
    '''
    regr = LinearRegression(n_jobs=1)
    regr.fit(confound.T, data.T)
    y_pred = regr.predict(confound.T)

    return data - y_pred.T


def demean_detrend_data(data, TR, order):
    '''
    data should be voxels/vertices by timepoints dimension
    order=1
    # order of polynomial detrend is usually obtained from
    # order = floor(1 + TR*nVOLS / 150)
    TR= repetition time
    this can be use for both confound and bold
    '''

    # demean the data first, check if it has been demean

    mean_data = np.mean(data, axis=1)
    means_expanded = np.outer(mean_data, np.ones(data.shape[1]))
    demeand = data - means_expanded

    x = np.linspace(0, (data.shape[1] - 1) * TR, num=data.shape[1])
    predicted = np.zeros_like(demeand)
    for j in range(demeand.shape[0]):
        model = np.polyfit(x, demeand[j, :], order)
        predicted[j, :] = np.polyval(model, x)
    return demeand - predicted


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
