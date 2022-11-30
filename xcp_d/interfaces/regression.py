# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Regression interfaces."""
import numpy as np
import pandas as pd
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from sklearn.linear_model import LinearRegression

from xcp_d.utils.filemanip import fname_presuffix, split_filename
from xcp_d.utils.utils import demean_detrend_data
from xcp_d.utils.write_save import despikedatacifti, read_ndata, write_ndata

LOGGER = logging.getLogger("nipype.interface")


class _RegressInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="The bold file to be regressed")
    confounds = File(
        exists=True,
        mandatory=True,
        desc="The selected confounds for regression, in a TSV file.",
    )
    # TODO: Use Enum maybe?
    params = traits.Str(mandatory=True, desc="Parameter set to use.")
    TR = traits.Float(mandatory=True, desc="Repetition time")
    mask = File(exists=True, mandatory=False, desc="Brain mask for nifti files")


class _RegressOutputSpec(TraitedSpec):
    res_file = File(exists=True, mandatory=True, desc="Residual file after regression")


class Regress(SimpleInterface):
    """Takes in the confound tsv, turns it to a matrix and expands it.

    Custom confounds are added in during this step if present.

    Then reads in the bold file, does demeaning and a linear detrend.

    Finally, uses sklearns's Linear Regression to regress out the confounds from
    the bold files and returns the residual image, as well as the confounds for testing.
    """

    input_spec = _RegressInputSpec
    output_spec = _RegressOutputSpec

    def _run_interface(self, runtime):

        # Get the confound matrix
        confound = pd.read_table(self.inputs.confounds)

        # Any columns starting with "signal__" are assumed to be signal regressors
        signal_columns = [c for c in confound.columns if c.startswith("signal__")]
        if signal_columns:
            LOGGER.info(
                "Performing nonaggressive denoising using the following signal columns: "
                f"{', '.join(signal_columns)}"
            )
            noise_columns_idx = [
                i for i, c in enumerate(confound.columns) if c not in signal_columns
            ]

        # Transpose confounds matrix to line up with bold matrix
        confound_arr = confound.to_numpy().T

        # Get the nifti/cifti matrix
        bold_matrix = read_ndata(datafile=self.inputs.in_file, maskfile=self.inputs.mask)

        # Demean and detrend the data
        demeaned_detrended_data = demean_detrend_data(data=bold_matrix)

        # Regress out the confounds via linear regression from sklearn
        if demeaned_detrended_data.shape[1] < confound_arr.shape[0]:
            LOGGER.warning(
                "Warning: Regression might not be effective due to rank deficiency, "
                "i.e., the number of volumes in the bold file is smaller than the number "
                "of regressors."
            )

        regression = LinearRegression(n_jobs=1)
        regression.fit(confound_arr.T, demeaned_detrended_data.T)

        if signal_columns:
            betas = regression.coef_
            pred_noise_data = np.dot(
                confound_arr[noise_columns_idx, :],
                betas[noise_columns_idx, :],
            )
            residuals = demeaned_detrended_data - pred_noise_data.T
        else:
            pred_noise_data = regression.predict(confound_arr.T)
            residuals = demeaned_detrended_data - pred_noise_data.T

        # Write out the data
        _, _, extension = split_filename(self.inputs.in_file)
        suffix = f"_residualized{extension}"

        self._results["res_file"] = fname_presuffix(
            self.inputs.in_file,
            suffix=suffix,
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["res_file"] = write_ndata(
            data_matrix=residuals,
            template=self.inputs.in_file,
            filename=self._results["res_file"],
            mask=self.inputs.mask,
        )
        return runtime


class _CiftiDespikeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc=" cifti  file ")
    TR = traits.Float(mandatory=True, desc="repetition time")


class _CiftiDespikeOutputSpec(TraitedSpec):
    des_file = File(exists=True, mandatory=True, desc=" despike cifti")


class CiftiDespike(SimpleInterface):
    """Despike a CIFTI file."""

    input_spec = _CiftiDespikeInputSpec
    output_spec = _CiftiDespikeOutputSpec

    def _run_interface(self, runtime):

        # write the output out
        self._results["des_file"] = fname_presuffix(
            "ciftidepike",
            suffix=".dtseries.nii",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["des_file"] = despikedatacifti(
            cifti=self.inputs.in_file, TR=self.inputs.TR, basedir=runtime.cwd
        )
        return runtime
