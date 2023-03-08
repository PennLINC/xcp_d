"""Interfaces for Nilearn code."""
import os

from nilearn import maskers
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.nilearn import NilearnBaseInterface

from xcp_d.utils.utils import denoise_with_nilearn
from xcp_d.utils.write_save import read_ndata, write_ndata


class _MergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        File(exists=True),
        mandatory=True,
        desc="A list of images to concatenate.",
    )
    out_file = File(
        "concat_4d.nii.gz",
        usedefault=True,
        exists=False,
        desc="The name of the concatenated file to write out. concat_4d.nii.gz by default.",
    )


class _MergeOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Concatenated output file.",
    )


class Merge(NilearnBaseInterface, SimpleInterface):
    """Merge images."""

    input_spec = _MergeInputSpec
    output_spec = _MergeOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import concat_imgs

        img_4d = concat_imgs(self.inputs.in_files)
        self._results["out_file"] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_4d.to_filename(self._results["out_file"])

        return runtime


class _SmoothInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="An image to smooth.",
    )
    fwhm = traits.Either(
        traits.Float(),
        traits.List(
            traits.Float(),
            minlen=3,
            maxlen=3,
        ),
        desc="Smoothing strength, as a full-width at half maximum, in millimeters.",
    )
    out_file = File(
        "smooth_img.nii.gz",
        usedefault=True,
        exists=False,
        desc="The name of the smoothed file to write out. smooth_img.nii.gz by default.",
    )


class _SmoothOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Smoothed output file.",
    )


class Smooth(NilearnBaseInterface, SimpleInterface):
    """Smooth image."""

    input_spec = _SmoothInputSpec
    output_spec = _SmoothOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import smooth_img

        img_smoothed = smooth_img(self.inputs.in_file, fwhm=self.inputs.fwhm)
        self._results["out_file"] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_smoothed.to_filename(self._results["out_file"])

        return runtime


class _BinaryMathInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="An image to do math on.",
    )
    expression = traits.String(
        mandatory=True,
        desc="A mathematical expression to apply to the image. Must have 'img' in it.",
    )
    out_file = File(
        "out_img.nii.gz",
        usedefault=True,
        exists=False,
        desc="The name of the mathified file to write out. out_img.nii.gz by default.",
    )


class _BinaryMathOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Mathified output file.",
    )


class BinaryMath(NilearnBaseInterface, SimpleInterface):
    """Do math on an image."""

    input_spec = _BinaryMathInputSpec
    output_spec = _BinaryMathOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import math_img

        img_mathed = math_img(self.inputs.expression, img=self.inputs.in_file)
        self._results["out_file"] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_mathed.to_filename(self._results["out_file"])

        return runtime


class _ResampleToImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="An image to average over time.",
    )
    target_file = File(
        exists=True,
        mandatory=True,
        desc="",
    )
    out_file = File(
        "out_img.nii.gz",
        usedefault=True,
        exists=False,
        desc="The name of the resampled file to write out. out_img.nii.gz by default.",
    )


class _ResampleToImageOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Resampled output file.",
    )


class ResampleToImage(NilearnBaseInterface, SimpleInterface):
    """Resample a source image on a target image.

    No registration is performed: the image should already be aligned.
    """

    input_spec = _ResampleToImageInputSpec
    output_spec = _ResampleToImageOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import resample_to_img

        resampled_img = resample_to_img(
            source_img=self.inputs.in_file,
            target_img=self.inputs.target_file,
            interpolation="continuous",
        )
        self._results["out_file"] = os.path.join(runtime.cwd, self.inputs.out_file)
        resampled_img.to_filename(self._results["out_file"])


class _DenoiseImageInputSpec(BaseInterfaceInputSpec):
    """Used directly by the CIFTI interface, and modified slightly for the NIFTI one."""

    preprocessed_bold = File(
        exists=True,
        mandatory=True,
        desc=(
            "Preprocessed BOLD data, after dummy volume removal, "
            "but without any additional censoring."
        ),
    )
    confounds_file = File(
        exists=True,
        mandatory=True,
        desc="A tab-delimited file containing the confounds to remove from the BOLD data.",
    )
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc="The tab-delimited high-motion outliers file.",
    )
    TR = traits.Float(mandatory=True, desc="Repetition time")
    bandpass_filter = traits.Bool(mandatory=True, desc="To apply bandpass or not")
    low_pass = traits.Float(mandatory=True, default_value=0.10, desc="Lowpass filter in Hz")
    high_pass = traits.Float(mandatory=True, default_value=0.01, desc="Highpass filter in Hz")
    filter_order = traits.Int(mandatory=True, default_value=2, desc="Filter order")


class _DenoiseImageOutputSpec(TraitedSpec):
    """Used by both the CIFTI and NIFTI interfaces."""

    uncensored_denoised_bold = File(
        exists=True,
        desc=(
            "The result of denoising the full (uncensored) preprocessed BOLD data using "
            "betas estimated using the *censored* BOLD data and nuisance regressors."
        ),
    )
    interpolated_filtered_bold = File(
        exists=True,
        desc=(
            "The result of denoising the censored preprocessed BOLD data, "
            "followed by cubic spline interpolation and band-pass filtering."
        ),
    )


class DenoiseCifti(NilearnBaseInterface, SimpleInterface):
    """Denoise a CIFTI BOLD file with Nilearn."""

    input_spec = _DenoiseImageInputSpec
    output_spec = _DenoiseImageOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.bandpass_filter:
            low_pass, high_pass = None, None
        else:
            low_pass, high_pass = self.inputs.low_pass, self.inputs.high_pass

        preprocessed_bold_arr = read_ndata(self.inputs.preprocessed_bold)

        # Transpose from SxT (xcpd order) to TxS (nilearn order)
        preprocessed_bold_arr = preprocessed_bold_arr.T

        (
            uncensored_denoised_bold,
            interpolated_filtered_bold,
        ) = denoise_with_nilearn(
            preprocessed_bold=preprocessed_bold_arr,
            confounds_file=self.inputs.confounds_file,
            temporal_mask=self.inputs.temporal_mask,
            low_pass=low_pass,
            high_pass=high_pass,
            filter_order=self.inputs.filter_order,
            TR=self.inputs.TR,
        )

        # Transpose from TxS (nilearn order) to SxT (xcpd order)
        uncensored_denoised_bold = uncensored_denoised_bold.T
        interpolated_filtered_bold = interpolated_filtered_bold.T

        self._results["uncensored_denoised_bold"] = os.path.join(
            runtime.cwd,
            "uncensored_denoised.dtseries.nii",
        )
        write_ndata(
            uncensored_denoised_bold,
            template=self.inputs.preprocessed_bold,
            filename=self._results["uncensored_denoised_bold"],
            TR=self.inputs.TR,
        )

        self._results["interpolated_filtered_bold"] = os.path.join(
            runtime.cwd,
            "filtered_denoised.dtseries.nii",
        )
        write_ndata(
            interpolated_filtered_bold,
            template=self.inputs.preprocessed_bold,
            filename=self._results["interpolated_filtered_bold"],
            TR=self.inputs.TR,
        )

        return runtime


class _DenoiseNiftiInputSpec(_DenoiseImageInputSpec):
    mask = File(
        exists=True,
        mandatory=True,
        desc="A binary brain mask.",
    )


class DenoiseNifti(NilearnBaseInterface, SimpleInterface):
    """Denoise a NIfTI BOLD file with Nilearn."""

    input_spec = _DenoiseNiftiInputSpec
    output_spec = _DenoiseImageOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.bandpass_filter:
            low_pass, high_pass = None, None
        else:
            low_pass, high_pass = self.inputs.low_pass, self.inputs.high_pass

        # Use a NiftiMasker instead of apply_mask to retain TR in the image header.
        # Note that this doesn't use any of the masker's denoising capabilities.
        masker = maskers.NiftiMasker(
            mask_img=self.inputs.mask,
            runs=None,
            smoothing_fwhm=None,
            standardize=False,
            standardize_confounds=False,  # non-default
            detrend=False,
            high_variance_confounds=False,
            low_pass=None,
            high_pass=None,
            t_r=None,
            target_affine=None,
            target_shape=None,
        )
        preprocessed_bold_arr = masker.fit_transform(self.inputs.preprocessed_bold)

        (
            uncensored_denoised_bold,
            interpolated_filtered_bold,
        ) = denoise_with_nilearn(
            preprocessed_bold=preprocessed_bold_arr,
            confounds_file=self.inputs.confounds_file,
            temporal_mask=self.inputs.temporal_mask,
            low_pass=low_pass,
            high_pass=high_pass,
            filter_order=self.inputs.filter_order,
            TR=self.inputs.TR,
        )

        self._results["uncensored_denoised_bold"] = os.path.join(
            runtime.cwd,
            "uncensored_denoised.nii.gz",
        )
        uncensored_denoised_img = masker.inverse_transform(uncensored_denoised_bold)
        uncensored_denoised_img.to_filename(self._results["uncensored_denoised_bold"])

        self._results["interpolated_filtered_bold"] = os.path.join(
            runtime.cwd,
            "filtered_denoised.nii.gz",
        )
        filtered_denoised_img = masker.inverse_transform(interpolated_filtered_bold)
        filtered_denoised_img.to_filename(self._results["interpolated_filtered_bold"])

        return runtime
