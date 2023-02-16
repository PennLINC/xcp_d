"""Interfaces for Nilearn code."""
import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.nilearn import NilearnBaseInterface


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


class _MeanImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="An image to average over time.",
    )
    out_file = File(
        "out_img.nii.gz",
        usedefault=True,
        exists=False,
        desc="The name of the averaged file to write out. out_img.nii.gz by default.",
    )


class _MeanImageOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="Mathified output file.",
    )


class MeanImage(NilearnBaseInterface, SimpleInterface):
    """Get the mean over time of a 4D NIFTI image."""

    input_spec = _MeanImageInputSpec
    output_spec = _MeanImageOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import mean_img

        avg_img = mean_img(img=self.inputs.in_file)
        self._results["out_file"] = os.path.join(runtime.cwd, self.inputs.out_file)
        avg_img.to_filename(self._results["out_file"])

        return runtime
