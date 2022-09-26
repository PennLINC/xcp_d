"""Interfaces for Nilearn code."""
import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    SimpleInterface,
    TraitedSpec,
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
        desc="The name of the concatenated file to write out. concate_4d.nii.gz by default.",
    )


class _MergeOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc="tsv file containing the computed "
        "signals, with as many columns as there are labels and as "
        "many rows as there are timepoints in in_file, plus a "
        "header row with values from class_labels",
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
