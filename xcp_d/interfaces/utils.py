"""Miscellaneous utility interfaces."""
import os
import shutil

from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
    traits_extension,
)

from xcp_d.utils.filemanip import split_filename

LOGGER = logging.getLogger("nipype.interface")


class _FilterUndefinedInputSpec(BaseInterfaceInputSpec):
    inlist = InputMultiObject(
        traits.Str,
        mandatory=True,
        desc="List of objects to filter.",
    )


class _FilterUndefinedOutputSpec(TraitedSpec):
    outlist = OutputMultiObject(
        traits.Str,
        desc="Filtered list of objects.",
    )


class FilterUndefined(SimpleInterface):
    """Extract timeseries and compute connectivity matrices."""

    input_spec = _FilterUndefinedInputSpec
    output_spec = _FilterUndefinedOutputSpec

    def _run_interface(self, runtime):
        inlist = self.inputs.inlist
        outlist = []
        for item in inlist:
            if item is not None and traits_extension.isdefined(item):
                outlist.append(item)
        self._results["outlist"] = outlist
        return runtime


class _CleanExtensionInputSpec(BaseInterfaceInputSpec):
    in_file = traits.File(exists=True, mandatory=True, desc="Input file")


class _CleanExtensionOutputSpec(TraitedSpec):
    out_file = traits.File(exists=True, desc="Renamed file")


class CleanExtension(SimpleInterface):
    """Clean out any bad extension additions."""

    input_spec = _CleanExtensionInputSpec
    output_spec = _CleanExtensionOutputSpec

    def _run_interface(self, runtime):
        LOGGER.warning(self.inputs.in_file)
        extension = split_filename(self.inputs.in_file)[2]
        out_file = os.path.abspath(f"cleaned_file{extension}")
        shutil.copyfile(self.inputs.in_file, out_file)

        self._results["out_file"] = out_file
        return runtime
