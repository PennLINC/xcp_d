"""Miscellaneous utility interfaces."""
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
