"""Miscellaneous utility interfaces."""
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
    traits_extension,
)

from xcp_d.utils.modified_data import downcast_to_32

LOGGER = logging.getLogger("nipype.interface")


class _ConvertTo32InputSpec(BaseInterfaceInputSpec):
    bold_file = traits.Either(
        None,
        File(exists=True),
        desc="BOLD file",
        mandatory=False,
        usedefault=True,
    )
    boldref = traits.Either(
        None,
        File(exists=True),
        desc="BOLD reference file",
        mandatory=False,
        usedefault=True,
    )
    bold_mask = traits.Either(
        None,
        File(exists=True),
        desc="BOLD mask file",
        mandatory=False,
        usedefault=True,
    )
    t1w = traits.Either(
        None,
        File(exists=True),
        desc="T1-weighted anatomical file",
        mandatory=False,
        usedefault=True,
    )
    t2w = traits.Either(
        None,
        File(exists=True),
        desc="T2-weighted anatomical file",
        mandatory=False,
        usedefault=True,
    )
    anat_dseg = traits.Either(
        None,
        File(exists=True),
        desc="T1-space segmentation file",
        mandatory=False,
        usedefault=True,
    )


class _ConvertTo32OutputSpec(TraitedSpec):
    bold_file = traits.Either(
        None,
        File(exists=True),
        desc="BOLD file",
        mandatory=False,
    )
    boldref = traits.Either(
        None,
        File(exists=True),
        desc="BOLD reference file",
        mandatory=False,
    )
    bold_mask = traits.Either(
        None,
        File(exists=True),
        desc="BOLD mask file",
        mandatory=False,
    )
    t1w = traits.Either(
        None,
        File(exists=True),
        desc="T1-weighted anatomical file",
        mandatory=False,
    )
    t2w = traits.Either(
        None,
        File(exists=True),
        desc="T2-weighted anatomical file",
        mandatory=False,
    )
    anat_dseg = traits.Either(
        None,
        File(exists=True),
        desc="T1-space segmentation file",
        mandatory=False,
    )


class ConvertTo32(SimpleInterface):
    """Downcast files from >32-bit to 32-bit if necessary."""

    input_spec = _ConvertTo32InputSpec
    output_spec = _ConvertTo32OutputSpec

    def _run_interface(self, runtime):
        self._results["bold_file"] = downcast_to_32(self.inputs.bold_file)
        self._results["boldref"] = downcast_to_32(self.inputs.boldref)
        self._results["bold_mask"] = downcast_to_32(self.inputs.bold_mask)
        self._results["t1w"] = downcast_to_32(self.inputs.t1w)
        self._results["t2w"] = downcast_to_32(self.inputs.t2w)
        self._results["anat_dseg"] = downcast_to_32(self.inputs.anat_dseg)

        return runtime


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
