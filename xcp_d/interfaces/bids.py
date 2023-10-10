"""Adapted interfaces from Niworkflows."""
from json import loads
from pathlib import Path

from bids.layout import Config
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    DynamicTraitedSpec,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.io import IOBase, add_traits
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink
from pkg_resources import resource_filename as pkgrf

from xcp_d.utils.utils import _listify

# NOTE: Modified for xcpd's purposes
xcp_d_spec = loads(Path(pkgrf("xcp_d", "data/xcp_d_bids_config.json")).read_text())
bids_config = Config.load("bids")
deriv_config = Config.load("derivatives")

xcp_d_entities = {v["name"]: v["pattern"] for v in xcp_d_spec["entities"]}
merged_entities = {**bids_config.entities, **deriv_config.entities}
merged_entities = {k: v.pattern for k, v in merged_entities.items()}
merged_entities = {**merged_entities, **xcp_d_entities}
merged_entities = [{"name": k, "pattern": v} for k, v in merged_entities.items()]
config_entities = frozenset({e["name"] for e in merged_entities})

LOGGER = logging.getLogger("nipype.interface")


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using xcp_d's configuration files.
    """

    out_path_base = "xcp_d"
    _allowed_entities = set(config_entities)
    _config_entities = config_entities
    _config_entities_dict = merged_entities
    _file_patterns = xcp_d_spec["default_path_patterns"]


class _InferBIDSURIsInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    dataset_path = traits.Str(desc="Base directory of dataset.")
    dataset_name = traits.Str(desc="Dataset name for BIDS URI.")


class _InferBIDSURIsOutputSpec(TraitedSpec):
    bids_uris = traits.List(desc="Merged output")


def _ravel(in_val):
    if not isinstance(in_val, list):
        return in_val

    flat_list = []
    for val in in_val:
        raveled_val = _ravel(val)
        if isinstance(raveled_val, list):
            flat_list.extend(raveled_val)
        else:
            flat_list.append(raveled_val)

    return flat_list


class InferBIDSURIs(IOBase):
    """Basic interface class to merge inputs into a single list and infer BIDS URIs."""

    input_spec = _InferBIDSURIsInputSpec
    output_spec = _InferBIDSURIsOutputSpec

    def __init__(self, numinputs=0, **inputs):
        super().__init__(**inputs)
        self._numinputs = numinputs
        if numinputs >= 1:
            input_names = [f"in{i + 1}" for i in range(numinputs)]
        else:
            input_names = []

        add_traits(self.inputs, input_names)

    def _getval(self, idx):
        return getattr(self.inputs, f"in{idx + 1}")

    def _list_outputs(self):
        outputs = self._outputs().get()

        if self._numinputs < 1:
            return outputs
        else:
            values = [
                self._getval(idx) for idx in range(self._numinputs) if isdefined(self._getval(idx))
            ]

        # Ensure all inputs are lists
        lists = [_listify(val) for val in values]
        # Flatten list of lists
        raw_paths = [item for sublist in lists for item in sublist]

        # Now convert the strings to BIDS URIs
        bids_uris = [
            f"bids:{self.inputs.dataset_name}:{str(Path(p).relative_to(self.inputs.dataset_path))}"
            for p in raw_paths
        ]

        outputs["bids_uris"] = bids_uris
        return outputs
