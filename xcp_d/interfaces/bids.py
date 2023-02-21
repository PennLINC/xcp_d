"""Adapted interfaces from Niworkflows."""
from json import loads
from pathlib import Path

from bids.layout import Config
from nipype import logging
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink
from pkg_resources import resource_filename as pkgrf

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
