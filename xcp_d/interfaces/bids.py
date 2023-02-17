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
merged_entities = {**bids_config.entities, **deriv_config.entities, **xcp_d_spec["entities"][0]}
merged_file_patterns = sorted(
    list(
        set(
            bids_config.default_path_patterns
            + deriv_config.default_path_patterns
            + xcp_d_spec["default_path_patterns"]
        )
    )
)
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
    _file_patterns = merged_file_patterns
