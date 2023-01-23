"""Adapted interfaces from Niworkflows."""
from json import loads
from pathlib import Path

from nipype import logging
from niworkflows.interfaces.base import DerivativesDataSink as BaseDerivativesDataSink
from pkg_resources import resource_filename as pkgrf

# NOTE: Modified for xcpd's purposes
_pybids_spec = loads(Path(pkgrf("xcp_d", "data/nipreps.json")).read_text())
BIDS_DERIV_ENTITIES = frozenset({e["name"] for e in _pybids_spec["entities"]})
BIDS_DERIV_PATTERNS = tuple(_pybids_spec["default_path_patterns"])
LOGGER = logging.getLogger("nipype.interface")


class DerivativesDataSink(BaseDerivativesDataSink):
    """Store derivative files.

    A child class of the niworkflows DerivativesDataSink, using xcp_d's configuration files.
    """

    out_path_base = "xcp_d"
    _allowed_entities = set(BIDS_DERIV_ENTITIES)
    _config_entities = BIDS_DERIV_ENTITIES
    _file_patterns = BIDS_DERIV_PATTERNS
