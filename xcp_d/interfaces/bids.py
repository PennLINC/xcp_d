"""Nipype interfaces for working with BIDS data."""
import re
from collections import defaultdict
from json import loads
from pathlib import Path

from nipype import logging
from niworkflows.interfaces.bids import DerivativesDataSink as BaseDerivativesDataSink
from pkg_resources import resource_filename as _pkgres
from templateflow.api import templates as _get_template_list

regz = re.compile(r"\.gz$")
_pybids_spec = loads(Path(_pkgres("xcp_d", "data/nipreps.json")).read_text())
BIDS_DERIV_ENTITIES = frozenset({e["name"] for e in _pybids_spec["entities"]})
BIDS_DERIV_PATTERNS = tuple(_pybids_spec["default_path_patterns"])
STANDARD_SPACES = _get_template_list()
LOGGER = logging.getLogger("nipype.interface")


def _none():
    return None


# Automatically coerce certain suffixes (DerivativesDataSink)
DEFAULT_DTYPES = defaultdict(
    _none,
    (
        ("mask", "uint8"),
        ("dseg", "int16"),
        ("probseg", "float32"),
        ("boldref", "source"),
    ),
)


class DerivativesDataSink(BaseDerivativesDataSink):
    """An updated data-sink for xcp_d derivatives."""

    out_path_base = "xcp_d"
    BIDS_DERIV_ENTITIES = BIDS_DERIV_ENTITIES
    BIDS_DERIV_PATTERNS = BIDS_DERIV_PATTERNS
    DEFAULT_DTYPES = DEFAULT_DTYPES
    STANDARD_SPACES = STANDARD_SPACES
    regz = regz
    _allowed_entities = set(BIDS_DERIV_ENTITIES)
