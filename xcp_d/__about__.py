# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Base module variables."""
from xcp_d._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__packagename__ = "xcp_d"
__copyright__ = "Copyright 2020, Penn LINC and Damien LAB"
__credits__ = (
    "Contributors: please check the ``.zenodo.json`` file at the top-level folder"
    "of the repository"
)
__url__ = "https://github.com/pennlinc/xcp_d"

DOWNLOAD_URL = f"https://github.com/pennlinc/{__packagename__}/archive/{__version__}.tar.gz"
