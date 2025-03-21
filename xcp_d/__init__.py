#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""XCP-D : A Robust Postprocessing Pipeline of fMRI data.

This pipeline is developed by Ted Satterthwaite's lab (https://pennlinc.io/).
"""

import warnings

from xcp_d.__about__ import __copyright__, __credits__, __packagename__, __version__

__all__ = [
    '__copyright__',
    '__credits__',
    '__packagename__',
    '__version__',
]

# cmp is not used by fmriprep, so ignore nipype-generated warnings
warnings.filterwarnings('ignore', r'cmp not installed')
warnings.filterwarnings('ignore', r'This has not been fully tested. Please report any failures.')
warnings.filterwarnings('ignore', r"can't resolve package from __spec__ or __package__")
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', ResourceWarning)
