# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Miscellaneous file manipulation functions."""
import os.path as op

import numpy as np
from nipype import logging
from nipype.utils.misc import is_container

fmlogger = logging.getLogger("nipype.utils")

related_filetype_sets = [(".hdr", ".img", ".mat"), (".nii", ".mat"), (".BRIK", ".HEAD")]


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : :obj:`str`
        file or path name

    Returns
    -------
    pth : :obj:`str`
        base path from fname
    fname : :obj:`str`
        filename from fname, without extension
    ext : :obj:`str`
        file extension from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'
    """
    # TM 07152022 - edited to add cifti and workbench extensions
    special_extensions = [
        ".nii.gz",
        ".tar.gz",
        ".niml.dset",
        ".dconn.nii",
        ".dlabel.nii",
        ".dpconn.nii",
        ".dscalar.nii",
        ".dtseries.nii",
        ".fiberTEMP.nii",
        ".trajTEMP.wbsparse",
        ".pconn.nii",
        ".pdconn.nii",
        ".plabel.nii",
        ".pscalar.nii",
        ".ptseries.nii",
        ".sdseries.nii",
        ".label.gii",
        ".label.gii",
        ".func.gii",
        ".shape.gii",
        ".rgba.gii",
        ".surf.gii",
        ".dpconn.nii",
        ".dtraj.nii",
        ".pconnseries.nii",
        ".pconnscalar.nii",
        ".dfan.nii",
        ".dfibersamp.nii",
        ".dfansamp.nii",
    ]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext


def fname_presuffix(fname, prefix="", suffix="", newpath=None, use_ext=True):
    """Manipulate path and name of input filename.

    Parameters
    ----------
    fname : string
        A filename (may or may not include path)
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    str
        Absolute path of the modified filename

    Examples
    --------
    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    >>> from nipype.interfaces.base import Undefined
    >>> fname_presuffix(fname, 'pre', 'post', Undefined) == \
            fname_presuffix(fname, 'pre', 'post')
    True
    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ""

    # No need for isdefined: bool(Undefined) evaluates to False
    if newpath:
        pth = op.abspath(newpath)
    return op.join(pth, prefix + fname + suffix + ext)


def ensure_list(filename):
    """Return a list given either a string or a list."""
    if isinstance(filename, (str, bytes)):
        return [filename]
    elif isinstance(filename, (list, tuple, type(None), np.ndarray)):
        return filename
    elif is_container(filename):
        return [x for x in filename]
    else:
        return None
