# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for concatenating scans across runs."""
import os
from contextlib import suppress

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.image import concat_imgs
from nipype import logging

LOGGER = logging.getLogger("nipype.interface")


def concatenate_tsvs(tsv_files, out_file):
    """Concatenate framewise displacement time series across files.

    This function doesn't return anything, but it writes out the ``out_file`` file.

    Parameters
    ----------
    tsv_files : :obj:`list` of :obj:`str`
        Paths to TSV files to concatenate.
    out_file : :obj:`str`
        Path to the file that will be written out.
    """
    try:
        # Assume file has no header first.
        # If it has a header with string column names, this will go the except clause.
        data = [np.loadtxt(tsv_file, delimiter="\t") for tsv_file in tsv_files]
        data = np.vstack(data)
        np.savetxt(out_file, data, fmt="%.5f", delimiter="\t")
    except ValueError:
        # Load file with header.
        data = [pd.read_table(tsv_file) for tsv_file in tsv_files]
        data = pd.concat(data, axis=0)
        data.to_csv(out_file, sep="\t", index=False)

    return out_file


def concatenate_niimgs(files, out_file):
    """Concatenate niimgs.

    Parameters
    ----------
    files : :obj:`list` of :obj:`str`
        List of BOLD files to concatenate over the time dimension.
    out_file : :obj:`str`
        The concatenated file to write out.
    """
    is_nifti = False
    with suppress(nb.filebasedimages.ImageFileError):
        is_nifti = isinstance(nb.load(files[0]), nb.Nifti1Image)

    if is_nifti:
        concat_preproc_img = concat_imgs(files)
        concat_preproc_img.to_filename(out_file)
    else:
        os.system(f"wb_command -cifti-merge {out_file} -cifti {' -cifti '.join(files)}")
