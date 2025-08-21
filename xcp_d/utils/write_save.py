# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities to read and write nifiti and cifti data."""

import os

import nibabel as nb
import numpy as np
from nilearn import masking
from nipype import logging
from templateflow.api import get as get_template

from xcp_d.utils.doc import fill_doc
from xcp_d.utils.filemanip import split_filename

LOGGER = logging.getLogger('nipype.utils')


def read_ndata(datafile, maskfile=None):
    """Read nifti or cifti file as numpy array.

    Parameters
    ----------
    datafile : :obj:`str`
        nifti or cifti file
    maskfile : :obj:`str`
        Path to a binary mask.
        Unused for CIFTI data.

    Outputs
    -------
    data : (SxT) :obj:`numpy.ndarray`
        Vertices or voxels by timepoints.
    """
    # read cifti series
    cifti_extensions = ['.dtseries.nii', '.dlabel.nii', '.ptseries.nii', '.dscalar.nii']
    if any(datafile.endswith(ext) for ext in cifti_extensions):
        data = nb.load(datafile).get_fdata()

    # or nifti data, mask is required
    elif datafile.endswith('.nii.gz'):
        assert maskfile is not None, 'Input `maskfile` must be provided if `datafile` is a nifti.'
        data = masking.apply_mask(datafile, maskfile)

    else:
        raise ValueError(f'Unknown extension for {datafile}')

    # transpose from TxS to SxT
    data = data.T

    return data


def get_cifti_intents():
    """Return a dictionary of CIFTI extensions and associated intents.

    Copied from https://www.nitrc.org/projects/cifti/ PDF.
    """
    CIFTI_INTENTS = {
        '.dtseries.nii': 'ConnDenseSeries',
        '.dconn.nii': 'ConnDense',
        '.pconn.nii': 'ConnParcels',
        '.ptseries.nii': 'ConnParcelSries',
        '.dscalar.nii': 'ConnDenseScalar',
        '.dlabel.nii': 'ConnDenseLabel',
        '.pscalar.nii': 'ConnParcelScalr',
        '.pdconn.nii': 'ConnParcelDense',
        '.dpconn.nii': 'ConnDenseParcel',
        '.pconnseries.nii': 'ConnPPSr',
        '.pconnscalar.nii': 'ConnPPSc',
        '.dfan.nii': 'ConnDenseSeries',
        '.dfibersamp.nii': 'ConnUnknown',
        '.dfansamp.nii': 'ConnUnknown',
    }
    return CIFTI_INTENTS


@fill_doc
def write_ndata(data_matrix, template, filename, mask=None, TR=1):
    """Save numpy array to a nifti or cifti file.

    Parameters
    ----------
    data_matrix : (SxT) numpy.ndarray
        The array to save to a file.
    template : :obj:`str`
        Path to a template image, from which header and affine information will be used.
        If the output file is a CIFTI, this *must* be a .dtseries.nii CIFTI.
    filename : :obj:`str`
        Name of the output file to be written.
    mask : :obj:`str` or None, optional
        The path to a binary mask file.
        The mask is only used for nifti files- masking is not supported in ciftis.
        Default is None.
    %(TR)s

    Returns
    -------
    filename : :obj:`str`
        The name of the generated output file. Same as the "filename" input.

    Notes
    -----
    This function currently only works for NIfTIs and .dtseries.nii and .dscalar.nii CIFTIs.
    """
    assert data_matrix.ndim in (1, 2), f'Input data must be a 1-2D array, not {data_matrix.ndim}.'
    assert os.path.isfile(template)

    cifti_intents = get_cifti_intents()

    _, _, template_extension = split_filename(template)
    if template_extension in cifti_intents.keys():
        file_format = 'cifti'
    elif template.endswith('.nii.gz'):
        file_format = 'nifti'
        assert mask is not None, 'A binary mask must be provided for nifti inputs.'
        assert os.path.isfile(mask), f'The mask file does not exist: {mask}'
    else:
        raise ValueError(f'Unknown extension for {template}')

    # transpose from SxT to TxS
    data_matrix = data_matrix.T

    if file_format == 'cifti':
        # write cifti series
        template_img = nb.load(template)

        if data_matrix.ndim == 1:
            LOGGER.warning('1D data matrix provided. Adding singleton dimension.')
            data_matrix = data_matrix[None, :]

        n_volumes = data_matrix.shape[0]
        _, _, out_extension = split_filename(filename)

        if filename.endswith(('.dscalar.nii', '.pscalar.nii')):
            # Dense scalar files have (ScalarAxis, BrainModelAxis)
            # Parcellated scalar files have (ScalarAxis, ParcelsAxis)
            scalar_names = [f'#{i + 1}' for i in range(n_volumes)]
            ax_0 = nb.cifti2.cifti2_axes.ScalarAxis(name=scalar_names)
            ax_1 = template_img.header.get_axis(1)
            new_header = nb.Cifti2Header.from_axes((ax_0, ax_1))
            img = nb.Cifti2Image(data_matrix, new_header)

        elif filename.endswith(('.dtseries.nii', '.ptseries.nii')):
            # Dense series files have (SeriesAxis, BrainModelAxis)
            # Parcellated series files have (SeriesAxis, ParcelsAxis)
            if n_volumes == template_img.shape[0]:
                # same number of volumes in data as original image,
                # so we can just use the original axis
                img = nb.Cifti2Image(
                    dataobj=data_matrix,
                    header=template_img.header,
                    file_map=template_img.file_map,
                    nifti_header=template_img.nifti_header,
                )
            else:
                ax_1 = template_img.header.get_axis(1)

                # different number of volumes in data from original image,
                # so the time axis must be constructed manually based on its new length
                ax_0 = nb.cifti2.SeriesAxis(start=0, step=TR, size=n_volumes)

                # create new header and cifti object
                new_header = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
                img = nb.Cifti2Image(data_matrix, new_header)

        else:
            raise ValueError(f"Unsupported CIFTI extension '{out_extension}'")

        # Modify the intent code
        target_intent = cifti_intents[out_extension]
        img.nifti_header.set_intent(target_intent)

    else:
        # write nifti series
        img = masking.unmask(data_matrix.astype(np.float32), mask)
        # we'll override the default TR (1) in the header
        pixdim = list(img.header.get_zooms())
        pixdim[3] = TR
        img.header.set_zooms(pixdim)

    img.to_filename(filename)

    return filename


def write_gii(datat, template, filename, hemi):
    """Use nibabel to write surface file.

    Parameters
    ----------
    datatt : numpy.ndarray
        vector
    template : :obj:`str`
        real file loaded with nibabel to get header and filemap
    filename : :obj:`str`
        name of the output

    Returns
    -------
    filename
    """
    datax = np.array(datat, dtype='float32')
    template = str(
        get_template(
            'fsLR',
            hemi=hemi,
            suffix='midthickness',
            density='32k',
            desc='vaavg',
            raise_empty=True,
        )
    )
    template = nb.load(template)
    dataimg = nb.gifti.GiftiImage(
        header=template.header, file_map=template.file_map, extra=template.extra
    )
    dataimg = nb.gifti.GiftiImage(
        header=template.header,
        file_map=template.file_map,
        extra=template.extra,
        meta=template.meta,
    )
    d_timepoint = nb.gifti.GiftiDataArray(data=datax, intent='NIFTI_INTENT_NORMAL')
    dataimg.add_gifti_data_array(d_timepoint)
    dataimg.to_filename(filename)
    return filename


def read_gii(surf_gii):
    """Use nibabel to read surface file."""
    bold_data = nb.load(surf_gii)  # load the gifti
    gifti_data = bold_data.agg_data()  # aggregate the data
    if not hasattr(gifti_data, '__shape__'):  # if it doesn't have 'shape', reshape
        gifti_data = np.zeros((len(bold_data.darrays[0].data), len(bold_data.darrays)))
        for arr in range(len(bold_data.darrays)):
            gifti_data[:, arr] = bold_data.darrays[arr].data
    return gifti_data
