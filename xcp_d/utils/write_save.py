# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities to read and write nifiti and cifti data."""
import os
import subprocess

import nibabel as nb
import numpy as np
from nilearn import masking
from templateflow.api import get as get_template


def read_ndata(datafile, maskfile=None, scale=0):
    """Read nifti or cifti file.

    Parameters
    ----------
    datafile : str
        nifti or cifti file
    maskfile
    scale : ?

    Outputs
    -------
    data : (TxS) :obj:`numpy.ndarray`
        Vertices or voxels by timepoints.
    """
    # read cifti series
    if datafile.endswith(".dtseries.nii"):
        data = nb.load(datafile).get_fdata()

    # or nifti data, mask is required
    elif datafile.endswith(".nii.gz"):
        data = masking.apply_mask(datafile, maskfile)

    else:
        raise ValueError(f"Unknown extension for {datafile}")

    # transpose from TxS to SxT
    data = data.T

    if scale > 0:
        data = scalex(data, -scale, scale)

    return data


def write_ndata(data_matrix, template, filename, mask=None, TR=1, scale=0):
    """Save numpy array to a nifti or cifti file.

    Parameters
    ----------
    data matrix : (SxT) numpy.ndarray
        The array to save to a file.
    template : str
        Path to a template image, from which header and affine information will be used.
    filename : str
        Name of the output file to be written.
    mask : str or None, optional
        The path to a binary mask file.
        The mask is only used for nifti files- masking is not supported in ciftis.
        Default is None.
    TR : float, optional
    scale : float, optional

    Returns
    -------
    filename : str
        The name of the generated output file. Same as the "filename" input.
    """
    assert data_matrix.ndim == 2, f"Input data must be a 2D array, not {data_matrix.ndim}."
    assert os.path.isfile(template)

    if template.endswith(".dtseries.nii"):
        file_format = "cifti"
    elif template.endswith(".nii.gz"):
        file_format = "nifti"
        assert mask is not None, "A binary mask must be provided for nifti inputs."
        assert os.path.isfile(mask), f"The mask file does not exist: {mask}"
    else:
        raise ValueError(f"Unknown extension for {template}")

    if scale > 0:
        data_matrix = scalex(data_matrix, -scale, scale)

    # transpose from SxT to TxS
    data_matrix = data_matrix.T

    if file_format == "cifti":
        # write cifti series
        template_img = nb.load(template)

        if data_matrix.shape[1] == template_img.shape[0]:
            # same number of volumes in data as original image
            img = nb.Cifti2Image(
                dataobj=data_matrix,
                header=template_img.header,
                file_map=template_img.file_map,
                nifti_header=template_img.nifti_header,
            )

        else:
            # different number of volumes in data from original image
            # the time axis must be constructed manually based on its new length
            ax_0 = nb.cifti2.SeriesAxis(start=0, step=TR, size=data_matrix.shape[0])
            ax_1 = template_img.header.get_axis(1)

            # create new header and cifti object
            new_header = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
            img = nb.Cifti2Image(data_matrix, new_header)

        # NOTE: Intent is necessary for plotting functions,
        # but I don't know if we should assume that any saved CIFTI is a ConnDenseSeries.
        img.nifti_header.set_intent("ConnDenseSeries")

    else:
        # write nifti series
        img = masking.unmask(data_matrix, mask)
        # we'll override the default TR (1) in the header
        pixdim = list(img.header.get_zooms())
        pixdim[3] = TR
        img.header.set_zooms(pixdim)

    img.to_filename(filename)

    return filename


def edit_ciftinifti(in_file, out_file, datax):
    """
    this function create a fake nifti file from cifti
    in_file:
       cifti file. .dstreries etc
    out_file:
       output fake nifti file
    datax: numpy darray
      data matrix with vertices by timepoints dimension
    """
    thdata = nb.load(in_file)
    dataxx = thdata.get_fdata()
    dd = dataxx[:, :, :, 0:datax.shape[1]]
    dataimg = nb.Nifti1Image(dataobj=dd,
                             affine=thdata.affine,
                             header=thdata.header)
    dataimg.to_filename(out_file)
    return out_file


def run_shell(cmd, env=os.environ):
    """
    utilities to run shell in python
    cmd:
     shell command that wanted to be run
    """
    if type(cmd) is list:
        cmd = ' '.join(cmd)

    call_command = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        shell=True,
    )
    output, error = call_command.communicate("Hello from the other side!")
    call_command.wait()

    return output, error


def write_gii(datat, template, filename, hemi):
    '''
    datatt : vector
    template: real file loaded with nibabel to get header and filemap
    filename ; name of the output
    '''
    datax = np.array(datat, dtype='float32')
    template = str(
        get_template("fsLR", hemi=hemi, suffix='midthickness', density='32k', desc='vaavg'))
    template = nb.load(template)
    dataimg = nb.gifti.GiftiImage(header=template.header,
                                  file_map=template.file_map,
                                  extra=template.extra)
    dataimg = nb.gifti.GiftiImage(header=template.header,
                                  file_map=template.file_map,
                                  extra=template.extra,
                                  meta=template.meta)
    d_timepoint = nb.gifti.GiftiDataArray(data=datax,
                                          intent='NIFTI_INTENT_NORMAL')
    dataimg.add_gifti_data_array(d_timepoint)
    dataimg.to_filename(filename)
    return filename


def read_gii(surf_gii):
    """
    Using nibabel to read surface file
    """
    bold_data = nb.load(surf_gii)  # load the gifti
    gifti_data = bold_data.agg_data()  # aggregate the data
    if not hasattr(gifti_data, '__shape__'):  # if it doesn't have 'shape', reshape
        gifti_data = np.zeros((len(bold_data.darrays[0].data), len(bold_data.darrays)))
        for arr in range(len(bold_data.darrays)):
            gifti_data[:, arr] = bold_data.darrays[arr].data
    return gifti_data


def despikedatacifti(cifti, TR, basedir):
    """ despiking cifti """
    fake_cifti1 = str(basedir + '/fake_niftix.nii.gz')
    fake_cifti1_depike = str(basedir + '/fake_niftix_depike.nii.gz')
    cifti_despike = str(basedir + '/despike_nifti2cifti.dtseries.nii')
    run_shell([
        'OMP_NUM_THREADS=2 wb_command -cifti-convert -to-nifti ', cifti,
        fake_cifti1
    ])
    run_shell(
        ['3dDespike -nomask -NEW -prefix', fake_cifti1_depike, fake_cifti1])
    run_shell([
        'OMP_NUM_THREADS=2 wb_command  -cifti-convert -from-nifti  ',
        fake_cifti1_depike, cifti, cifti_despike, '-reset-timepoints',
        str(TR),
        str(0)
    ])
    return cifti_despike


def scalex(X, x_min, x_max):
    """Scale data to between minimum and maximum values."""
    nom = (X - X.min()) * (x_max - x_min)
    denom = X.max() - X.min()

    if denom == 0:
        denom = 1

    return x_min + (nom / denom)
