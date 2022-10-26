# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for generating the executive summary."""
import nibabel as nb
import nilearn.image as nlimage
import numpy as np
from nilearn.plotting import view_img
from skimage import measure


def generate_brain_sprite(template_image, stat_map, out_file):
    """Generate a brainsprite HTML file."""
    html_view = view_img(
        stat_map_img=stat_map,
        cmap='hsv',
        symmetric_cmap=False,
        black_bg=True,
        vmin=-1,
        vmax=3,
        colorbar=False,
        bg_img=template_image,
    )

    html_view.save_as_html(out_file)

    return out_file


def ribbon_to_statmap(ribbon, outfile):
    """Convert a ribbon to a volumetric statistical map."""
    # chek if the data is ribbon or seg_data files

    ngbdata = nb.load(ribbon)

    if ngbdata.get_fdata().max() > 5:  # that is ribbon
        contour_data = ngbdata.get_fdata() % 39
        white = nlimage.new_img_like(ngbdata, contour_data == 2)
        pial = nlimage.new_img_like(ngbdata, contour_data >= 2)
    else:
        contour_data = ngbdata.get_fdata()
        white = nlimage.new_img_like(ngbdata, contour_data == 2)
        pial = nlimage.new_img_like(ngbdata, contour_data == 1)

    datapial = _get_contour(pial.get_fdata())
    datawhite = _get_contour(white.get_fdata())

    datax = 2 * datapial + datawhite

    # save the output
    ngbdatax = nb.Nifti1Image(datax, ngbdata.affine, ngbdata.header)
    ngbdatax.to_filename(outfile)

    return outfile


def _get_contour(datax):
    """Get contour in each plane."""
    dims = datax.shape

    contour = np.zeros_like(datax)

    # get y-z plane
    for i in range(dims[2]):
        con = measure.find_contours(datax[:, :, i], fully_connected='low')
        conx = np.zeros_like(datax[:, :, i])
        for cx in con:
            conx[np.int64(cx[:, 0]), np.int64(cx[:, 1])] = 1
        contour[:, :, i] = conx

        # for xz plane
        # for i in range(dims[1]):
        # con = measure.find_contours(datax[:,i,:],fully_connected='low')
        # conx =np.zeros_like(datax[:,i,:])
        # for cx in con:
        # conx[np.int64(cx[:, 0]), np.int64(cx[:, 1])]=1 # +0.5 to avoid the 0.5 offset
        # contour[:,i,:]= conx

    # for yz plane
    # for i in range(dims[2]):
    # con = measure.find_contours(datax[:,:,i],fully_connected='low')
    # conx =np.zeros_like(datax[:,:,i])
    # for cx in con:
    # conx[np.int64(cx[:, 0]), np.int64(cx[:, 1])]=1
    # contour[:,:,i]= conx

    return contour
