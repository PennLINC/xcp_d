# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for generating the executive summary."""
import gzip
import os
from re import split

import nibabel as nb
import nilearn.image as nlimage
import numpy as np
from nilearn.plotting import view_img
from PIL import Image  # for BrainSprite
from skimage import measure

from xcp_d.interfaces.workbench import ShowScene


def natural_sort(list_):
    """Need this function so frames sort in correct order."""

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in split("([0-9]+)", key)]

    return sorted(list_, key=alphanum_key)


def make_brainsprite_html(mosaic_file, selected_png_files, input_type):
    from xcp_d.interfaces import constants
    from xcp_d.interfaces.layout_builder import ModalSlider

    viewer = input_type + "-viewer"
    spriteImg = input_type + "-spriteImg"

    spriteviewer = constants.SPRITE_VIEWER_HTML.format(
        viewer=viewer,
        spriteImg=spriteImg,
        mosaic_path=mosaic_file,
        width="100%",
    )
    spriteloader = constants.SPRITE_LOAD_SCRIPT.format(
        tx=input_type,
        viewer=viewer,
        spriteImg=spriteImg,
    )

    # Just a sanity check, since we happen to know how many to expect.
    if len(selected_png_files) != 9:
        # TODO: log WARNING
        print(f"Expected 9 {input_type} pngs but found {len(selected_png_files)}.")

    # Make a modal container with a slider and add the pngs.
    pngs_slider = ModalSlider(f"{input_type}_modal", f"{input_type}pngs")
    pngs_slider.add_images(selected_png_files)

    # Add HTML for the bar with the brainsprite label and pngs button,
    # and for the brainsprite viewer.
    btn_label = f"View {input_type} pngs"
    html_code = constants.T1X_SECTION.format(
        tx=input_type,
        t1_pngs_button=pngs_slider.get_button(btn_label),
        t1wbrainplot=spriteviewer,
    )

    html_code += pngs_slider.get_container()
    html_code += spriteloader
    html_code += pngs_slider.get_scripts()

    out_file = os.path.abspath("brainsprite.html")
    with open(out_file, "w") as fo:
        fo.write(html_code)

    return out_file


def make_mosaic(png_files):
    """Take path to .png anatomical slices, create a mosaic, and save to file.

    The mosaic will be usable in a BrainSprite viewer.
    """
    mosaic_file = os.path.abspath("mosaic.jpg")
    files = sorted(png_files)  # just in case they get shuffled
    files = files[::-1]  # we want last first, I guess?

    image_dim = 218
    images_per_side = int(np.sqrt(len(files)))
    square_dim = image_dim * images_per_side
    result = Image.new("RGB", (square_dim, square_dim))

    for index, file_ in enumerate(files):
        # Get relative path to file, from user's home folder
        path = os.path.expanduser(file_)
        with Image.open(path) as img:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.thumbnail((image_dim, image_dim), resample=Image.ANTIALIAS)

            x = index % images_per_side * image_dim
            y = index // images_per_side * image_dim
            w, h = img.size
            result.paste(img, (x, y, x + w, y + h))

    result.save(mosaic_file, "JPEG", quality=95)
    return mosaic_file


def build_scene_from_brainsprite_template(
    tx_img,
    rh_pial_file,
    lh_pial_file,
    rh_white_file,
    lh_white_file,
    scene_template,
):
    """Create modified .scene text file to be used for creating PNGs later."""
    paths = {
        "TX_IMG": tx_img,
        "R_PIAL": rh_pial_file,
        "L_PIAL": lh_pial_file,
        "R_WHITE": rh_white_file,
        "L_WHITE": lh_white_file,
    }

    out_file = os.path.abspath("modified_scene.scene")

    if scene_template.endswith(".gz"):
        with gzip.open(scene_template, mode="rt") as fo:
            data = fo.read()
    else:
        with open(scene_template, "r") as fo:
            data = fo.read()

    for template, path in paths.items():
        # Replace templated pathnames and filenames in local copy.
        data = data.replace(f"{template}_NAME_and_PATH", path)
        filename = os.path.basename(path)
        data = data.replace(f"{template}_NAME", filename)

    with open(out_file, "w") as fo:
        fo.write(data)

    return out_file


def create_image_from_brainsprite_scene(scene_file, frame_number):
    """Create a single PNG for a brainsprite.

    Parameters
    ----------
    output_dir
    scene_file
    frame_number : int
        Starts with 1.
    """
    out_file = os.path.abspath(f"frame_{frame_number:06g}.png")

    show_scene = ShowScene(
        scene_file=scene_file,
        scene_name_or_number=frame_number,
        out_file=out_file,
        image_width=900,
        image_height=800,
    )
    _ = show_scene.run()

    return out_file


def get_n_frames(scene_file):
    with open(scene_file, "r") as fo:
        data = fo.read()

    total_frames = data.count("SceneInfo Index=")
    frame_numbers = list(range(1, total_frames + 1))
    return frame_numbers


def create_images_from_brainsprite_scene(output_dir, scene_file):
    """Create a series of PNG files that will later be used in a brainsprite."""
    with open(scene_file, "r") as fo:
        data = fo.read()

    total_frames = data.count("SceneInfo Index=")

    for i_frame in range(total_frames):
        _ = create_image_from_brainsprite_scene(output_dir, scene_file, i_frame + 1)

    return output_dir


def generate_brain_sprite(template_image, stat_map, out_file):
    """Generate a brainsprite HTML file."""
    html_view = view_img(
        stat_map_img=stat_map,
        cmap="hsv",
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
        con = measure.find_contours(datax[:, :, i], fully_connected="low")
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
