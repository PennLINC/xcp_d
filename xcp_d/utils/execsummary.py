# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for generating the executive summary."""
import nibabel as nb
import nilearn.image as nlimage
import numpy as np
from nilearn.plotting import view_img
from skimage import measure


def make_brainsprite_html(mosaic_file, selected_png_files, image_type):
    """Create HTML file with brainsprite figure stored within."""
    import os

    from xcp_d.interfaces import constants
    from xcp_d.interfaces.layout_builder import ModalSlider

    viewer = image_type + "-viewer"
    spriteImg = image_type + "-spriteImg"

    spriteviewer = constants.SPRITE_VIEWER_HTML.format(
        viewer=viewer,
        spriteImg=spriteImg,
        mosaic_path=mosaic_file,
        width="100%",
    )
    spriteloader = constants.SPRITE_LOAD_SCRIPT.format(
        tx=image_type,
        viewer=viewer,
        spriteImg=spriteImg,
    )

    # Just a sanity check, since we happen to know how many to expect.
    if len(selected_png_files) != 9:
        # TODO: log WARNING
        print(f"Expected 9 {image_type} pngs but found {len(selected_png_files)}.")

    # Let's say that the PNGs are in the same folder as the HTML file for now.
    selected_png_files = [os.path.basename(f) for f in selected_png_files]

    # Make a modal container with a slider and add the pngs.
    pngs_slider = ModalSlider(f"{image_type}_modal", f"{image_type}pngs")
    pngs_slider.add_images(selected_png_files)

    # Add HTML for the bar with the brainsprite label and pngs button,
    # and for the brainsprite viewer.
    btn_label = f"View {image_type} pngs"
    html_code = constants.BRAINSPRITE_CODE.format(
        image_type=image_type,
        anat_pngs_button=pngs_slider.get_button(btn_label),
        anat_brainsprite=spriteviewer,
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
    import os

    import numpy as np
    from PIL import Image  # for BrainSprite

    mosaic_file = os.path.abspath("mosaic.jpg")
    files = png_files[::-1]  # we want last first, I guess?

    image_dim = 218
    images_per_side = int(np.sqrt(len(files)))
    n_images_to_plot = images_per_side**2
    if n_images_to_plot != len(files):
        print(f"{n_images_to_plot}/{len(files)} files will be plotted.")

    files = files[:n_images_to_plot]

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


def modify_brainsprite_scene_template(
    slice_number,
    anat_file,
    rh_pial_file,
    lh_pial_file,
    rh_white_file,
    lh_white_file,
    scene_template,
):
    """Create modified .scene text file to be used for creating brainsprite PNGs later."""
    import gzip
    import os

    paths = {
        "TX_IMG": anat_file,
        "RPIAL": rh_pial_file,
        "LPIAL": lh_pial_file,
        "RWHITE": rh_white_file,
        "LWHITE": lh_white_file,
    }

    out_file = os.path.abspath("modified_scene.scene")

    if scene_template.endswith(".gz"):
        with gzip.open(scene_template, mode="rt") as fo:
            data = fo.read()
    else:
        with open(scene_template, "r") as fo:
            data = fo.read()

    data = data.replace("XAXIS_COORDINATE", str(slice_number))

    for template, path in paths.items():
        filename = os.path.basename(path)

        # Replace templated pathnames and filenames in local copy.
        data = data.replace(f"{template}_PATH", path)
        data = data.replace(f"{template}_NAME", filename)

    with open(out_file, "w") as fo:
        fo.write(data)

    return out_file


def modify_pngs_scene_template(
    anat_file,
    rh_pial_file,
    lh_pial_file,
    rh_white_file,
    lh_white_file,
    scene_template,
):
    """Create modified .scene text file to be used for creating PNGs later."""
    import gzip
    import os

    paths = {
        "TX_IMG": anat_file,
        "RPIAL": rh_pial_file,
        "LPIAL": lh_pial_file,
        "RWHITE": rh_white_file,
        "LWHITE": lh_white_file,
    }

    out_file = os.path.abspath("modified_scene.scene")

    if scene_template.endswith(".gz"):
        with gzip.open(scene_template, mode="rt") as fo:
            data = fo.read()
    else:
        with open(scene_template, "r") as fo:
            data = fo.read()

    for template, path in paths.items():
        filename = os.path.basename(path)

        # Replace templated pathnames and filenames in local copy.
        data = data.replace(f"{template}_PATH", path)
        data = data.replace(f"{template}_NAME", filename)

    with open(out_file, "w") as fo:
        fo.write(data)

    return out_file


def get_n_frames(anat_file):
    """Infer the number of frames from an image."""
    import nibabel as nb

    img = nb.load(anat_file)

    # Get number of slices in x axis.
    n_slices = img.shape[0]

    frame_numbers = np.arange(1, n_slices + 1, dtype=int)
    ijk = np.ones((n_slices, 3), dtype=int)
    frame_numbers = list(range(1, n_slices + 1))
    ijk[:, 0] = frame_numbers
    xyz = nb.affines.apply_affine(img.affine, ijk)
    frame_numbers = xyz[:, 0].tolist()

    return frame_numbers


def get_png_image_names():
    """Get a list of scene names for which to produce PNGs."""
    image_descriptions = [
        "AxialInferiorTemporalCerebellum",
        "AxialBasalGangliaPutamen",
        "AxialSuperiorFrontal",
        "CoronalPosteriorParietalLingual",
        "CoronalCaudateAmygdala",
        "CoronalOrbitoFrontal",
        "SagittalInsulaFrontoTemporal",
        "SagittalCorpusCallosum",
        "SagittalInsulaTemporalHippocampalSulcus",
    ]

    scene_index = list(range(1, image_descriptions + 1))

    return scene_index, image_descriptions


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
