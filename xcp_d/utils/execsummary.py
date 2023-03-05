# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for generating the executive summary."""
from nipype import logging

LOGGER = logging.getLogger("nipype.utils")


def make_mosaic(png_files):
    """Take path to .png anatomical slices, create a mosaic, and save to file.

    The mosaic will be usable in a BrainSprite viewer.

    NOTE: This is a Node function.
    """
    import os

    import numpy as np
    from PIL import Image  # for BrainSprite

    mosaic_file = os.path.abspath("mosaic.png")
    files = png_files[::-1]  # we want last first, I guess?

    IMAGE_DIM = 218
    images_per_side = int(np.ceil(np.sqrt(len(files))))
    square_dim = IMAGE_DIM * images_per_side
    result = Image.new("RGB", (square_dim, square_dim), color=1)

    for index, file_ in enumerate(files):
        # Get relative path to file, from user's home folder
        path = os.path.expanduser(file_)
        with Image.open(path) as img:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.thumbnail((IMAGE_DIM, IMAGE_DIM), resample=Image.ANTIALIAS)

            x = index % images_per_side * IMAGE_DIM
            y = index // images_per_side * IMAGE_DIM
            w, h = img.size
            result.paste(img, (x, y, x + w, y + h))

    result.save(mosaic_file, "PNG", quality=95)
    return mosaic_file


def modify_brainsprite_scene_template(
    slice_number,
    anat_file,
    rh_pial_surf,
    lh_pial_surf,
    rh_wm_surf,
    lh_wm_surf,
    scene_template,
):
    """Create modified .scene text file to be used for creating brainsprite PNGs later.

    NOTE: This is a Node function.
    """
    import gzip
    import os

    paths = {
        "TX_IMG": anat_file,
        "RPIAL": rh_pial_surf,
        "LPIAL": lh_pial_surf,
        "RWHITE": rh_wm_surf,
        "LWHITE": lh_wm_surf,
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
    rh_pial_surf,
    lh_pial_surf,
    rh_wm_surf,
    lh_wm_surf,
    scene_template,
):
    """Create modified .scene text file to be used for creating PNGs later.

    NOTE: This is a Node function.
    """
    import gzip
    import os

    paths = {
        "TX_IMG": anat_file,
        "RPIAL": rh_pial_surf,
        "LPIAL": lh_pial_surf,
        "RWHITE": rh_wm_surf,
        "LWHITE": lh_wm_surf,
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
    """Infer the number of slices in x axis from an image.

    NOTE: This is a Node function.

    Parameters
    ----------
    anat_file

    Returns
    -------
    frame_numbers
    """
    import nibabel as nb
    import numpy as np

    img = nb.load(anat_file)

    # Get number of slices in x axis.
    n_slices = img.shape[0]

    frame_numbers = np.arange(1, n_slices + 1, dtype=int)
    ijk = np.ones((2, 3), dtype=int)
    ijk[:, 0] = [0, n_slices]
    xyz = nb.affines.apply_affine(img.affine, ijk)
    first_slice, last_slice = xyz[:, 0].astype(int)
    frame_numbers = list(np.arange(first_slice, last_slice + 1, dtype=int))

    return frame_numbers


def get_png_image_names():
    """Get a list of scene names for which to produce PNGs.

    NOTE: This is a Node function.
    """
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

    scene_index = list(range(1, len(image_descriptions) + 1))

    return scene_index, image_descriptions
