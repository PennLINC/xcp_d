# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for generating the executive summary."""

from nipype import logging
import numpy as np


LOGGER = logging.getLogger('nipype.utils')


def plot_gii(mesh, coord, color, slicer, view, max_distance=10.):
    _ax = slicer.axes[list(slicer.axes.keys())[0]]

    if view == 'x':
        plane_origin = [coord, 0, 0]
        plane_normal = [1, 0, 0]  # Normal for the sagittal plane
    elif view == 'y':
        plane_origin = [0, coord, 0]
        plane_normal = [0, 1, 0]  # Normal for the coronal plane
    else:
        plane_origin = [0, 0, coord]
        plane_normal = [0, 0, 1]  # Normal for the axial plane

    # Use trimesh to find the intersection of the mesh and the plane
    slice_section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if slice_section is None:
        return

    for disc in slice_section.discrete:
        temp = disc
        # Check that there aren't defects in the Line
        differences = np.abs(np.diff(temp, axis=0))
        if np.any(differences.max(axis=0) > max_distance):
            continue
        if view == 'x':
            _ax.ax.plot(temp[:, 1], temp[:, 2], color=color)
        elif view == 'y':
            _ax.ax.plot(temp[:, 0], temp[:, 2], color=color)
        else:
            _ax.ax.plot(temp[:, 0], temp[:, 1], color=color)


def get_mesh(filename):
    import nibabel as nb
    import trimesh

    img = nb.load(filename)
    vertices = img.darrays[0].data  # Vertex coordinates
    faces = img.darrays[1].data  # Triangle indices
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def plot_slice_for_brainsprite(nifti, lh_wm, lh_pial, rh_wm, rh_pial):
    import os

    import matplotlib.pyplot as plt
    import nibabel as nb
    import numpy as np
    from nilearn import plotting

    from xcp_d.utils.execsummary import get_mesh, plot_gii

    img = nb.load(nifti)
    affine = img.affine
    lh_wm = get_mesh(lh_wm)
    lh_pial = get_mesh(lh_pial)
    rh_wm = get_mesh(rh_wm)
    rh_pial = get_mesh(rh_pial)

    n_x = img.shape[0]
    filenames = []

    data = img.get_fdata()
    vmin = np.percentile(data, 2)
    vmax = np.percentile(data, 98)

    for i_slice in range(n_x):
        fig, ax = plt.subplots(figsize=(9, 7.5))

        # Get the appropriate coordinate
        # TODO: Shift so middle is center of image
        coord = nb.affines.apply_affine(affine, [i_slice, 0, 0])[0]

        # Display a sagittal slice (adjust 'display_mode' and 'cut_coords' as needed)
        slicer = plotting.plot_anat(
            img,
            display_mode='x',
            cut_coords=[coord],
            axes=ax,
            figure=fig,
            annotate=False,
            vmin=vmin,
            vmax=vmax,
        )

        # Load the surface mesh (GIFTI format)
        plot_gii(lh_pial, coord, 'darkred', slicer, 'x')
        plot_gii(rh_pial, coord, 'darkred', slicer, 'x')
        plot_gii(lh_wm, coord, 'black', slicer, 'x')
        plot_gii(rh_wm, coord, 'black', slicer, 'x')

        filename = os.path.abspath(f'test_{i_slice:03d}.png')
        fig.savefig(filename, bbox_inches='tight', facecolor='black')
        filenames.append(filename)
    return filenames


def make_mosaic(png_files):
    """Take path to .png anatomical slices, create a mosaic, and save to file.

    The mosaic will be usable in a BrainSprite viewer.

    NOTE: This is a Node function.
    """
    import os

    import numpy as np
    from PIL import Image  # for BrainSprite

    mosaic_file = os.path.abspath('mosaic.png')
    files = png_files[::-1]  # we want last first, I guess?

    IMAGE_DIM = 218
    images_per_side = int(np.ceil(np.sqrt(len(files))))
    square_dim = IMAGE_DIM * images_per_side
    result = Image.new('RGB', (square_dim, square_dim), color=1)

    for index, file_ in enumerate(files):
        # Get relative path to file, from user's home folder
        path = os.path.expanduser(file_)
        with Image.open(path) as img:
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img.thumbnail((IMAGE_DIM, IMAGE_DIM), resample=Image.Resampling.LANCZOS)

            x = index % images_per_side * IMAGE_DIM
            y = index // images_per_side * IMAGE_DIM
            w, h = img.size
            result.paste(img, (x, y, x + w, y + h))

    result.save(mosaic_file, 'PNG', quality=95)
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
        'TX_IMG': anat_file,
        'RPIAL': rh_pial_surf,
        'LPIAL': lh_pial_surf,
        'RWHITE': rh_wm_surf,
        'LWHITE': lh_wm_surf,
    }

    out_file = os.path.abspath('modified_scene.scene')

    if scene_template.endswith('.gz'):
        with gzip.open(scene_template, mode='rt') as fo:
            data = fo.read()
    else:
        with open(scene_template) as fo:
            data = fo.read()

    data = data.replace('XAXIS_COORDINATE', str(slice_number))

    for template, path in paths.items():
        filename = os.path.basename(path)

        # Replace templated pathnames and filenames in local copy.
        data = data.replace(f'{template}_PATH', path)
        data = data.replace(f'{template}_NAME', filename)

    with open(out_file, 'w') as fo:
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
        'TX_IMG': anat_file,
        'RPIAL': rh_pial_surf,
        'LPIAL': lh_pial_surf,
        'RWHITE': rh_wm_surf,
        'LWHITE': lh_wm_surf,
    }

    out_file = os.path.abspath('modified_scene.scene')

    if scene_template.endswith('.gz'):
        with gzip.open(scene_template, mode='rt') as fo:
            data = fo.read()
    else:
        with open(scene_template) as fo:
            data = fo.read()

    for template, path in paths.items():
        filename = os.path.basename(path)

        # Replace templated pathnames and filenames in local copy.
        data = data.replace(f'{template}_PATH', path)
        data = data.replace(f'{template}_NAME', filename)

    with open(out_file, 'w') as fo:
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
    first_slice, last_slice = sorted(xyz[:, 0].astype(int))
    frame_numbers = list(np.arange(first_slice, last_slice + 1, dtype=int))

    return frame_numbers


def get_png_image_names():
    """Get a list of scene names for which to produce PNGs.

    NOTE: This is a Node function.
    """
    image_descriptions = [
        'AxialInferiorTemporalCerebellum',
        'AxialBasalGangliaPutamen',
        'AxialSuperiorFrontal',
        'CoronalPosteriorParietalLingual',
        'CoronalCaudateAmygdala',
        'CoronalOrbitoFrontal',
        'SagittalInsulaFrontoTemporal',
        'SagittalCorpusCallosum',
        'SagittalInsulaTemporalHippocampalSulcus',
    ]

    scene_index = list(range(1, len(image_descriptions) + 1))

    return scene_index, image_descriptions
