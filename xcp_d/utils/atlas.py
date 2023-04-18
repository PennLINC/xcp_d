"""Functions for working with atlases."""


def get_atlas_names():
    """Get a list of atlases to be used for parcellation and functional connectivity analyses.

    The actual list of files for the atlases is loaded from a different function.

    NOTE: This is a Node function.

    Returns
    -------
    :obj:`list` of :obj:`str`
        List of atlases.
    """
    return [
        "Schaefer117",
        "Schaefer217",
        "Schaefer317",
        "Schaefer417",
        "Schaefer517",
        "Schaefer617",
        "Schaefer717",
        "Schaefer817",
        "Schaefer917",
        "Schaefer1017",
        "Glasser",
        "Gordon",
        "Tian",
        "ciftiSubcortical",
    ]


def get_atlas_nifti(atlas_name):
    """Select atlas by name from xcp_d/data using pkgrf.

    All atlases are in MNI space.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlas_name : {"Schaefer117", "Schaefer217", "Schaefer317", "Schaefer417", \
                  "Schaefer517", "Schaefer617", "Schaefer717", "Schaefer817", \
                  "Schaefer917", "Schaefer1017", "Glasser", "Gordon", \
                  "Tian", "ciftiSubcortical"}
        The name of the NIFTI atlas to fetch.

    Returns
    -------
    atlas_file : :obj:`str`
        Path to the atlas file.
    """
    import os

    from pkg_resources import resource_filename as pkgrf

    if "Schaefer" in atlas_name:
        n_parcels = int(atlas_name[8:]) - 17
        atlas_fname = (
            "tpl-MNI152NLin6Asym_atlas-Schaefer2018v0143_res-02_"
            f"desc-{n_parcels}Parcels17Networks_dseg.nii.gz"
        )
        tsv_fname = f"atlas-Schaefer2018v0143_desc-{n_parcels}Parcels17Networks_dseg.tsv"
    elif atlas_name in ("Glasser", "Gordon"):
        # 1 mm3 atlases
        atlas_fname = f"tpl-MNI152NLin6Asym_atlas-{atlas_name}_res-01_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas_name}_dseg.tsv"
    else:
        # 2 mm3 atlases
        atlas_fname = f"tpl-MNI152NLin6Asym_atlas-{atlas_name}_res-02_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas_name}_dseg.tsv"

    atlas_file = pkgrf("xcp_d", f"data/atlases/{atlas_fname}")
    atlas_labels_file = pkgrf("xcp_d", f"data/atlases/{tsv_fname}")

    if not os.path.isfile(atlas_file):
        raise FileNotFoundError(f"File DNE: {atlas_file}")

    if not os.path.isfile(atlas_labels_file):
        raise FileNotFoundError(f"File DNE: {atlas_labels_file}")

    return atlas_file, atlas_labels_file


def get_atlas_cifti(atlas_name):
    """Select atlas by name from xcp_d/data.

    All atlases are in 91K space.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlas_name : {"Schaefer117", "Schaefer217", "Schaefer317", "Schaefer417", \
                  "Schaefer517", "Schaefer617", "Schaefer717", "Schaefer817", \
                  "Schaefer917", "Schaefer1017", "Glasser", "Gordon", \
                  "Tian", "ciftiSubcortical"}
        The name of the CIFTI atlas to fetch.

    Returns
    -------
    atlas_file : :obj:`str`
        Path to the atlas file.
    """
    import os

    from pkg_resources import resource_filename as pkgrf

    if "Schaefer" in atlas_name:
        n_parcels = int(atlas_name[8:]) - 17
        atlas_fname = (
            "tpl-fsLR_atlas-Schaefer2018v0143_den-32k_"
            f"desc-{n_parcels}Parcels17Networks_dseg.dlabel.nii"
        )
        tsv_fname = f"atlas-Schaefer2018v0143_desc-{n_parcels}Parcels17Networks_dseg.tsv"
    else:
        atlas_fname = f"tpl-fsLR_atlas-{atlas_name}_den-32k_dseg.dlabel.nii"
        tsv_fname = f"atlas-{atlas_name}_dseg.tsv"

    atlas_file = pkgrf("xcp_d", f"data/atlases/{atlas_fname}")
    atlas_labels_file = pkgrf("xcp_d", f"data/atlases/{tsv_fname}")

    if not os.path.isfile(atlas_file):
        raise FileNotFoundError(f"File DNE: {atlas_file}")

    if not os.path.isfile(atlas_labels_file):
        raise FileNotFoundError(f"File DNE: {atlas_labels_file}")

    return atlas_file, atlas_labels_file
