"""Functions for working with atlases."""


def get_atlas_names():
    """Get a list of atlases to be used for parcellation and functional connectivity analyses.

    The actual list of files for the atlases is loaded from a different function.

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
        "subcortical",
    ]


def get_atlas_file(atlas_name, cifti):
    """Select atlas by name from xcp_d/data using pkgrf.

    All atlases are in MNI or fsLR space.

    Parameters
    ----------
    atlas_name : {"Schaefer117", "Schaefer217", "Schaefer317", "Schaefer417", \
                  "Schaefer517", "Schaefer617", "Schaefer717", "Schaefer817", \
                  "Schaefer917", "Schaefer1017", "Glasser", "Gordon", \
                  "subcortical"}
        The name of the NIFTI atlas to fetch.
    cifti : bool
        True if requesting CIFTI atlases, False if requesting NIFTI atlases.

    Returns
    -------
    atlas_file : str
        Path to the atlas file.
    node_labels_file : str
        Path to the node labels file.
    """
    import os

    from pkg_resources import resource_filename as pkgrf

    suffix = "_fsLR_32k.dlabel.nii" if cifti else "_FSLMNI152_2mm.nii.gz"
    folder = "ciftiatlas" if cifti else "niftiatlas"

    if atlas_name[:8] == "Schaefer":
        if atlas_name[8:12] == "1017":
            n_parcels = "1000"
        else:
            n_parcels = f"{atlas_name[8]}00"

        filename = f"Schaefer2018_{n_parcels}Parcels_17Networks_order{suffix}"

    elif atlas_name == "Glasser":
        filename = f"Glasser_360Parcels{suffix}"

    elif atlas_name == "Gordon":
        filename = f"Gordon_333Parcels{suffix}"

    elif atlas_name == "subcortical":
        filename = f"Tian{suffix}"

    else:
        raise RuntimeError(f"Atlas '{atlas_name}' not available")

    atlas_file = pkgrf("xcp_d", f"data/{folder}/{filename}")
    node_labels_file = atlas_file.replace(suffix, "_info.tsv")

    assert os.path.isfile(node_labels_file)

    return atlas_file, node_labels_file
