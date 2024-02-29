"""Functions for working with atlases."""


def select_atlases(atlases, subset):
    """Get a list of atlases to be used for parcellation and functional connectivity analyses.

    The actual list of files for the atlases is loaded from a different function.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlases : None or list of str
    subset : {"all", "subcortical", "cortical"}
        Description of the subset of atlases to collect.

    Returns
    -------
    :obj:`list` of :obj:`str`
        List of atlases.
    """
    BUILTIN_ATLASES = {
        "cortical": [
            "4S156Parcels",
            "4S256Parcels",
            "4S356Parcels",
            "4S456Parcels",
            "4S556Parcels",
            "4S656Parcels",
            "4S756Parcels",
            "4S856Parcels",
            "4S956Parcels",
            "4S1056Parcels",
            "Glasser",
            "Gordon",
        ],
        "subcortical": [
            "Tian",
            "HCP",
        ],
    }
    BUILTIN_ATLASES["all"] = sorted(
        list(set(BUILTIN_ATLASES["cortical"] + BUILTIN_ATLASES["subcortical"]))
    )
    subset_atlases = BUILTIN_ATLASES[subset]
    if atlases:
        assert all([atlas in BUILTIN_ATLASES["all"] for atlas in atlases])
        selected_atlases = [atlas for atlas in atlases if atlas in subset_atlases]
    else:
        selected_atlases = subset_atlases

    return selected_atlases


def get_atlas_nifti(atlas):
    """Select atlas by name from xcp_d/data using pkgrf.

    All atlases are in MNI space.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlas : {"4S156Parcels", "4S256Parcels", "4S356Parcels", "4S456Parcels", \
             "4S556Parcels", "4S656Parcels", "4S756Parcels", "4S856Parcels", \
             "4S956Parcels", "4S1056Parcels", "Glasser", "Gordon", \
             "Tian", "HCP"}
        The name of the NIFTI atlas to fetch.

    Returns
    -------
    atlas_file : :obj:`str`
        Path to the atlas file.
    atlas_labels_file : :obj:`str`
        Path to the atlas labels file.
    atlas_metadata_file : :obj:`str`
        Path to the atlas metadata file.
    """
    from os.path import isfile, join

    from pkg_resources import resource_filename as pkgrf

    if "4S" in atlas or atlas in ("Glasser", "Gordon"):
        # 1 mm3 atlases
        atlas_fname = f"tpl-MNI152NLin6Asym_atlas-{atlas}_res-01_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas}_dseg.tsv"
    else:
        # 2 mm3 atlases
        atlas_fname = f"tpl-MNI152NLin6Asym_atlas-{atlas}_res-02_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas}_dseg.tsv"

    if "4S" in atlas:
        atlas_file = join("/AtlasPack", atlas_fname)
        atlas_labels_file = join("/AtlasPack", tsv_fname)
        atlas_metadata_file = f"/AtlasPack/tpl-MNI152NLin6Asym_atlas-{atlas}_dseg.json"
    else:
        atlas_file = pkgrf("xcp_d", f"data/atlases/{atlas_fname}")
        atlas_labels_file = pkgrf("xcp_d", f"data/atlases/{tsv_fname}")
        atlas_metadata_file = pkgrf(
            "xcp_d",
            f"data/atlases/tpl-MNI152NLin6Asym_atlas-{atlas}_dseg.json",
        )

    if not (isfile(atlas_file) and isfile(atlas_labels_file) and isfile(atlas_metadata_file)):
        raise FileNotFoundError(
            f"File(s) DNE:\n\t{atlas_file}\n\t{atlas_labels_file}\n\t{atlas_metadata_file}"
        )

    return atlas_file, atlas_labels_file, atlas_metadata_file


def get_atlas_cifti(atlas):
    """Select atlas by name from xcp_d/data.

    All atlases are in 91K space.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlas : {"4S156Parcels", "4S256Parcels", "4S356Parcels", "4S456Parcels", \
             "4S556Parcels", "4S656Parcels", "4S756Parcels", "4S856Parcels", \
             "4S956Parcels", "4S1056Parcels", "Glasser", "Gordon", \
             "Tian", "HCP"}
        The name of the CIFTI atlas to fetch.

    Returns
    -------
    atlas_file : :obj:`str`
        Path to the atlas file.
    atlas_labels_file : :obj:`str`
        The labels TSV file associated with the atlas.
    atlas_metadata_file : :obj:`str`
        The metadata JSON file associated with the atlas.
    """
    from os.path import isfile

    from pkg_resources import resource_filename as pkgrf

    if "4S" in atlas:
        atlas_file = f"/AtlasPack/tpl-fsLR_atlas-{atlas}_den-91k_dseg.dlabel.nii"
        atlas_labels_file = f"/AtlasPack/atlas-{atlas}_dseg.tsv"
        atlas_metadata_file = f"/AtlasPack/tpl-fsLR_atlas-{atlas}_dseg.json"
    else:
        atlas_file = pkgrf(
            "xcp_d",
            f"data/atlases/tpl-fsLR_atlas-{atlas}_den-32k_dseg.dlabel.nii",
        )
        atlas_labels_file = pkgrf("xcp_d", f"data/atlases/atlas-{atlas}_dseg.tsv")
        atlas_metadata_file = pkgrf("xcp_d", f"data/atlases/tpl-fsLR_atlas-{atlas}_dseg.json")

    if not (isfile(atlas_file) and isfile(atlas_labels_file) and isfile(atlas_metadata_file)):
        raise FileNotFoundError(
            f"File(s) DNE:\n\t{atlas_file}\n\t{atlas_labels_file}\n\t{atlas_metadata_file}"
        )

    return atlas_file, atlas_labels_file, atlas_metadata_file


def copy_atlas(name_source, in_file, output_dir, atlas):
    """Copy atlas file to output directory.

    Parameters
    ----------
    name_source : :obj:`str`
        The source name of the atlas file.
    in_file : :obj:`str`
        The atlas file to copy.
    output_dir : :obj:`str`
        The output directory.
    atlas : :obj:`str`
        The name of the atlas.

    Returns
    -------
    out_file : :obj:`str`
        The path to the copied atlas file.

    Notes
    -----
    I can't use DerivativesDataSink because it has a problem with dlabel CIFTI files.
    It gives the following error:
    "AttributeError: 'Cifti2Header' object has no attribute 'set_data_dtype'"

    I can't override the CIFTI atlas's data dtype ahead of time because setting it to int8 or int16
    somehow converts all of the values in the data array to weird floats.
    This could be a version-specific nibabel issue.

    I've also updated this function to handle JSON and TSV files as well.
    """
    import os
    import shutil

    from xcp_d.utils.bids import get_entity

    if in_file.endswith(".json"):
        out_basename = f"atlas-{atlas}_dseg.json"
    elif in_file.endswith(".tsv"):
        out_basename = f"atlas-{atlas}_dseg.tsv"
    else:
        extension = ".nii.gz" if name_source.endswith(".nii.gz") else ".dlabel.nii"
        space = get_entity(name_source, "space")
        res = get_entity(name_source, "res")
        den = get_entity(name_source, "den")
        cohort = get_entity(name_source, "cohort")

        cohort_str = f"_cohort-{cohort}" if cohort else ""
        res_str = f"_res-{res}" if res else ""
        den_str = f"_den-{den}" if den else ""
        if extension == ".dlabel.nii":
            out_basename = f"atlas-{atlas}_space-{space}{den_str}{cohort_str}_dseg{extension}"
        elif extension == ".nii.gz":
            out_basename = f"atlas-{atlas}_space-{space}{res_str}{cohort_str}_dseg{extension}"

    atlas_out_dir = os.path.join(output_dir, f"xcp_d/atlases/atlas-{atlas}")
    os.makedirs(atlas_out_dir, exist_ok=True)
    out_file = os.path.join(atlas_out_dir, out_basename)
    # Don't copy the file if it exists, to prevent any race conditions between parallel processes.
    if not os.path.isfile(out_file):
        shutil.copyfile(in_file, out_file)

    return out_file
