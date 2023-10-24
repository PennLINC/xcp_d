"""Functions for working with atlases."""


def get_atlas_names(subset):
    """Get a list of atlases to be used for parcellation and functional connectivity analyses.

    The actual list of files for the atlases is loaded from a different function.

    NOTE: This is a Node function.

    Parameters
    ----------
    subset = {"all", "subcortical", "cortical"}
        Description of the subset of atlases to collect.

    Returns
    -------
    :obj:`list` of :obj:`str`
        List of atlases.
    """
    atlases = {
        "cortical": [
            "4S152Parcels",
            "4S252Parcels",
            "4S352Parcels",
            "4S452Parcels",
            "4S552Parcels",
            "4S652Parcels",
            "4S752Parcels",
            "4S852Parcels",
            "4S952Parcels",
            "4S1052Parcels",
            "Glasser",
            "Gordon",
        ],
        "subcortical": [
            "Tian",
            "HCP",
        ],
    }
    atlases["all"] = sorted(list(set(atlases["cortical"] + atlases["subcortical"])))
    return atlases[subset]


def get_atlas_nifti(atlas_name):
    """Select atlas by name from xcp_d/data using pkgrf.

    All atlases are in MNI space.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlas_name : {"4S152Parcels", "4S252Parcels", "4S352Parcels", "4S452Parcels", \
                  "4S552Parcels", "4S652Parcels", "4S752Parcels", "4S852Parcels", \
                  "4S952Parcels", "4S1052Parcels", "Glasser", "Gordon", \
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

    if "4S" in atlas_name or atlas_name in ("Glasser", "Gordon"):
        # 1 mm3 atlases
        atlas_fname = f"tpl-MNI152NLin6Asym_atlas-{atlas_name}_res-01_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas_name}_dseg.tsv"
    else:
        # 2 mm3 atlases
        atlas_fname = f"tpl-MNI152NLin6Asym_atlas-{atlas_name}_res-02_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas_name}_dseg.tsv"

    if "4S" in atlas_name:
        atlas_file = join("/AtlasPack", atlas_fname)
        atlas_labels_file = join("/AtlasPack", tsv_fname)
        atlas_metadata_file = f"/AtlasPack/tpl-MNI152NLin6Asym_atlas-{atlas_name}_dseg.json"
    else:
        atlas_file = pkgrf("xcp_d", f"data/atlases/{atlas_fname}")
        atlas_labels_file = pkgrf("xcp_d", f"data/atlases/{tsv_fname}")
        atlas_metadata_file = pkgrf(
            "xcp_d",
            f"data/atlases/tpl-MNI152NLin6Asym_atlas-{atlas_name}_dseg.json",
        )

    if not (isfile(atlas_file) and isfile(atlas_labels_file) and isfile(atlas_metadata_file)):
        raise FileNotFoundError(
            f"File(s) DNE:\n\t{atlas_file}\n\t{atlas_labels_file}\n\t{atlas_metadata_file}"
        )

    return atlas_file, atlas_labels_file, atlas_metadata_file


def get_atlas_cifti(atlas_name):
    """Select atlas by name from xcp_d/data.

    All atlases are in 91K space.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlas_name : {"4S152Parcels", "4S252Parcels", "4S352Parcels", "4S452Parcels", \
                  "4S552Parcels", "4S652Parcels", "4S752Parcels", "4S852Parcels", \
                  "4S952Parcels", "4S1052Parcels", "Glasser", "Gordon", \
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

    if "4S" in atlas_name:
        atlas_file = f"/AtlasPack/tpl-fsLR_atlas-{atlas_name}_den-91k_dseg.dlabel.nii"
        atlas_labels_file = f"/AtlasPack/atlas-{atlas_name}_dseg.tsv"
        atlas_metadata_file = f"/AtlasPack/tpl-fsLR_atlas-{atlas_name}_dseg.json"
    else:
        atlas_file = pkgrf(
            "xcp_d",
            f"data/atlases/tpl-fsLR_atlas-{atlas_name}_den-32k_dseg.dlabel.nii",
        )
        atlas_labels_file = pkgrf("xcp_d", f"data/atlases/atlas-{atlas_name}_dseg.tsv")
        atlas_metadata_file = pkgrf("xcp_d", f"data/atlases/tpl-fsLR_atlas-{atlas_name}_dseg.json")

    if not (isfile(atlas_file) and isfile(atlas_labels_file) and isfile(atlas_metadata_file)):
        raise FileNotFoundError(
            f"File(s) DNE:\n\t{atlas_file}\n\t{atlas_labels_file}\n\t{atlas_metadata_file}"
        )

    return atlas_file, atlas_labels_file, atlas_metadata_file


def copy_atlas(in_file, output_dir, atlas, extension, space=None, res=None, den=None, cohort=None):
    """Copy atlas file to output directory.

    I can't use DerivativesDataSink because it has a problem with dlabel CIFTI files.
    It gives the following error:
    "AttributeError: 'Cifti2Header' object has no attribute 'set_data_dtype'"

    I can't override the CIFTI atlas's data dtype ahead of time because setting it to int8 or int16
    somehow converts all of the values in the data array to weird floats.
    This could be a version-specific nibabel issue.
    """
    import os
    import shutil

    cohort_str = f"_cohort-{cohort}" if cohort else ""
    res_str = f"_res-{res}" if res else ""
    den_str = f"_den-{den}" if den else ""
    if extension == ".dlabel.nii":
        atlas_basename = f"space-{space}_atlas-{atlas}{den_str}{cohort_str}_dseg{extension}"
    elif extension == ".nii.gz":
        atlas_basename = f"space-{space}_atlas-{atlas}{res_str}{cohort_str}_dseg{extension}"

    out_atlas_file = os.path.join(output_dir, "xcp_d", atlas_basename)
    shutil.copyfile(in_file, out_atlas_file)
    return out_atlas_file
