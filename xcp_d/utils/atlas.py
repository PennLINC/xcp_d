"""Functions for working with atlases."""

from nipype import logging

LOGGER = logging.getLogger("nipype.utils")


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
            "MIDB",
            "MyersLabonte",
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
    """Select atlas by name from xcp_d/data using load_data.

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
    import json
    from os.path import isfile, join

    from xcp_d.data import load as load_data

    if "4S" in atlas:
        # 1 mm3 atlases
        atlas_fname = f"tpl-MNI152NLin6Asym_atlas-{atlas}_res-01_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas}_dseg.tsv"
    elif atlas in ("Glasser", "Gordon"):
        # 1 mm3 atlases
        atlas_fname = f"atlas-{atlas}_space-MNI152NLin6Asym_res-01_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas}_dseg.tsv"
    else:
        # 2 mm3 atlases
        atlas_fname = f"atlas-{atlas}_space-MNI152NLin6Asym_res-02_dseg.nii.gz"
        tsv_fname = f"atlas-{atlas}_dseg.tsv"

    if "4S" in atlas:
        atlas_file = join("/AtlasPack", atlas_fname)
        atlas_labels_file = join("/AtlasPack", tsv_fname)
        atlas_metadata_file = f"/AtlasPack/tpl-MNI152NLin6Asym_atlas-{atlas}_dseg.json"
    else:
        atlas_dir = f"atlases/atlas-{atlas}/"
        atlas_file = str(load_data(f"{atlas_dir}/{atlas_fname}"))
        atlas_labels_file = str(load_data(f"{atlas_dir}/{tsv_fname}"))
        atlas_metadata_file = str(
            load_data(f"{atlas_dir}/atlas-{atlas}_space-MNI152NLin6Asym_dseg.json")
        )

    if not (isfile(atlas_file) and isfile(atlas_labels_file) and isfile(atlas_metadata_file)):
        raise FileNotFoundError(
            f"File(s) DNE:\n\t{atlas_file}\n\t{atlas_labels_file}\n\t{atlas_metadata_file}"
        )

    with open(atlas_metadata_file, "r") as fo:
        atlas_metadata = json.load(fo)

    return {
        "image": atlas_file,
        "labels": atlas_labels_file,
        "metadata": atlas_metadata,
        "dataset": "XCP-D",
    }


def get_atlas_cifti(atlas):
    """Select atlas by name from xcp_d/data.

    All atlases are in 91K space.

    NOTE: This is a Node function.

    Parameters
    ----------
    atlas : {"4S156Parcels", "4S256Parcels", "4S356Parcels", "4S456Parcels", \
             "4S556Parcels", "4S656Parcels", "4S756Parcels", "4S856Parcels", \
             "4S956Parcels", "4S1056Parcels", "Glasser", "Gordon", \
             "Tian", "HCP", "MIDB", "MyersLabonte"}
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
    import json
    from os.path import isfile

    from xcp_d.data import load as load_data

    atlas_dir = f"atlases/atlas-{atlas}/"

    if "4S" in atlas:
        atlas_file = f"/AtlasPack/tpl-fsLR_atlas-{atlas}_den-91k_dseg.dlabel.nii"
        atlas_labels_file = f"/AtlasPack/atlas-{atlas}_dseg.tsv"
        atlas_metadata_file = f"/AtlasPack/tpl-fsLR_atlas-{atlas}_dseg.json"
    elif "MIDB" in atlas:
        atlas_file = str(
            load_data(
                f"{atlas_dir}/atlas-{atlas}_space-fsLR_den-32k_desc-abcdThresh75_dseg.dlabel.nii"
            )
        )
        atlas_labels_file = str(load_data(f"{atlas_dir}/atlas-{atlas}_dseg.tsv"))
        atlas_metadata_file = str(
            load_data(f"{atlas_dir}/atlas-{atlas}_space-fsLR_den-32k_desc-abcdThresh75_dseg.json")
        )
    elif "MyersLabonte" in atlas:
        atlas_file = str(
            load_data(
                f"{atlas_dir}/atlas-{atlas}_space-fsLR_den-32k_desc-thresh50_dseg.dlabel.nii"
            )
        )
        atlas_labels_file = str(load_data(f"{atlas_dir}/atlas-{atlas}_desc-thresh50_dseg.tsv"))
        atlas_metadata_file = str(
            load_data(f"{atlas_dir}/atlas-{atlas}_space-fsLR_den-32k_desc-thresh50_dseg.json")
        )
    else:
        atlas_file = str(
            load_data(f"{atlas_dir}/atlas-{atlas}_space-fsLR_den-32k_dseg.dlabel.nii")
        )
        atlas_labels_file = str(load_data(f"{atlas_dir}/atlas-{atlas}_dseg.tsv"))
        atlas_metadata_file = str(load_data(f"{atlas_dir}/atlas-{atlas}_space-fsLR_dseg.json"))

    if not (isfile(atlas_file) and isfile(atlas_labels_file) and isfile(atlas_metadata_file)):
        raise FileNotFoundError(
            f"File(s) DNE:\n\t{atlas_file}\n\t{atlas_labels_file}\n\t{atlas_metadata_file}"
        )

    with open(atlas_metadata_file, "r") as fo:
        atlas_metadata = json.load(fo)

    return {
        "image": atlas_file,
        "labels": atlas_labels_file,
        "metadata": atlas_metadata,
        "dataset": "XCP-D",
    }


def collect_atlases(datasets, file_format, bids_filters={}):
    """Collect atlases from a list of BIDS-Atlas datasets.

    Selection of labels files and metadata does not leverage the inheritance principle.
    That probably won't be possible until PyBIDS supports the BIDS-Atlas extension natively.

    Parameters
    ----------
    datasets : list of str or BIDSLayout

    Returns
    -------
    atlases : dict
    """
    import json

    import pandas as pd
    from bids.layout import BIDSLayout

    from xcp_d.data import load as load_data

    atlas_cfg = load_data("atlas_bids_config.json")
    bids_filters = bids_filters or {}

    builtin_atlases = select_atlases(atlases=None, subset="all")
    atlas_datasets = sorted(list(set(datasets) - set(builtin_atlases)))
    builtin_atlases = sorted(list(set(datasets) - set(atlas_datasets)))

    atlas_filter = bids_filters.get("atlas", {})
    atlas_filter["extension"] = [".nii.gz", ".nii"] if file_format == "nifti" else ".dlabel.nii"
    # Hardcoded spaces for now
    if file_format == "cifti":
        atlas_filter["space"] = atlas_filter.get("space") or "fsLR"
        atlas_filter["den"] = atlas_filter.get("den") or "32k"
    else:
        atlas_filter["space"] = atlas_filter.get("space") or [
            "MNI152NLin6Asym",
            "MNI152NLin2009cAsym",
            "MNIInfant",
        ]

    collection_func = get_atlas_nifti if file_format == "nifti" else get_atlas_cifti

    atlases = {}
    for builtin_atlas in builtin_atlases:
        atlases[builtin_atlas] = collection_func(builtin_atlas)

    for dataset in atlas_datasets:
        if not isinstance(dataset, BIDSLayout):
            layout = BIDSLayout(dataset, config=[atlas_cfg], validate=False)
        else:
            layout = dataset

        description = layout.get_dataset_description()
        assert description.get("DatasetType") == "atlas"

        atlases_in_dataset = layout.get_atlases(**atlas_filter)
        if not atlases_in_dataset:
            LOGGER.warning(f"No atlases found in dataset {layout.root} with query {atlas_filter}")
            continue

        for atlas in atlases_in_dataset:
            atlas_images = layout.get(
                atlas=atlas,
                **atlas_filter,
                return_type="file",
            )
            if not atlas_images:
                LOGGER.warning(f"No atlas images found for {atlas} with query {atlas_filter}")
                continue
            elif len(atlas_images) > 1:
                bulleted_list = "\n".join([f"  - {img}" for img in atlas_images])
                LOGGER.warning(
                    f"Multiple atlas images found for {atlas} with query {atlas_filter}:\n"
                    f"{bulleted_list}\nUsing {atlas_images[0]}."
                )

            atlas_image = atlas_images[0]
            atlas_labels = layout.get_nearest(atlas_image, extension=".tsv", strict=False)
            atlas_metadata_file = layout.get_nearest(atlas_image, extension=".json", strict=True)

            atlas_metadata = None
            if atlas_metadata_file:
                with open(atlas_metadata_file, "r") as fo:
                    atlas_metadata = json.load(fo)

            atlases[atlas] = {
                "dataset": description.get("Name", layout.root),
                "image": atlas_image,
                "labels": atlas_labels,
                "metadata": atlas_metadata,
            }

    for atlas, atlas_info in atlases.items():
        if not atlas_info["labels"]:
            raise FileNotFoundError(f"No TSV file found for {atlas_info['image']}")

        # Check the contents of the labels file
        df = pd.read_table(atlas_info["labels"])
        if "label" not in df.columns:
            raise ValueError(f"'label' column not found in {atlas_info['labels']}")

        if "index" not in df.columns:
            raise ValueError(f"'index' column not found in {atlas_info['labels']}")

    return atlases
