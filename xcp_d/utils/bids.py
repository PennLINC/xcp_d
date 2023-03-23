# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities for fmriprep bids derivatives and layout.

Most of the code is copied from niworkflows.
A PR will be submitted to niworkflows at some point.
"""
import warnings

import nibabel as nb
import yaml
from bids import BIDSLayout
from nipype import logging
from packaging.version import Version

from xcp_d.utils.doc import fill_doc
from xcp_d.utils.filemanip import ensure_list

LOGGER = logging.getLogger("nipype.utils")

# TODO: Add and test fsaverage.
DEFAULT_ALLOWED_SPACES = {
    "cifti": ["fsLR"],
    "nifti": [
        "MNI152NLin6Asym",
        "MNI152NLin2009cAsym",
        "MNIInfant",
    ],
}
INPUT_TYPE_ALLOWED_SPACES = {
    "nibabies": {
        "cifti": ["fsLR"],
        "nifti": [
            "MNIInfant",
            "MNI152NLin6Asym",
            "MNI152NLin2009cAsym",
        ],
    },
}
# The volumetric NIFTI template associated with each supported CIFTI template.
ASSOCIATED_TEMPLATES = {
    "fsLR": "MNI152NLin6Asym",
}


class BIDSError(ValueError):
    """A generic error related to BIDS datasets.

    Parameters
    ----------
    message : :obj:`str`
        The error message.
    bids_root : :obj:`str`
        The path to the BIDS dataset.
    """

    def __init__(self, message, bids_root):
        indent = 10
        header = (
            f'{"".join(["-"] * indent)} BIDS root folder: "{bids_root}" '
            f'{"".join(["-"] * indent)}'
        )
        self.msg = (
            f"\n{header}\n{''.join([' '] * (indent + 1))}{message}\n"
            f"{''.join(['-'] * len(header))}"
        )
        super(BIDSError, self).__init__(self.msg)
        self.bids_root = bids_root


class BIDSWarning(RuntimeWarning):
    """A generic warning related to BIDS datasets."""

    pass


def collect_participants(bids_dir, participant_label=None, strict=False, bids_validate=False):
    """Collect a list of participants from a BIDS dataset.

    Parameters
    ----------
    bids_dir : :obj:`str` or pybids.layout.BIDSLayout
    participant_label : None or str, optional
    strict : bool, optional
    bids_validate : bool, optional

    Returns
    -------
    found_label

    Examples
    --------
    Requesting all subjects in a BIDS directory root:
    #>>> collect_participants(str(datadir / 'ds114'), bids_validate=False)
    ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    Requesting two subjects, given their IDs:
    #>>> collect_participants(str(datadir / 'ds114'), participant_label=['02', '04'],
    #...                      bids_validate=False)
    ['02', '04']
    ...
    """
    if isinstance(bids_dir, BIDSLayout):
        layout = bids_dir
    else:
        layout = BIDSLayout(str(bids_dir), validate=bids_validate, derivatives=True)

    all_participants = set(layout.get_subjects())

    # Error: bids_dir does not contain subjects
    if not all_participants:
        raise BIDSError(
            "Could not find participants. Please make sure the BIDS derivatives "
            "are accessible to Docker/ are in BIDS directory structure.",
            bids_dir,
        )

    # No --participant-label was set, return all
    if not participant_label:
        return sorted(all_participants)

    if isinstance(participant_label, str):
        participant_label = [participant_label]

    # Drop sub- prefixes
    participant_label = [sub[4:] if sub.startswith("sub-") else sub for sub in participant_label]
    # Remove duplicates
    participant_label = sorted(set(participant_label))
    # Remove labels not found
    found_label = sorted(set(participant_label) & all_participants)
    if not found_label:
        raise BIDSError(
            f"Could not find participants [{', '.join(participant_label)}]",
            bids_dir,
        )

    if notfound_label := sorted(set(participant_label) - all_participants):
        exc = BIDSError(
            f"Some participants were not found: {', '.join(notfound_label)}",
            bids_dir,
        )
        if strict:
            raise exc
        warnings.warn(exc.msg, BIDSWarning)

    return found_label


@fill_doc
def collect_data(
    bids_dir,
    input_type,
    participant_label,
    task=None,
    bids_validate=False,
    bids_filters=None,
    cifti=False,
    layout=None,
):
    """Collect data from a BIDS dataset.

    Parameters
    ----------
    bids_dir
    %(input_type)s
    participant_label
    task
    bids_validate
    bids_filters
    %(cifti)s
    %(layout)s

    Returns
    -------
    %(layout)s
    subj_data : dict
    """
    if not isinstance(layout, BIDSLayout):
        layout = BIDSLayout(
            str(bids_dir),
            validate=bids_validate,
            derivatives=True,
            config=["bids", "derivatives"],
        )

    queries = {
        # all preprocessed BOLD files in the right space/resolution/density
        "bold": {"datatype": "func", "suffix": "bold", "desc": ["preproc", None]},
        # native T1w-space, preprocessed T1w file
        "t1w": {
            "datatype": "anat",
            "space": None,
            "desc": "preproc",
            "suffix": "T1w",
            "extension": ".nii.gz",
        },
        # native T2w-space, preprocessed T1w file
        "t2w": {
            "datatype": "anat",
            "space": [None, "T1w"],
            "desc": "preproc",
            "suffix": "T2w",
            "extension": ".nii.gz",
        },
        # native T1w-space dseg file
        "anat_dseg": {
            "datatype": "anat",
            "space": None,
            "desc": None,
            "suffix": "dseg",
            "extension": ".nii.gz",
        },
        # transform from standard space to T1w or T2w space
        # "from" entity will be set later
        "template_to_anat_xfm": {
            "datatype": "anat",
            "to": "T1w",
            "suffix": "xfm",
        },
        # native T1w-space brain mask
        "anat_brainmask": {
            "datatype": "anat",
            "space": None,
            "desc": "brain",
            "suffix": "mask",
            "extension": ".nii.gz",
        },
        # transform from T1w or T2w space to standard space
        # "to" entity will be set later
        "anat_to_template_xfm": {
            "datatype": "anat",
            "from": "T1w",
            "suffix": "xfm",
        },
    }
    if input_type == "hcp":
        queries["t1w"]["space"] = "MNI152NLin6Asym"
        queries["t2w"]["space"] = "MNI152NLin6Asym"
        queries["anat_dseg"]["desc"] = "aparcaseg"
        queries["anat_dseg"]["space"] = "MNI152NLin6Asym"
        queries["anat_brainmask"]["space"] = "MNI152NLin6Asym"

    queries["bold"]["extension"] = ".dtseries.nii" if cifti else ".nii.gz"

    # Apply filters. These may override anything.
    bids_filters = bids_filters or {}
    for acq, entities in bids_filters.items():
        queries[acq].update(entities)

    # Some filters are applied as parameters to the function though.
    if task:
        queries["bold"]["task"] = task

    # Select the best available space.
    if "space" in queries["bold"]:
        # Hopefully no one puts in multiple spaces here,
        # but we'll grab the first one with available data if they did.
        allowed_spaces = ensure_list(queries["bold"]["space"])
    else:
        allowed_spaces = INPUT_TYPE_ALLOWED_SPACES.get(
            input_type,
            DEFAULT_ALLOWED_SPACES,
        )["cifti" if cifti else "nifti"]

    for space in allowed_spaces:
        queries["bold"]["space"] = space
        bold_data = layout.get(**queries["bold"])
        if bold_data:
            # will leave the best available space in the query
            break

    if not bold_data:
        raise FileNotFoundError(
            f"No BOLD data found in allowed spaces ({', '.join(allowed_spaces)})."
        )

    if cifti:
        # Select the appropriate volumetric space for the CIFTI template.
        # This space will be used in the executive summary and T1w/T2w workflows.
        temp_query = queries["anat_to_template_xfm"].copy()
        volumetric_space = ASSOCIATED_TEMPLATES[space]

        temp_query["to"] = volumetric_space
        transform_files = layout.get(**temp_query)
        if not transform_files:
            raise FileNotFoundError(
                f"No nifti transforms found to allowed space ({volumetric_space})"
            )

        queries["anat_to_template_xfm"]["to"] = volumetric_space
        queries["template_to_anat_xfm"]["from"] = volumetric_space
    else:
        # use the BOLD file's space if the BOLD file is a nifti.
        queries["anat_to_template_xfm"]["to"] = queries["bold"]["space"]
        queries["template_to_anat_xfm"]["from"] = queries["bold"]["space"]

    # Grab the first (and presumably best) density and resolution if there are multiple.
    # This probably works well for resolution (1 typically means 1x1x1,
    # 2 typically means 2x2x2, etc.), but probably doesn't work well for density.
    resolutions = layout.get_res(**queries["bold"])
    densities = layout.get_den(**queries["bold"])
    if len(resolutions) > 1:
        queries["bold"]["resolution"] = resolutions[0]

    if len(densities) > 1:
        queries["bold"]["den"] = densities[0]

    # Check for anatomical images, and determine if T2w xfms must be used.
    t1w_files = layout.get(return_type="file", subject=participant_label, **queries["t1w"])
    t2w_files = layout.get(return_type="file", subject=participant_label, **queries["t2w"])
    if not t1w_files and not t2w_files:
        raise FileNotFoundError("No T1w or T2w files found.")
    elif t2w_files and not t1w_files:
        LOGGER.warning("T2w found, but no T1w. Enabling T2w-only processing.")
        queries["template_to_anat_xfm"]["to"] = "T2w"
        queries["anat_to_template_xfm"]["from"] = "T2w"

    # Search for the files.
    subj_data = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                **query,
            )
        )
        for dtype, query in queries.items()
    }

    # Check the query results.
    for field, filenames in subj_data.items():
        # All fields except the BOLD data should have a single file
        if field != "bold" and isinstance(filenames, list):
            if field not in ("t1w", "t2w") and not filenames:
                raise FileNotFoundError(f"No {field} found with query: {queries[field]}")

            if len(filenames) == 1:
                subj_data[field] = filenames[0]
            elif len(filenames) > 1:
                filenames_str = "\n\t".join(filenames)
                LOGGER.warning(f"Multiple files found for query '{field}':\n\t{filenames_str}")
                subj_data[field] = filenames[0]
            else:
                subj_data[field] = None

    LOGGER.log(25, f"Collected data:\n{yaml.dump(subj_data, default_flow_style=False, indent=4)}")

    return layout, subj_data


def _find_standard_space_surfaces(layout, participant_label, queries):
    """Find standard-space surfaces for a given set of queries.

    Parameters
    ----------
    layout : BIDSLayout
    participant_label : str
    queries : dict of dict

    Returns
    -------
    surface_files_found : bool
    standard_space_surfaces : bool
    out_surface_files : dict
    """
    standard_space_surfaces = True
    for name, query in queries.items():
        # First, try to grab the first base surface file in standard space.
        # If it's not available, switch to native T1w-space data.
        temp_files = layout.get(
            return_type="file",
            subject=participant_label,
            datatype="anat",
            space="fsLR",
            den="32k",
            **query,
        )
        if len(temp_files) == 0:
            LOGGER.info("No standard-space surfaces found.")
            standard_space_surfaces = False
        elif len(temp_files) > 1:
            LOGGER.warning(f"{name}: More than one standard-space surface found.")

    # Now that we know if there are standard-space surfaces available, we can grab the files.
    if standard_space_surfaces:
        query_extras = {
            "space": "fsLR",
            "den": "32k",
        }
    else:
        query_extras = {
            "space": None,
        }

    surface_files = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                datatype="anat",
                **query,
                **query_extras,
            )
        )
        for dtype, query in queries.items()
    }

    out_surface_files = {}
    surface_files_found = True
    for dtype, surface_files_ in surface_files.items():
        if len(surface_files_) == 1:
            out_surface_files[dtype] = surface_files_[0]

        elif len(surface_files_) == 0:
            surface_files_found = False
            out_surface_files[dtype] = None

        else:
            surface_files_found = False
            surface_str = "\n\t".join(surface_files_)
            raise ValueError(
                "More than one surface found.\n"
                f"Surfaces found:\n\t{surface_str}\n"
                f"Query: {queries[dtype]}"
            )

    return surface_files_found, standard_space_surfaces, out_surface_files


@fill_doc
def collect_surface_data(layout, participant_label):
    """Collect surface files from preprocessed derivatives.

    This function will try to collect fsLR-space, 32k-resolution surface files first.
    If these standard-space surface files aren't available, it will default to native T1w-space
    files.

    Parameters
    ----------
    %(layout)s
    participant_label : :obj:`str`
        Subject ID.

    Returns
    -------
    mesh_available : :obj:`bool`
        True if surface mesh files (pial and smoothwm) were found. False if they were not.
    shape_available : :obj:`bool`
        True if surface shape files (curv, sulc, and thickness) were found. False if they were not.
    standard_space_mesh : :obj:`bool`
        True if standard-space (fsLR) surface mesh files were found. False if they were not.
    surface_files : :obj:`dict`
        Dictionary of surface file identifiers and their paths.
        If the surface files weren't found, then the paths will be Nones.
    """
    # Surfaces to use for brainsprite and anatomical workflow
    # The base surfaces can be used to generate the derived surfaces.
    # The base surfaces may be in native or standard space.
    mesh_queries = {
        "lh_pial_surf": {
            "hemi": "L",
            "desc": None,
            "suffix": "pial",
            "extension": ".surf.gii",
        },
        "rh_pial_surf": {
            "hemi": "R",
            "desc": None,
            "suffix": "pial",
            "extension": ".surf.gii",
        },
        "lh_wm_surf": {
            "hemi": "L",
            "desc": None,
            "suffix": "smoothwm",
            "extension": ".surf.gii",
        },
        "rh_wm_surf": {
            "hemi": "R",
            "desc": None,
            "suffix": "smoothwm",
            "extension": ".surf.gii",
        },
    }

    mesh_available, standard_space_mesh, mesh_files = _find_standard_space_surfaces(
        layout,
        participant_label,
        mesh_queries,
    )

    shape_queries = {
        "lh_sulcal_depth": {
            "hemi": "L",
            "desc": None,
            "suffix": "sulc",
            "extension": ".shape.gii",
        },
        "rh_sulcal_depth": {
            "hemi": "R",
            "desc": None,
            "suffix": "sulc",
            "extension": ".shape.gii",
        },
        "lh_sulcal_curv": {
            "hemi": "L",
            "desc": None,
            "suffix": "curv",
            "extension": ".shape.gii",
        },
        "rh_sulcal_curv": {
            "hemi": "R",
            "desc": None,
            "suffix": "curv",
            "extension": ".shape.gii",
        },
        "lh_cortical_thickness": {
            "hemi": "L",
            "desc": None,
            "suffix": "thickness",
            "extension": ".shape.gii",
        },
        "rh_cortical_thickness": {
            "hemi": "R",
            "desc": None,
            "suffix": "thickness",
            "extension": ".shape.gii",
        },
    }

    shape_available, _, shape_files = _find_standard_space_surfaces(
        layout,
        participant_label,
        shape_queries,
    )

    surface_files = {**mesh_files, **shape_files}

    LOGGER.log(
        25,
        (
            f"Collected surface data:\n"
            f"{yaml.dump(surface_files, default_flow_style=False, indent=4)}"
        ),
    )

    return mesh_available, shape_available, standard_space_mesh, surface_files


@fill_doc
def collect_run_data(layout, input_type, bold_file, cifti, primary_anat):
    """Collect data associated with a given BOLD file.

    Parameters
    ----------
    %(layout)s
    %(input_type)s
    bold_file : :obj:`str`
        Path to the BOLD file.
    %(cifti)s
        Whether to collect files associated with a CIFTI image (True) or a NIFTI (False).
    primary_anat : {"T1w", "T2w"}
        The anatomical modality to use for the anat-to-native transform.

    Returns
    -------
    run_data : :obj:`dict`
        A dictionary of file types (e.g., "confounds") and associated filenames.
    """
    bids_file = layout.get_file(bold_file)
    run_data, metadata = {}, {}
    run_data["confounds"] = layout.get_nearest(
        bids_file.path,
        strict=False,
        desc="confounds",
        suffix="timeseries",
        extension=".tsv",
    )
    metadata["bold_metadata"] = layout.get_metadata(bold_file)
    # Ensure that we know the TR
    if "RepetitionTime" not in metadata["bold_metadata"].keys():
        metadata["bold_metadata"]["RepetitionTime"] = _get_tr(bold_file)

    if not cifti:
        run_data["boldref"] = layout.get_nearest(
            bids_file.path,
            strict=False,
            suffix="boldref",
        )
        run_data["boldmask"] = layout.get_nearest(
            bids_file.path,
            strict=False,
            desc="brain",
            suffix="mask",
        )
        run_data["anat_to_native_xfm"] = layout.get_nearest(
            bids_file.path,
            strict=False,
            **{"from": primary_anat},  # "from" is protected Python kw
            to="scanner",
            suffix="xfm",
        )
    else:
        allowed_nifti_spaces = INPUT_TYPE_ALLOWED_SPACES.get(
            input_type,
            DEFAULT_ALLOWED_SPACES,
        )["nifti"]
        run_data["boldref"] = layout.get_nearest(
            bids_file.path,
            strict=False,
            space=allowed_nifti_spaces,
            suffix="boldref",
        )
        run_data["nifti_file"] = layout.get_nearest(
            bids_file.path,
            strict=False,
            space=allowed_nifti_spaces,
            desc="preproc",
            suffix="bold",
            extension=[".nii", ".nii.gz"],
        )

    LOGGER.log(
        25,
        (
            f"Collected run data for {bold_file}:\n"
            f"{yaml.dump(run_data, default_flow_style=False, indent=4)}"
        ),
    )

    for k, v in run_data.items():
        if v is None:
            raise FileNotFoundError(f"No {k} file found for {bids_file.path}")

        metadata[f"{k}_metadata"] = layout.get_metadata(v)

    run_data.update(metadata)

    return run_data


def write_dataset_description(fmri_dir, xcpd_dir):
    """Write dataset_description.json file for derivatives.

    Parameters
    ----------
    fmri_dir : :obj:`str`
        Path to the BIDS derivative dataset being ingested.
    xcpd_dir : :obj:`str`
        Path to the output xcp-d dataset.
    """
    import json
    import os

    from xcp_d.__about__ import DOWNLOAD_URL, __version__

    orig_dset_description = os.path.join(fmri_dir, "dataset_description.json")
    if not os.path.isfile(orig_dset_description):
        dset_desc = {}

    else:
        with open(orig_dset_description, "r") as fo:
            dset_desc = json.load(fo)

        assert dset_desc["DatasetType"] == "derivative"

    # Update dataset description
    dset_desc["Name"] = "XCP-D: A Robust Postprocessing Pipeline of fMRI data"
    generated_by = dset_desc.get("GeneratedBy", [])
    generated_by.insert(
        0,
        {
            "Name": "xcp_d",
            "Version": __version__,
            "CodeURL": DOWNLOAD_URL,
        },
    )
    dset_desc["GeneratedBy"] = generated_by
    dset_desc["HowToAcknowledge"] = "Include the generated boilerplate in the methods section."

    xcpd_dset_description = os.path.join(xcpd_dir, "dataset_description.json")
    if os.path.isfile(xcpd_dset_description):
        with open(xcpd_dset_description, "r") as fo:
            old_dset_desc = json.load(fo)

        old_version = old_dset_desc["GeneratedBy"][0]["Version"]
        if Version(__version__).public != Version(old_version).public:
            LOGGER.warning(f"Previous output generated by version {old_version} found.")

    else:
        with open(xcpd_dset_description, "w") as fo:
            json.dump(dset_desc, fo, indent=4, sort_keys=True)


def get_preproc_pipeline_info(input_type, fmri_dir):
    """Get preprocessing pipeline information from the dataset_description.json file."""
    import json
    import os

    dataset_description = os.path.join(fmri_dir, "dataset_description.json")
    if os.path.isfile(dataset_description):
        with open(dataset_description) as f:
            dataset_dict = json.load(f)

    info_dict = {
        "name": dataset_dict["GeneratedBy"][0]["Name"],
        "version": dataset_dict["GeneratedBy"][0]["Version"]
        if "Version" in dataset_dict["GeneratedBy"][0].keys()
        else "unknown",
    }
    if input_type == "fmriprep":
        info_dict["references"] = "[@esteban2019fmriprep;@esteban2020analysis, RRID:SCR_016216]"
    elif input_type == "dcan":
        info_dict["references"] = "[@Feczko_Earl_perrone_Fair_2021;@feczko2021adolescent]"
    elif input_type == "hcp":
        info_dict["references"] = "[@hcppipelines]"
    elif input_type == "nibabies":
        info_dict["references"] = "[@goncalves_mathias_2022_7072346]"
    else:
        raise ValueError(f"Unsupported input_type '{input_type}'")

    return info_dict


def _add_subject_prefix(subid):
    """Extract or compile subject entity from subject ID.

    Parameters
    ----------
    subid : :obj:`str`
        A subject ID (e.g., 'sub-XX' or just 'XX').

    Returns
    -------
    str
        Subject entity (e.g., 'sub-XX').
    """
    return subid if subid.startswith("sub-") else "-".join(("sub", subid))


def _get_tr(img):
    """Attempt to extract repetition time from NIfTI/CIFTI header.

    Examples
    --------
    _get_tr(nb.load(Path(test_data) /
    ...    'sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz'))
    2.2
     _get_tr(nb.load(Path(test_data) /
    ...    'sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii'))
    2.0
    """
    if isinstance(img, str):
        img = nb.load(img)

    try:
        return img.header.matrix.get_index_map(0).series_step  # Get TR
    except AttributeError:  # Error out if not in cifti
        return img.header.get_zooms()[-1]
    raise RuntimeError("Could not extract TR - unknown data structure type")


def get_freesurfer_dir(fmri_dir):
    """Find FreeSurfer derivatives associated with preprocessing pipeline.

    NOTE: This is a Node function.

    Parameters
    ----------
    fmri_dir : :obj:`str`
        Path to preprocessed derivatives.

    Returns
    -------
    freesurfer_path : :obj:`str`
        Path to FreeSurfer derivatives.

    Raises
    ------
    ValueError
        If more than one potential FreeSurfer derivative folder is found.
    NotADirectoryError
        If no FreeSurfer derivatives are found.
    """
    import glob
    import os

    # for fMRIPrep/Nibabies versions >=20.2.1
    freesurfer_paths = sorted(glob.glob(os.path.join(fmri_dir, "sourcedata/*freesurfer*")))
    if len(freesurfer_paths) == 0:
        # for fMRIPrep/Nibabies versions <20.2.1
        freesurfer_paths = sorted(
            glob.glob(os.path.join(os.path.dirname(fmri_dir), "*freesurfer*"))
        )

    if len(freesurfer_paths) == 1:
        freesurfer_path = freesurfer_paths[0]

    elif len(freesurfer_paths) > 1:
        freesurfer_paths_str = "\n\t".join(freesurfer_paths)
        raise ValueError(
            "More than one candidate for FreeSurfer derivatives found. "
            "We recommend mounting only one FreeSurfer directory in your Docker/Singularity "
            "image. "
            f"Detected candidates:\n\t{freesurfer_paths_str}"
        )

    else:
        raise NotADirectoryError("No FreeSurfer derivatives found.")

    return freesurfer_path


def get_freesurfer_sphere(freesurfer_path, subject_id, hemisphere):
    """Find FreeSurfer sphere file.

    NOTE: This is a Node function.

    Parameters
    ----------
    freesurfer_path : :obj:`str`
        Path to the FreeSurfer derivatives.
    subject_id : :obj:`str`
        Subject ID. This may or may not be prefixed with "sub-".
    hemisphere : {"L", "R"}
        The hemisphere to grab.

    Returns
    -------
    sphere_raw : :obj:`str`
        Sphere file for the requested subject and hemisphere.

    Raises
    ------
    FileNotFoundError
        If the sphere file cannot be found.
    """
    import os

    assert hemisphere in ("L", "R"), hemisphere

    if not subject_id.startswith("sub-"):
        subject_id = f"sub-{subject_id}"

    sphere_raw = os.path.join(
        freesurfer_path,
        subject_id,
        "surf",
        f"{hemisphere.lower()}h.sphere.reg",
    )

    if not os.path.isfile(sphere_raw):
        raise FileNotFoundError(f"Sphere file not found at '{sphere_raw}'")

    return sphere_raw


def get_entity(filename, entity):
    """Extract a given entity from a BIDS filename via string manipulation.

    Parameters
    ----------
    filename : :obj:`str`
        Path to the BIDS file.
    entity : :obj:`str`
        The entity to extract from the filename.

    Returns
    -------
    entity_value : :obj:`str` or None
        The BOLD file's entity value associated with the requested entity.
    """
    import os
    import re

    folder, file_base = os.path.split(filename)

    # Allow + sign, which is not allowed in BIDS,
    # but is used by templateflow for the MNIInfant template.
    entity_values = re.findall(f"{entity}-([a-zA-Z0-9+]+)", file_base)
    entity_value = None if len(entity_values) < 1 else entity_values[0]
    if entity == "space" and entity_value is None:
        foldername = os.path.basename(folder)
        if foldername == "anat":
            entity_value = "T1w"
        elif foldername == "func":
            entity_value = "native"
        else:
            raise ValueError(f"Unknown space for {filename}")

    return entity_value


def group_across_runs(in_files):
    """Group preprocessed BOLD files by unique sets of entities, ignoring run.

    Parameters
    ----------
    in_files : :obj:`list` of :obj:`str`
        A list of preprocessed BOLD files to group.

    Returns
    -------
    out_files : :obj:`list` of :obj:`list` of :obj:`str`
        The grouped BOLD files. Each sublist corresponds to a single set of runs.
    """
    import os
    import re

    # First, extract run information and sort the input files by the runs,
    # so that any cases where files are not already in ascending run order get fixed.
    run_numbers = []
    for in_file in in_files:
        run = get_entity(in_file, "run")
        if run is None:
            run = 0

        run_numbers.append(int(run))

    # Sort the files by the run numbers.
    zipped_pairs = zip(run_numbers, in_files)
    sorted_in_files = [x for _, x in sorted(zipped_pairs)]

    # Extract the unique sets of entities (i.e., the filename, minus the run entity).
    unique_filenames = [re.sub("_run-[0-9]+_", "_", os.path.basename(f)) for f in sorted_in_files]

    # Assign each in_file to a group of files with the same entities, except run.
    out_files, grouped_unique_filenames = [], []
    for i_file, in_file in enumerate(sorted_in_files):
        unique_filename = unique_filenames[i_file]
        if unique_filename not in grouped_unique_filenames:
            grouped_unique_filenames.append(unique_filename)
            out_files.append([])

        group_idx = grouped_unique_filenames.index(unique_filename)
        out_files[group_idx].append(in_file)

    return out_files
