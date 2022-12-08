# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities for fmriprep bids derivatives and layout.

Most of the code is copied from niworkflows.
A PR will be submitted to niworkflows at some point.
"""
import logging
import os
import warnings

import nibabel as nb
import yaml
from bids import BIDSLayout
from packaging.version import Version

from xcp_d.utils.doc import fill_doc
from xcp_d.utils.filemanip import ensure_list

LOGGER = logging.getLogger("nipype.utils")


class BIDSError(ValueError):
    """A generic error related to BIDS datasets.

    Parameters
    ----------
    message : str
        The error message.
    bids_root : str
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
    bids_dir : str or pybids.layout.BIDSLayout
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

    # Warn if some IDs were not found
    notfound_label = sorted(set(participant_label) - all_participants)
    if notfound_label:
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

    Returns
    -------
    layout : pybids.layout.BIDSLayout
    subj_data : dict
    """
    if input_type not in ("dcan", "hcp"):
        layout = BIDSLayout(
            str(bids_dir),
            validate=bids_validate,
            derivatives=True,
            config=["bids", "derivatives"],
        )
    # TODO: Add and test fsaverage.
    default_allowed_spaces = {
        "cifti": ["fsLR"],
        "nifti": [
            "MNI152NLin6Asym",
            "MNI152NLin2009cAsym",
            "MNIInfant",
        ],
    }
    input_type_allowed_spaces = {
        "nibabies": {
            "cifti": ["fsLR"],
            "nifti": [
                "MNIInfant",
                "MNI152NLin6Asym",
                "MNI152NLin2009cAsym",
            ],
        },
    }

    queries = {
        # all preprocessed BOLD files in the right space/resolution/density
        "bold": {"datatype": "func", "suffix": "bold", "desc": ["preproc", None]},
        # native T1w-space, preprocessed T1w file
        "t1w": {"datatype": "anat", "space": None, "suffix": "T1w", "extension": ".nii.gz"},
        # native T1w-space dseg file, but not aseg or aparcaseg
        "t1w_seg": {
            "datatype": "anat",
            "space": None,
            "desc": None,
            "suffix": "dseg",
            "extension": ".nii.gz",
        },
        # transform from standard space to T1w space
        # from entity will be set later
        "template_to_t1w_xform": {
            "datatype": "anat",
            "to": "T1w",
            "suffix": "xfm",
        },
        # native T1w-space brain mask
        "t1w_mask": {
            "datatype": "anat",
            "space": None,
            "desc": "brain",
            "suffix": "mask",
            "extension": ".nii.gz",
        },
        # transform from T1w space to standard space
        # to entity will be set later
        "t1w_to_template_xform": {
            "datatype": "anat",
            "from": "T1w",
            "suffix": "xfm",
        },
    }
    if cifti:
        queries["bold"]["extension"] = ".dtseries.nii"
    else:
        queries["bold"]["extension"] = ".nii.gz"

    # Apply filters. These may override anything.
    bids_filters = bids_filters or {}
    for acq, entities in bids_filters.items():
        queries[acq].update(entities)

    if task:
        queries["bold"]["task"] = task

    # Select the best available space
    if "space" in queries["bold"]:
        # Hopefully no one puts in multiple spaces here,
        # but we'll grab the first one with available data if they did.
        allowed_spaces = ensure_list(queries["bold"]["space"])
    else:
        allowed_spaces = input_type_allowed_spaces.get(input_type, default_allowed_spaces)[
            "cifti" if cifti else "nifti"
        ]

    for space in allowed_spaces:
        queries["bold"]["space"] = space
        bold_data = layout.get(**queries["bold"])
        if bold_data:
            # will leave the best available space in the query
            break

    if not bold_data:
        allowed_space_str = ", ".join(allowed_spaces)
        raise FileNotFoundError(f"No BOLD data found in allowed spaces ({allowed_space_str}).")

    if not cifti:
        # use the BOLD file's space if the BOLD file is a nifti
        queries["t1w_to_template_xform"]["to"] = queries["bold"]["space"]
        queries["template_to_t1w_xform"]["from"] = queries["bold"]["space"]
    else:
        # Select the best *volumetric* space, based on available nifti BOLD files.
        # This space will be used in the executive summary and T1w/T2w workflows.
        temp_bold_query = queries["bold"].copy()
        temp_bold_query["extension"] = ".nii.gz"
        temp_allowed_spaces = input_type_allowed_spaces.get(
            input_type,
            default_allowed_spaces,
        )["nifti"]

        if input_type in ("hcp", "dcan"):
            temp_allowed_spaces = ["MNI152NLin2009cAsym"]
            # HCP and DCAN files don't have nifti BOLD data, we will use the boldref
            temp_bold_query["desc"] = None
            temp_bold_query["suffix"] = "boldref"

        for space in temp_allowed_spaces:
            temp_bold_query["space"] = space
            nifti_bold_data = layout.get(**temp_bold_query)
            if nifti_bold_data:
                queries["t1w_to_template_xform"]["to"] = space
                queries["template_to_t1w_xform"]["from"] = space
                break

        if not nifti_bold_data:
            allowed_space_str = ", ".join(temp_allowed_spaces)
            raise FileNotFoundError(
                f"No nifti BOLD data found in allowed spaces ({allowed_space_str})"
            )

    # Grab the first (and presumably best) density and resolution if there are multiple.
    # This probably works well for resolution (1 typically means 1x1x1,
    # 2 typically means 2x2x2, etc.), but probably doesn't work well for density.
    resolutions = layout.get_res(**queries["bold"])
    densities = layout.get_den(**queries["bold"])
    if len(resolutions) > 1:
        queries["bold"]["resolution"] = resolutions[0]

    if len(densities) > 1:
        queries["bold"]["density"] = densities[0]

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

    for field, filenames in subj_data.items():
        # All fields except the BOLD data should have a single file
        if field != "bold" and isinstance(filenames, list):
            if not filenames:
                raise FileNotFoundError(f"No {field} found with query: {queries[field]}")

            subj_data[field] = filenames[0]

    return layout, subj_data


def collect_run_data(layout, input_type, bold_file, cifti=False):
    """Collect data associated with a given BOLD file.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        The BIDSLayout object used to grab files from the dataset.
    bold_file : :obj:`str`
        Path to the BOLD file.
    cifti : :obj:`bool`, optional
        Whether to collect files associated with a CIFTI image (True) or a NIFTI (False).
        Default is False.
    input_type: :obj:`str`
        Input type.

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

    if not cifti and input_type not in ("hcp", "dcan"):
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
        run_data["t1w_to_native_xform"] = layout.get_nearest(
            bids_file.path,
            strict=False,
            **{"from": "T1w"},  # "from" is protected Python kw
            to="scanner",
            suffix="xfm",
        )

    elif not cifti:
        run_data["boldref"] = layout.get_nearest(
            bids_file.path,
            strict=False,
            suffix="boldref",
        )
        run_data["boldmask"] = layout.get(
            return_type="file",
            suffix="mask",
            desc="brain",
        )
        run_data["t1w_to_native_xform"] = layout.get(
            return_type="file", datatype="anat", suffix="xfm", to="MNI152NLin2009cAsym"
        )

    LOGGER.debug(
        f"Collected run data for {bold_file}:\n"
        f"{yaml.dump(run_data, default_flow_style=False, indent=4)}"
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
    fmri_dir : str
        Path to the BIDS derivative dataset being ingested.
    xcpd_dir : str
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

    info_dict = {}

    dataset_description = os.path.join(fmri_dir, "dataset_description.json")
    if os.path.isfile(dataset_description):
        with open(dataset_description) as f:
            dataset_dict = json.load(f)
    try:
        info_dict["version"] = dataset_dict["GeneratedBy"][0]["Version"]
    except:
        info_dict["version"] = "unknown"

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
    subid : str
        A subject ID (e.g., 'sub-XX' or just 'XX').

    Returns
    -------
    str
        Subject entity (e.g., 'sub-XX').
    """
    if subid.startswith("sub-"):
        return subid
    return "-".join(("sub", subid))


def _getsesid(filename):
    """Get session id from filename if available.

    Parameters
    ----------
    filename : str
        The BIDS filename from which to extract the session ID.

    Returns
    -------
    ses_id : str or None
        The session ID in the filename.
        If the file does not have a session entity, ``None`` will be returned.
    """
    ses_id = None
    base_filename = os.path.basename(filename)

    file_id = base_filename.split("_")
    for k in file_id:
        if "ses" in k:
            ses_id = k.split("-")[1]
            break

    return ses_id


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


def find_nifti_bold_files(bold_file, template_to_t1w):
    """Find nifti bold and boldref files associated with a given input file.

    Parameters
    ----------
    bold_file : str
        Path to the preprocessed BOLD file that XCPD will denoise elsewhere.
        If this is a cifti file, then the appropriate nifti file will be determined based on
        entities in this file, as well as the space and, potentially, cohort in the
        template_to_t1w file.
        When this is a nifti file, it is returned without modification.
    template_to_t1w : str
        The transform from standard space to T1w space.
        This is used to determine the volumetric template when bold_file is a cifti file.
        When bold_file is a nifti file, this is not used.

    Returns
    -------
    nifti_bold_file : str
        Path to the volumetric (nifti) preprocessed BOLD file.
    nifti_boldref_file : str
        Path to the volumetric (nifti) BOLD reference file associated with nifti_bold_file.
    """
    import glob
    import os
    import re

    # Get the nifti reference file
    if bold_file.endswith(".nii.gz"):
        nifti_bold_file = bold_file
        nifti_boldref_file = bold_file.split("desc-preproc_bold.nii.gz")[0] + "boldref.nii.gz"
        if not os.path.isfile(nifti_boldref_file):
            raise FileNotFoundError(f"boldref file not found: {nifti_boldref_file}")

    else:  # Get the cifti reference file
        # Infer the volumetric space from the transform
        nifti_template = re.findall("from-([a-zA-Z0-9+]+)", os.path.basename(template_to_t1w))[0]
        if "+" in nifti_template:
            nifti_template, cohort = nifti_template.split("+")
            search_substring = f"space-{nifti_template}_cohort-{cohort}"
        else:
            search_substring = f"space-{nifti_template}"

        bb_file_prefix = bold_file.split("space-fsLR_den-91k_bold.dtseries.nii")[0]

        # Find the appropriate _bold file.
        bold_search_str = bb_file_prefix + search_substring + "*preproc_bold.nii.gz"
        nifti_bold_file = sorted(glob.glob(bold_search_str))
        if len(nifti_bold_file) > 1:
            LOGGER.warn(f"More than one nifti bold file found: {', '.join(nifti_bold_file)}")
        elif len(nifti_bold_file) == 0:
            raise FileNotFoundError(f"bold file not found: {bold_search_str}")
        nifti_bold_file = nifti_bold_file[0]

        # Find the associated _boldref file.
        boldref_search_str = bb_file_prefix + search_substring + "*boldref.nii.gz"
        nifti_boldref_file = sorted(glob.glob(boldref_search_str))
        if len(nifti_boldref_file) > 1:
            LOGGER.warn(f"More than one nifti boldref found: {', '.join(nifti_boldref_file)}")
        elif len(nifti_boldref_file) == 0:
            raise FileNotFoundError(f"boldref file not found: {boldref_search_str}")
        nifti_boldref_file = nifti_boldref_file[0]

    return nifti_bold_file, nifti_boldref_file
