# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Utilities for fmriprep bids derivatives and layout.

Most of the code is copied from niworkflows.
A PR will be submitted to niworkflows at some point.
"""
import fnmatch
import os
import warnings

from bids import BIDSLayout
from nipype import logging

LOGGER = logging.getLogger("nipype.interface")


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


def collect_participants(
    bids_dir, participant_label=None, strict=False, bids_validate=False
):
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
    participant_label = [
        sub[4:] if sub.startswith("sub-") else sub for sub in participant_label
    ]
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


def collect_data(
    bids_dir,
    participant_label,
    task=None,
    bids_validate=False,
    bids_filters=None,
):
    """Collect data from a BIDS dataset.

    Parameters
    ----------
    bids_dir
    participant_label
    task
    bids_validate
    bids_filters

    Returns
    -------
    layout : pybids.layout.BIDSLayout
    subj_data : dict
    """
    layout = BIDSLayout(str(bids_dir), validate=bids_validate, derivatives=True)

    queries = {
        "regfile": {"datatype": "anat", "suffix": "xfm"},
        "boldfile": {"datatype": "func", "suffix": "bold"},
        "t1w": {"datatype": "anat", "suffix": "T1w"},
        "seg_data": {"datatype": "anat", "suffix": "dseg"},
        "pial": {"datatype": "anat", "suffix": "pial"},
        "wm": {"datatype": "anat", "suffix": "smoothwm"},
        "midthickness": {"datatype": "anat", "suffix": "midthickness"},
        "inflated": {"datatype": "anat", "suffix": "inflated"},
    }

    bids_filters = bids_filters or {}
    for acq, entities in bids_filters.items():
        queries[acq].update(entities)

    if task:
        # queries["preproc_bold"]["task"] = task
        queries["boldfile"]["task"] = task

    subj_data = {
        dtype: sorted(
            layout.get(
                return_type="file",
                subject=participant_label,
                extension=["nii", "nii.gz", "dtseries.nii", "h5", "gii"],
                **query,
            )
        )
        for dtype, query in queries.items()
    }

    # reg_file = select_registrationfile(subj_data,template=template)

    # bold_file= select_cifti_bold(subj_data)

    return layout, subj_data


def select_registrationfile(subj_data):
    """Select a registration file from a derivatives dataset.

    Parameters
    ----------
    subj_data : dict
        Dictionary where keys are filetypes and values are filenames.

    Returns
    -------
    mni_to_t1w : str
        Path to the MNI-to-T1w transform file.
    t1w_to_mni : str
        Path to the T1w-to-MNI transform file.
    """
    regfile = subj_data["regfile"]

    # get the file with the template name
    template1 = "MNI152NLin6Asym"  # default for fmriprep / nibabies with cifti output
    template2 = "MNI152NLin2009cAsym"  # default template for fmriprep,dcan and hcp
    template3 = "MNIInfant"  # nibabies

    mni_to_t1w = None
    t1w_to_mni = None

    for j in regfile:
        if (
            "from-" + template1 in j
            or ("from-" + template2 in j and mni_to_t1w is None)
            or ("from-" + template3 in j and mni_to_t1w is None)
        ):
            mni_to_t1w = j
        elif (
            "to-" + template1 in j
            or ("to-" + template2 in j and t1w_to_mni is None)
            or ("to-" + template3 in j and t1w_to_mni is None)
        ):
            t1w_to_mni = j
    # for validation, we need to check presence of MNI152NLin2009cAsym
    # if not we use MNI152NLin2006cAsym for nibabies
    # print(mni_to_t1w)

    return mni_to_t1w, t1w_to_mni


def select_cifti_bold(subj_data):
    """Split list of preprocessed fMRI files into bold (volumetric) and cifti.

    Parameters
    ----------
    subj_data

    Returns
    -------
    bold_file : list of str
        List of paths to preprocessed BOLD files.
    cifti_file : list of str
        List of paths to preprocessed BOLD CIFTI files.
    """
    boldfile = subj_data["boldfile"]
    bold_files = []
    cifti_files = []

    for file_ in boldfile:
        if "preproc_bold" in file_:
            bold_files.append(file_)

        elif "bold.dtseries.nii" in file_:
            cifti_files.append(file_)

    return bold_files, cifti_files


def extract_t1w_seg(subj_data):
    """Select preprocessed T1w and segmentation files.

    Parameters
    ----------
    subj_data : dict

    Returns
    -------
    selected_t1w_file : str
        Preprocessed T1-weighted file.
    selected_t1w_seg_file : str
        Segmentation file.
    """
    selected_t1w_file, selected_t1w_seg_file = None, None
    for t1w_file in subj_data["t1w"]:
        t1w_filename = os.path.basename(t1w_file)
        if not fnmatch.fnmatch(t1w_filename, "*_space-*"):
            selected_t1w_file = t1w_file

    for t1w_seg_file in subj_data["seg_data"]:
        t1w_seg_filename = os.path.basename(t1w_seg_file)
        if not (
            fnmatch.fnmatch(t1w_seg_filename, "*_space-*")
            or fnmatch.fnmatch(t1w_seg_filename, "*aseg*")
        ):
            selected_t1w_seg_file = t1w_seg_file

    if not selected_t1w_file:
        raise ValueError("No segmentation file found.")

    if not selected_t1w_seg_file:
        raise ValueError("No segmentation file found.")

    return selected_t1w_file, selected_t1w_seg_file


def write_dataset_description(fmri_dir, xcpd_dir):
    import json

    from xcp_d import __version__

    orig_dset_description = os.path.join(fmri_dir, "dataset_description.json")
    if not os.path.isfile(orig_dset_description):
        dset_desc = {}

    else:
        with open(orig_dset_description, "r") as fo:
            dset_desc = json.load(fo)

        assert dset_desc["DatasetType"] == "derivative"

    # Update dataset description
    dset_desc["Name"] = "xcpd"
    generated_by = dset_desc.get("GeneratedBy", [])
    generated_by.insert(
        0,
        {
            "Name": "xcpd",
            "Version": __version__,
            "CodeURL": f"https://github.com/PennLINC/xcp_d/archive/{__version__}.tar.gz"
        },
    )
    dset_desc["GeneratedBy"] = generated_by
    dset_desc["HowToAcknowledge"] = "Include the generated boilerplate in the methods section."

    xcpd_dset_description = os.path.join(xcpd_dir, "dataset_description.json")
    if os.path.isfile(xcpd_dset_description):
        # check hash of file
        pass

    else:
        with open(xcpd_dset_description, "w") as fo:
            json.dump(dset_desc, fo, indent=4, sort_keys=True)
