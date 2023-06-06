# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for converting HCP-format data to fMRIPrep format."""
import glob
import os

import nibabel as nb
import pandas as pd
from nipype import logging
from pkg_resources import resource_filename as pkgrf

from xcp_d.utils.filemanip import ensure_list
from xcp_d.utils.ingestion import (
    collect_anatomical_files,
    collect_confounds,
    collect_surfaces,
    copy_file,
    plot_bbreg,
    write_json,
)

LOGGER = logging.getLogger("nipype.utils")


def convert_hcp2bids(in_dir, out_dir, participant_ids=None):
    """Convert HCP derivatives to BIDS-compliant derivatives.

    Parameters
    ----------
    in_dir : str
        Path to HCP derivatives.
    out_dir : str
        Path to the output BIDS-compliant derivatives folder.
    participant_ids : None or list of str
        List of participant IDs to run conversion on.
        The participant IDs must not have the "sub-" prefix.
        If None, the function will search for all subjects in ``in_dir`` and convert all of them.

    Returns
    -------
    participant_ids : list of str
        The list of subjects whose derivatives were converted.

    Notes
    -----
    Since the T1w is in standard space already, we use identity transforms instead of the
    individual transforms available in the DCAN derivatives.
    """
    LOGGER.warning("convert_hcp2bids is an experimental function.")
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    # a list of folders that are not subject identifiers
    EXCLUDE_LIST = [
        "BiasField",
        "Native",
        "ROIs",
        "Results",
        "T1w",
        "T1w_restore",
        "T1w_restore_brain",
        "T2w",
        "T2w_restore",
        "T2w_restore_brain",
        "aparc",
        "aparc.a2009s+aseg",
        "brainmask_fs",
        "fsaverage_LR32k",
        "ribbon",
        "wmparc",
        "xfms",
    ]

    if participant_ids is None:
        subject_folders = sorted(
            glob.glob(os.path.join(in_dir, "*", "*", "*", "*R.pial.32k_fs_LR.surf.gii"))
        )
        subject_folders = [
            subject_folder for subject_folder in subject_folders if os.path.exists(subject_folder)
        ]
        participant_ids = [os.path.basename(subject_folder) for subject_folder in subject_folders]
        all_subject_ids = []
        for subject_id in participant_ids:
            subject_id = subject_id.split(".")[0]
            if subject_id not in all_subject_ids and subject_id not in EXCLUDE_LIST:
                all_subject_ids.append(f"sub-{subject_id}")

            participant_ids = all_subject_ids

        if len(participant_ids) == 0:
            raise ValueError(f"No subject found in {in_dir}")

    else:
        participant_ids = ensure_list(participant_ids)

    for subject_id in participant_ids:
        LOGGER.info(f"Converting {subject_id}")
        convert_hcp_to_bids_single_subject(
            in_dir=in_dir,
            out_dir=out_dir,
            sub_ent=subject_id,
        )

    return participant_ids


def convert_hcp_to_bids_single_subject(in_dir, out_dir, sub_ent):
    """Convert HCP derivatives to BIDS-compliant derivatives for a single subject.

    Parameters
    ----------
    in_dir : str
        Path to the subject's HCP derivatives.
    out_dir : str
        Path to the output fMRIPrep-style derivatives folder.
    sub_ent : str
        Subject identifier, with "sub-" prefix.

    Notes
    -----
    Since the T1w is in standard space already, we use identity transforms instead of the
    individual transforms available in the DCAN derivatives.
    """
    assert isinstance(in_dir, str)
    assert os.path.isdir(in_dir)
    assert isinstance(out_dir, str)
    assert isinstance(sub_ent, str)

    sub_id = sub_ent.replace("sub-", "")
    # Reset the subject entity in case the sub- prefix wasn't included originally.
    sub_ent = f"sub-{sub_id}"
    subses_ents = sub_ent

    VOLSPACE = "MNI152NLin6Asym"
    volspace_ent = f"space-{VOLSPACE}"
    RES_ENT = "res-2"

    anat_dir_orig = os.path.join(in_dir, sub_id, "MNINonLinear")
    subject_dir_fmriprep = os.path.join(out_dir, sub_ent)
    anat_dir_fmriprep = os.path.join(subject_dir_fmriprep, "anat")
    func_dir_fmriprep = os.path.join(subject_dir_fmriprep, "func")
    work_dir = os.path.join(subject_dir_fmriprep, "work")

    if os.path.isdir(func_dir_fmriprep):
        LOGGER.info("Converted dataset already exists. Skipping conversion.")
        return

    os.makedirs(anat_dir_fmriprep, exist_ok=True)
    os.makedirs(func_dir_fmriprep, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Get necessary files
    csf_mask = pkgrf("xcp_d", f"/data/masks/{volspace_ent}_{RES_ENT}_label-CSF_mask.nii.gz")
    wm_mask = pkgrf("xcp_d", f"/data/masks/{volspace_ent}_{RES_ENT}_label-WM_mask.nii.gz")

    # A dictionary of mappings from HCP derivatives to fMRIPrep derivatives.
    # Values will be lists, to allow one-to-many mappings.
    copy_dictionary = {}

    # The identity xform is used in place of any actual ones.
    identity_xfm = pkgrf("xcp_d", "/data/transform/itkIdentityTransform.txt")
    copy_dictionary[identity_xfm] = []

    t1w_to_template_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{subses_ents}_from-T1w_to-{VOLSPACE}_mode-image_xfm.txt",
    )
    copy_dictionary[identity_xfm].append(t1w_to_template_fmriprep)

    template_to_t1w_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{subses_ents}_from-{VOLSPACE}_to-T1w_mode-image_xfm.txt",
    )
    copy_dictionary[identity_xfm].append(template_to_t1w_fmriprep)

    # Collect anatomical files to copy
    base_anatomical_ents = f"{subses_ents}_{volspace_ent}_{RES_ENT}"
    anat_dict = collect_anatomical_files(anat_dir_orig, anat_dir_fmriprep, base_anatomical_ents)
    copy_dictionary = {**copy_dictionary, **anat_dict}

    # Grab surface morphometry files
    surfaces_dict = collect_surfaces(anat_dir_orig, anat_dir_fmriprep, sub_id, subses_ents)
    copy_dictionary = {**copy_dictionary, **surfaces_dict}
    LOGGER.info("Finished collecting anatomical files")

    # Collect functional files to copy
    subject_task_folders = sorted(
        glob.glob(os.path.join(in_dir, sub_id, "MNINonLinear", "Results", "*"))
    )
    subject_task_folders = [
        task for task in subject_task_folders if task.endswith("RL") or task.endswith("LR")
    ]
    for subject_task_folder in subject_task_folders:
        LOGGER.info(f"Processing {subject_task_folder}")
        # NOTE: What is the first element in the folder name?
        _, task_id, dir_id = os.path.basename(subject_task_folder).split("_")
        task_ent = f"task-{task_id}"
        dir_ent = f"dir-{dir_id}"
        # TODO: Rename variable
        run_foldername = os.path.basename(subject_task_folder)

        # Find original task files
        bold_nifti_orig = os.path.join(subject_task_folder, f"{run_foldername}.nii.gz")
        bold_nifti_fmriprep = os.path.join(
            func_dir_fmriprep,
            (
                f"{subses_ents}_{task_ent}_{dir_ent}_{volspace_ent}_{RES_ENT}_"
                "desc-preproc_bold.nii.gz"
            ),
        )
        copy_dictionary[bold_nifti_orig] = [bold_nifti_fmriprep]

        sbref_orig = os.path.join(subject_task_folder, "SBRef_dc.nii.gz")
        boldref_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{subses_ents}_{task_ent}_{dir_ent}_{volspace_ent}_{RES_ENT}_boldref.nii.gz",
        )
        copy_dictionary[sbref_orig] = [boldref_fmriprep]

        bold_cifti_orig = os.path.join(
            subject_task_folder,
            f"{run_foldername}_Atlas_MSMAll.dtseries.nii",
        )
        bold_cifti_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{subses_ents}_{task_ent}_{dir_ent}_space-fsLR_den-91k_bold.dtseries.nii",
        )
        copy_dictionary[bold_cifti_orig] = [bold_cifti_fmriprep]

        # More transforms
        native_to_t1w_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{subses_ents}_{task_ent}_{dir_ent}_from-scanner_to-T1w_mode-image_xfm.txt",
        )
        copy_dictionary[identity_xfm].append(native_to_t1w_fmriprep)

        t1w_to_native_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{subses_ents}_{task_ent}_{dir_ent}_from-T1w_to-scanner_mode-image_xfm.txt",
        )
        copy_dictionary[identity_xfm].append(t1w_to_native_fmriprep)

        # Extract metadata for JSON files
        TR = nb.load(bold_nifti_orig).header.get_zooms()[-1]  # repetition time
        bold_nifti_json_dict = {
            "RepetitionTime": float(TR),
            "TaskName": task_id,
        }
        bold_nifti_json_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{subses_ents}_{task_ent}_{dir_ent}_{volspace_ent}_{RES_ENT}_desc-preproc_bold.json",
        )
        write_json(bold_nifti_json_dict, bold_nifti_json_fmriprep)

        bold_cifti_json_dict = {
            "RepetitionTime": float(TR),
            "TaskName": task_id,
            "grayordinates": "91k",
            "space": "HCP grayordinates",
            "surface": "fsLR",
            "surface_density": "32k",
            "volume": "MNI152NLin6Asym",
        }
        bold_cifti_json_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{subses_ents}_{task_ent}_{dir_ent}_space-fsLR_den-91k_bold.dtseries.json",
        )
        write_json(bold_cifti_json_dict, bold_cifti_json_fmriprep)

        # Create confound regressors
        base_task_ents = f"{subses_ents}_{task_ent}_{dir_ent}"
        brainmask_orig_temp = os.path.join(subject_task_folder, "brainmask_fs.2.nii.gz")
        collect_confounds(
            subject_task_folder,
            func_dir_fmriprep,
            base_task_ents,
            work_dir=work_dir,
            bold_file=bold_nifti_orig,
            brainmask_file=brainmask_orig_temp,
            csf_mask_file=csf_mask,
            wm_mask_file=wm_mask,
        )

        # Make figures
        figdir = os.path.join(subject_dir_fmriprep, "figures")
        os.makedirs(figdir, exist_ok=True)
        bbref_fig_fmriprep = os.path.join(
            figdir,
            f"{subses_ents}_{task_ent}_{dir_ent}_desc-bbregister_bold.svg",
        )
        t1w = os.path.join(anat_dir_orig, "T1w.nii.gz")
        ribbon = os.path.join(anat_dir_orig, "ribbon.nii.gz")
        bbref_fig_fmriprep = plot_bbreg(
            fixed_image=t1w,
            moving_image=sbref_orig,
            out_file=bbref_fig_fmriprep,
            contour=ribbon,
        )

        LOGGER.info(f"Finished {subject_task_folder}")

    LOGGER.info("Finished collecting functional files")

    # Copy HCP files to fMRIPrep folder
    LOGGER.info("Copying files")
    for file_orig, files_fmriprep in copy_dictionary.items():
        if not isinstance(files_fmriprep, list):
            raise ValueError(
                f"Entry for {file_orig} should be a list, but is a {type(files_fmriprep)}"
            )

        if len(files_fmriprep) > 1:
            LOGGER.warning(f"File used for more than one output: {file_orig}")

        for file_fmriprep in files_fmriprep:
            copy_file(file_orig, file_fmriprep)

    LOGGER.info("Finished copying files")

    # Write the dataset description out last
    dataset_description_dict = {
        "Name": "HCP",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "HCP",
                "Version": "unknown",
                "CodeURL": "https://github.com/Washington-University/HCPpipelines",
            },
        ],
    }
    dataset_description_fmriprep = os.path.join(out_dir, "dataset_description.json")
    if not os.path.isfile(dataset_description_fmriprep):
        write_json(dataset_description_dict, dataset_description_fmriprep)

    # Write out the mapping from HCP to fMRIPrep
    scans_dict = {}
    for key, values in copy_dictionary.items():
        for item in values:
            scans_dict[item] = key

    scans_tuple = tuple(scans_dict.items())
    scans_df = pd.DataFrame(scans_tuple, columns=["filename", "source_file"])
    scans_tsv = os.path.join(subject_dir_fmriprep, f"{subses_ents}_scans.tsv")
    scans_df.to_csv(scans_tsv, sep="\t", index=False)
    LOGGER.info("Conversion completed")
