# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for converting ABCD-BIDS-format derivatives to fMRIPrep format.

These functions are specifically designed to work with abcd-hcp-pipeline version 0.1.3.
https://github.com/DCAN-Labs/abcd-hcp-pipeline/releases/tag/v0.1.3
"""
import glob
import os
import re

import nibabel as nb
import pandas as pd
from nipype import logging

from xcp_d.data import load as load_data
from xcp_d.ingression.utils import (
    collect_anatomical_files,
    collect_hcp_confounds,
    collect_meshes,
    collect_morphs,
    copy_files_in_dict,
    plot_bbreg,
    write_json,
)
from xcp_d.utils.filemanip import ensure_list

LOGGER = logging.getLogger("nipype.utils")


def convert_dcan2bids(in_dir, out_dir, participant_ids=None):
    """Convert DCAN derivatives to BIDS-compliant derivatives.

    Parameters
    ----------
    in_dir : str
        Path to DCAN derivatives.
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
    LOGGER.warning("convert_dcan2bids is an experimental function.")
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    if participant_ids is None:
        subject_folders = sorted(glob.glob(os.path.join(in_dir, "sub*")))
        subject_folders = [
            subject_folder for subject_folder in subject_folders if os.path.isdir(subject_folder)
        ]
        participant_ids = [os.path.basename(subject_folder) for subject_folder in subject_folders]
        if not participant_ids:
            raise ValueError(f"No subject found in {in_dir}")

    else:
        participant_ids = ensure_list(participant_ids)

    for subject_id in participant_ids:
        LOGGER.info(f"Processing {subject_id}")
        convert_dcan_to_bids_single_subject(
            in_dir=in_dir,
            out_dir=out_dir,
            sub_ent=subject_id,
        )

    return participant_ids


def convert_dcan_to_bids_single_subject(in_dir, out_dir, sub_ent):
    """Convert DCAN derivatives to BIDS-compliant derivatives for a single subject.

    Parameters
    ----------
    in_dir : str
        Path to the subject's DCAN derivatives.
    out_dir : str
        Path to the output BIDS-compliant derivatives folder.
    sub_ent : str
        Subject identifier, with "sub-" prefix.

    Notes
    -----
    Since the T1w is in standard space already, we use identity transforms instead of the
    individual transforms available in the DCAN derivatives.

    .. code-block::

        sub-<sub_id>
            └── ses-<ses_id>
                └── files
                    └── MNINonLinear
                        ├── Results
                        │   ├── ses-<ses_id>_task-<task_id>_run-<run_id>
                        │   │   ├── ses-<ses_id>_task-<task_id>_run-<run_id>_SBRef.nii.gz
                        │   │   ├── ses-<ses_id>_task-<task_id>_run-<run_id>.nii.gz
                        │   │   ├── ses-<ses_id>_task-<task_id>_run-<run_id>_Atlas.dtseries.nii
                        │   │   ├── Movement_Regressors.txt
                        │   │   ├── Movement_AbsoluteRMS.txt
                        │   │   └── brainmask_fs.2.0.nii.gz
                        ├── fsaverage_LR32k
                        │   ├── L.pial.32k_fs_LR.surf.gii
                        │   ├── R.pial.32k_fs_LR.surf.gii
                        │   ├── L.white.32k_fs_LR.surf.gii
                        │   ├── R.white.32k_fs_LR.surf.gii
                        │   ├── <sub_id>.L.thickness.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.R.thickness.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.L.corrThickness.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.R.corrThickness.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.L.curvature.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.R.curvature.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.L.sulc.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.R.sulc.32k_fs_LR.shape.gii
                        │   ├── <sub_id>.L.MyelinMap.32k_fs_LR.func.gii
                        │   ├── <sub_id>.R.MyelinMap.32k_fs_LR.func.gii
                        │   ├── <sub_id>.L.SmoothedMyelinMap.32k_fs_LR.func.gii
                        │   └── <sub_id>.R.SmoothedMyelinMap.32k_fs_LR.func.gii
                        ├── T1w.nii.gz
                        ├── aparc+aseg.nii.gz
                        ├── brainmask_fs.nii.gz
                        ├── ribbon.nii.gz
                        ├── vent_2mm_<sub_id>_mask_eroded.nii.gz
                        └── wm_2mm_<sub_id>_mask_eroded.nii.gz
    """
    assert isinstance(in_dir, str)
    assert os.path.isdir(in_dir), f"Folder DNE: {in_dir}"
    assert isinstance(out_dir, str)
    assert isinstance(sub_ent, str)

    sub_id = sub_ent.replace("sub-", "")
    # Reset the subject entity in case the sub- prefix wasn't included originally.
    sub_ent = f"sub-{sub_id}"

    VOLSPACE = "MNI152NLin6Asym"
    volspace_ent = f"space-{VOLSPACE}"
    RES_ENT = "res-2"

    subject_dir_bids = os.path.join(out_dir, sub_ent)
    os.makedirs(subject_dir_bids, exist_ok=True)

    # get session ids
    session_folders = sorted(glob.glob(os.path.join(in_dir, sub_ent, "s*")))
    ses_entities = [
        os.path.basename(ses_dir) for ses_dir in session_folders if os.path.isdir(ses_dir)
    ]
    if not ses_entities:
        raise FileNotFoundError(f"No session volumes found in {os.path.join(in_dir, sub_ent)}")

    dataset_description_fmriprep = os.path.join(out_dir, "dataset_description.json")
    if os.path.isfile(dataset_description_fmriprep):
        LOGGER.info("Converted dataset folder already exists. Skipping conversion.")
        return

    # A dictionary of mappings from HCP derivatives to fMRIPrep derivatives.
    # Values will be lists, to allow one-to-many mappings.
    copy_dictionary = {}

    # The identity xform is used in place of any actual ones.
    identity_xfm = str(load_data("transform/itkIdentityTransform.txt"))
    copy_dictionary[identity_xfm] = []
    morph_dict_all_ses = {}

    for ses_ent in ses_entities:
        LOGGER.info(f"Processing {ses_ent}")
        subses_ents = f"{sub_ent}_{ses_ent}"
        session_dir_fmriprep = os.path.join(subject_dir_bids, ses_ent)
        anat_dir_orig = os.path.join(in_dir, sub_ent, ses_ent, "files", "MNINonLinear")
        anat_dir_bids = os.path.join(session_dir_fmriprep, "anat")
        func_dir_orig = os.path.join(anat_dir_orig, "Results")
        func_dir_bids = os.path.join(session_dir_fmriprep, "func")
        work_dir = os.path.join(subject_dir_bids, "work")

        os.makedirs(anat_dir_bids, exist_ok=True)
        os.makedirs(func_dir_bids, exist_ok=True)
        os.makedirs(work_dir, exist_ok=True)

        # Create identity-based transforms
        t1w_to_template_fmriprep = os.path.join(
            anat_dir_bids,
            f"{subses_ents}_from-T1w_to-{VOLSPACE}_mode-image_xfm.txt",
        )
        copy_dictionary[identity_xfm].append(t1w_to_template_fmriprep)

        template_to_t1w_fmriprep = os.path.join(
            anat_dir_bids,
            f"{subses_ents}_from-{VOLSPACE}_to-T1w_mode-image_xfm.txt",
        )
        copy_dictionary[identity_xfm].append(template_to_t1w_fmriprep)

        # Collect anatomical files to copy
        base_anatomical_ents = f"{subses_ents}_{volspace_ent}_{RES_ENT}"
        anat_dict = collect_anatomical_files(
            anat_dir_orig,
            anat_dir_bids,
            base_anatomical_ents,
        )
        copy_dictionary = {**copy_dictionary, **anat_dict}

        # Collect surface files to copy
        mesh_dict = collect_meshes(anat_dir_orig, anat_dir_bids, sub_id, subses_ents)
        copy_dictionary = {**copy_dictionary, **mesh_dict}

        # Convert morphometry files
        morphometry_dict = collect_morphs(anat_dir_orig, anat_dir_bids, sub_id, subses_ents)
        morph_dict_all_ses = {**morph_dict_all_ses, **morphometry_dict}
        LOGGER.info("Finished collecting anatomical files")

        # Get masks to be used to extract confounds
        wm_mask = os.path.join(anat_dir_orig, f"wm_2mm_{sub_id}_mask_eroded.nii.gz")
        csf_mask = os.path.join(anat_dir_orig, f"vent_2mm_{sub_id}_mask_eroded.nii.gz")

        # Collect functional files to copy
        task_dirs_orig = sorted(glob.glob(os.path.join(func_dir_orig, f"{ses_ent}_task-*")))
        task_names = [os.path.basename(f) for f in task_dirs_orig if os.path.isdir(f)]

        for base_task_name in task_names:
            LOGGER.info(f"Processing {base_task_name}")
            # Names seem to follow ses-X_task-Y_run-Z format.
            found_task_info = re.findall(
                r".*_task-([0-9a-zA-Z]+[a-zA-Z]+)_run-(\d+)",
                base_task_name,
            )
            if len(found_task_info) != 1:
                LOGGER.warning(
                    f"Task name and run number could not be inferred for {base_task_name}. "
                    "Skipping."
                )
                continue

            task_id, run_id = found_task_info[0]
            task_ent = f"task-{task_id}"
            run_ent = f"run-{run_id}"

            task_dir_orig = os.path.join(func_dir_orig, base_task_name)
            func_prefix = f"{subses_ents}_{task_ent}_{run_ent}"

            # Find original task files
            sbref_orig = os.path.join(task_dir_orig, f"{base_task_name}_SBRef.nii.gz")
            boldref_fmriprep = os.path.join(
                func_dir_bids,
                f"{func_prefix}_{volspace_ent}_{RES_ENT}_boldref.nii.gz",
            )
            copy_dictionary[sbref_orig] = [boldref_fmriprep]

            bold_nifti_orig = os.path.join(task_dir_orig, f"{base_task_name}.nii.gz")
            bold_nifti_fmriprep = os.path.join(
                func_dir_bids,
                f"{func_prefix}_{volspace_ent}_{RES_ENT}_desc-preproc_bold.nii.gz",
            )
            copy_dictionary[bold_nifti_orig] = [bold_nifti_fmriprep]

            bold_cifti_orig = os.path.join(task_dir_orig, f"{base_task_name}_Atlas.dtseries.nii")
            bold_cifti_fmriprep = os.path.join(
                func_dir_bids,
                f"{func_prefix}_space-fsLR_den-91k_bold.dtseries.nii",
            )
            copy_dictionary[bold_cifti_orig] = [bold_cifti_fmriprep]

            # Extract metadata for JSON files
            bold_metadata = {
                "RepetitionTime": float(nb.load(bold_nifti_orig).header.get_zooms()[-1]),
                "TaskName": task_id,
            }
            bold_nifti_json_fmriprep = os.path.join(
                func_dir_bids,
                f"{func_prefix}_{volspace_ent}_{RES_ENT}_desc-preproc_bold.json",
            )
            write_json(bold_metadata, bold_nifti_json_fmriprep)

            bold_metadata.update(
                {
                    "grayordinates": "91k",
                    "space": "HCP grayordinates",
                    "surface": "fsLR",
                    "surface_density": "32k",
                    "volume": "MNI152NLin6Asym",
                },
            )
            bold_cifti_json_fmriprep = os.path.join(
                func_dir_bids,
                f"{func_prefix}_space-fsLR_den-91k_bold.dtseries.json",
            )
            write_json(bold_metadata, bold_cifti_json_fmriprep)

            # Create confound regressors
            collect_hcp_confounds(
                task_dir_orig=task_dir_orig,
                out_dir=func_dir_bids,
                prefix=func_prefix,
                work_dir=work_dir,
                bold_file=bold_nifti_orig,
                # This file is the anatomical brain mask downsampled to 2 mm3.
                brainmask_file=os.path.join(task_dir_orig, "brainmask_fs.2.0.nii.gz"),
                csf_mask_file=csf_mask,
                wm_mask_file=wm_mask,
            )

            # Make figures
            figdir = os.path.join(subject_dir_bids, "figures")
            os.makedirs(figdir, exist_ok=True)
            bbref_fig_fmriprep = os.path.join(
                figdir,
                f"{func_prefix}_desc-bbregister_bold.svg",
            )
            t1w = os.path.join(anat_dir_orig, "T1w.nii.gz")
            ribbon = os.path.join(anat_dir_orig, "ribbon.nii.gz")
            bbref_fig_fmriprep = plot_bbreg(
                fixed_image=t1w,
                moving_image=sbref_orig,
                out_file=bbref_fig_fmriprep,
                contour=ribbon,
            )

            LOGGER.info(f"Finished {base_task_name}")

    LOGGER.info("Finished collecting functional files")

    # Copy ABCD files to fMRIPrep folder
    LOGGER.info("Copying files")
    copy_files_in_dict(copy_dictionary)
    LOGGER.info("Finished copying files")

    # Write the dataset description out last
    dataset_description_dict = {
        "Name": "ABCD-DCAN",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "DCAN",
                "Version": "0.0.4",
                "CodeURL": "https://github.com/DCAN-Labs/abcd-hcp-pipeline",
            },
        ],
    }

    if not os.path.isfile(dataset_description_fmriprep):
        write_json(dataset_description_dict, dataset_description_fmriprep)

    # Write out the mapping from DCAN to fMRIPrep
    copy_dictionary = {**copy_dictionary, **morph_dict_all_ses}
    scans_dict = {}
    for key, values in copy_dictionary.items():
        for item in values:
            scans_dict[item] = key

    scans_tuple = tuple(scans_dict.items())
    scans_df = pd.DataFrame(scans_tuple, columns=["filename", "source_file"])
    scans_tsv = os.path.join(subject_dir_bids, f"{subses_ents}_scans.tsv")
    scans_df.to_csv(scans_tsv, sep="\t", index=False)
    LOGGER.info("Conversion completed")
