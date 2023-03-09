# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for converting HCP-format data to fMRIPrep format."""
import glob
import logging
import os

import nibabel as nb
import numpy as np
import pandas as pd
from pkg_resources import resource_filename as pkgrf

from xcp_d.utils.dcan2fmriprep import copyfileobj_example, extractreg, writejson
from xcp_d.utils.filemanip import ensure_list

LOGGER = logging.getLogger("hcp")


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
    """
    LOGGER.warning("convert_hcp2bids is an experimental function.")
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    EXCLUDE_LIST = [  # a list of folders that are not subject identifiers
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
            glob.glob(os.path.join(in_dir, "*", "*.L.BA.164k_fs_LR.label.gii"))
        )
        subject_folders = [
            subject_folder for subject_folder in subject_folders if os.path.exists(subject_folder)
        ]
        participant_ids = [os.path.basename(subject_folder) for subject_folder in subject_folders]
        all_subject_ids = []
        for subject_id in participant_ids:
            subject_id = subject_id.split(".")[0]
            if subject_id not in all_subject_ids and subject_id not in EXCLUDE_LIST:
                all_subject_ids.append("sub-" + str(subject_id))

            participant_ids = all_subject_ids

        if len(participant_ids) == 0:
            raise ValueError(f"No subject found in {in_dir}")

    else:
        participant_ids = ensure_list(participant_ids)

    for subject_id in participant_ids:
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
    """
    assert isinstance(in_dir, str)
    assert os.path.isdir(in_dir)
    assert isinstance(out_dir, str)
    assert isinstance(sub_ent, str)

    sub_id = sub_ent.replace("sub-", "")

    volspace = "MNI152NLin6Asym"
    volspace_ent = f"space-{volspace}"
    res_ent = "res-2"

    anat_dir_orig = os.path.join(in_dir, "MNINonLinear")

    subject_dir_fmriprep = os.path.join(out_dir, sub_ent)
    anat_dir_fmriprep = os.path.join(subject_dir_fmriprep, "anat")
    func_dir_fmriprep = os.path.join(subject_dir_fmriprep, "func")
    os.makedirs(anat_dir_fmriprep, exist_ok=True)
    os.makedirs(func_dir_fmriprep, exist_ok=True)

    # Get necessary files
    csf_mask = pkgrf("xcp_d", f"/data/masks/{volspace_ent}_{res_ent}_label-CSF_mask.nii.gz")
    wm_mask = pkgrf("xcp_d", f"/data/masks/{volspace_ent}_{res_ent}_label-WM_mask.nii.gz")

    # A dictionary of mappings from HCP derivatives to fMRIPrep derivatives.
    # Values will be lists, to allow one-to-many mappings.
    copy_dictionary = {}

    # Collect anatomical files to copy
    t1w_orig = os.path.join(anat_dir_orig, "T1w_restore.nii.gz")
    t1w_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_ent}_{volspace_ent}_{res_ent}_desc-preproc_T1w.nii.gz",
    )
    copy_dictionary[t1w_orig] = [t1w_fmriprep]

    brainmask_orig = os.path.join(anat_dir_orig, "brainmask_fs.nii.gz")
    brainmask_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_ent}_{volspace_ent}_{res_ent}_desc-brain_mask.nii.gz",
    )
    copy_dictionary[brainmask_orig] = [brainmask_fmriprep]

    # NOTE: What is this file for?
    ribbon_orig = os.path.join(anat_dir_orig, "ribbon.nii.gz")
    ribbon_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_ent}_{volspace_ent}_{res_ent}_desc-ribbon_T1w.nii.gz",
    )
    copy_dictionary[ribbon_orig] = [ribbon_fmriprep]

    dseg_orig = os.path.join(anat_dir_orig, "aparc.a2009s+aseg.nii.gz")
    dseg_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_ent}_{volspace_ent}_{res_ent}_desc-aparcaseg_dseg.nii.gz",
    )
    copy_dictionary[dseg_orig] = [dseg_fmriprep]

    # Grab transforms
    identity_xfm = pkgrf("xcp_d", "/data/transform/itkIdentityTranform.txt")
    t1w_to_template_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_ent}_from-T1w_to-{volspace}_mode-image_xfm.txt",
    )
    copy_dictionary[identity_xfm] = [t1w_to_template_fmriprep]

    template_to_t1w_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_ent}_from-{volspace}_to-T1w_mode-image_xfm.txt",
    )
    copy_dictionary[identity_xfm].append(template_to_t1w_fmriprep)

    # Grab surface morphometry files
    fsaverage_dir_orig = os.path.join(anat_dir_orig, "fsaverage_LR32k")

    SURFACE_DICT = {
        "R.midthickness": "hemi-R_desc-hcp_midthickness",
        "L.midthickness": "hemi-L_desc-hcp_midthickness",
        "R.inflated": "hemi-R_desc-hcp_inflated",
        "L.inflated": "hemi-L_desc-hcp_inflated",
        "R.very_inflated": "hemi-R_desc-hcp_vinflated",
        "L.very_inflated": "hemi-L_desc-hcp_vinflated",
        "R.pial": "hemi-R_pial",
        "L.pial": "hemi-L_pial",
        "R.white": "hemi-R_smoothwm",
        "L.white": "hemi-L_smoothwm",
    }
    for in_str, out_str in SURFACE_DICT.items():
        surf_orig = os.path.join(
            fsaverage_dir_orig,
            f"{sub_id}.{in_str}.32k_fs_LR.surf.gii",
        )
        surf_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_ent}_space-fsLR_den-32k_{out_str}.surf.gii",
        )
        copy_dictionary[surf_orig] = [surf_fmriprep]

    print("finished collecting anat files")

    # Collect functional files to copy
    subject_task_folders = sorted(glob.glob(os.path.join(in_dir, "*", "Results", "*")))
    subject_task_folders = [
        task for task in subject_task_folders if task.endswith("RL") or task.endswith("LR")
    ]
    for subject_task_folder in subject_task_folders:
        # NOTE: What is the first element in the folder name?
        _, task_id, dir_id = os.path.basename(subject_task_folder).split("_")
        task_ent = f"task-{task_id}"
        dir_ent = f"dir-{dir_id}"
        # TODO: Rename variable
        run_foldername = os.path.basename(subject_task_folder)

        # Find original task files
        brainmask_orig_temp = os.path.join(subject_task_folder, "brainmask_fs.2.nii.gz")

        bold_nifti_orig = os.path.join(subject_task_folder, f"{run_foldername}.nii.gz")
        bold_nifti_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_ent}_{task_ent}_{dir_ent}_{volspace_ent}_{res_ent}_desc-preproc_bold.nii.gz",
        )
        copy_dictionary[bold_nifti_orig] = [bold_nifti_fmriprep]

        boldref_orig = os.path.join(subject_task_folder, "SBRef_dc.nii.gz")
        boldref_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_ent}_{task_ent}_{dir_ent}_{volspace_ent}_{res_ent}_boldref.nii.gz",
        )
        copy_dictionary[boldref_orig] = [boldref_fmriprep]

        bold_cifti_orig = os.path.join(
            subject_task_folder,
            f"{run_foldername}_Atlas_MSMAll.dtseries.nii",
        )
        bold_cifti_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_ent}_{task_ent}_{dir_ent}_space-fsLR_den-91k_bold.dtseries.nii",
        )
        copy_dictionary[bold_cifti_orig] = [bold_cifti_fmriprep]

        # Grab transforms

        native_to_t1w_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_ent}_{task_ent}_{dir_ent}_from-scanner_to-T1w_mode-image_xfm.txt",
        )
        copy_dictionary[identity_xfm].append(native_to_t1w_fmriprep)

        t1w_to_native_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_ent}_{task_ent}_{dir_ent}_from-T1w_to-scanner_mode-image_xfm.txt",
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
            f"{sub_ent}_{task_ent}_{dir_ent}_{volspace_ent}_{res_ent}_desc-preproc_bold.json",
        )
        writejson(bold_nifti_json_dict, bold_nifti_json_fmriprep)

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
            f"{sub_ent}_{task_ent}_{dir_ent}_space-fsLR_den-91k_bold.dtseries.json",
        )
        writejson(bold_cifti_json_dict, bold_cifti_json_fmriprep)

        # Create confound regressors
        mvreg = pd.read_csv(
            os.path.join(subject_task_folder, "Movement_Regressors.txt"),
            header=None,
            delimiter=r"\s+",
        )
        mvreg.columns = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "trans_x_derivative1",
            "trans_y_derivative1",
            "trans_z_derivative1",
            "rot_x_derivative1",
            "rot_y_derivative1",
            "rot_z_derivative1",
        ]
        # convert rotations from degrees to radians
        rot_columns = [c for c in mvreg.columns if c.startswith("rot")]
        for col in rot_columns:
            mvreg[col] = mvreg[col] * np.pi / 180

        # set first row of derivative columns to nan, for fMRIPrep compatibility
        deriv_columns = [c for c in mvreg.columns if c.endswith("derivative1")]
        for col in deriv_columns:
            mvreg.loc[0, col] = None

        # get powers
        columns = mvreg.columns.tolist()
        for col in columns:
            mvreg[f"{col}_power2"] = mvreg[col] ** 2

        # use masks: brain, csf, and wm mask to extract timeseries
        gsreg = extractreg(mask=brainmask_orig_temp, nifti=bold_nifti_orig)
        csfreg = extractreg(mask=csf_mask, nifti=bold_nifti_orig)
        wmreg = extractreg(mask=wm_mask, nifti=bold_nifti_orig)
        rmsd = np.loadtxt(os.path.join(subject_task_folder, "Movement_AbsoluteRMS.txt"))

        brainreg = pd.DataFrame(
            {"global_signal": gsreg, "white_matter": wmreg, "csf": csfreg, "rmsd": rmsd}
        )

        # get derivatives and powers
        brainreg["global_signal_derivative1"] = brainreg["global_signal"].diff()
        brainreg["white_matter_derivative1"] = brainreg["white_matter"].diff()
        brainreg["csf_derivative1"] = brainreg["csf"].diff()

        brainreg["global_signal_derivative1_power2"] = brainreg["global_signal_derivative1"] ** 2
        brainreg["global_signal_power2"] = brainreg["global_signal"] ** 2

        brainreg["white_matter_derivative1_power2"] = brainreg["white_matter_derivative1"] ** 2
        brainreg["white_matter_power2"] = brainreg["white_matter"] ** 2

        brainreg["csf_derivative1_power2"] = brainreg["csf_derivative1"] ** 2
        brainreg["csf_power2"] = brainreg["csf"] ** 2

        # Merge the two DataFrames
        regressors = pd.concat([mvreg, brainreg], axis=1)

        # write out the confounds
        regressors_file_base = f"{sub_ent}_{task_ent}_{dir_ent}_desc-confounds_timeseries"
        regressors_tsv_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{regressors_file_base}.tsv",
        )
        regressors.to_csv(regressors_tsv_fmriprep, index=False, sep="\t", na_rep="n/a")

        # NOTE: Is this JSON any good?
        regressors_json_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{regressors_file_base}.json",
        )
        regressors.to_json(regressors_json_fmriprep)

    print("finished collecting func files")

    # Copy HCP files to fMRIPrep folder
    for file_orig, files_fmriprep in copy_dictionary.items():
        if not isinstance(files_fmriprep, list):
            raise ValueError(
                f"Entry for {file_orig} should be a list, but is a {type(files_fmriprep)}"
            )

        if len(files_fmriprep) > 1:
            print(f"File used for more than one output: {file_orig}")

        for file_fmriprep in files_fmriprep:
            copyfileobj_example(file_orig, file_fmriprep)
    print("finished copying files")

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
        writejson(dataset_description_dict, dataset_description_fmriprep)

    # Write out the mapping from HCP to fMRIPrep
    scans_dict = {}
    for key, values in copy_dictionary.items():
        for item in values:
            scans_dict[item] = key

    scans_tuple = tuple(scans_dict.items())
    scans_df = pd.DataFrame(scans_tuple, columns=["filename", "source_file"])
    scans_tsv = os.path.join(subject_dir_fmriprep, f"{sub_ent}_scans.tsv")
    scans_df.to_csv(scans_tsv, sep="\t", index=False)
