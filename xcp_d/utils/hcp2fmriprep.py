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

LOGGER = logging.getLogger("hcp")


def hcp2fmriprep(hcpdir, outdir, participant_ids=None):
    """Convert HCP-format data to fMRIPrep format."""
    LOGGER.warning("This is an experimental function and has not been tested yet.")
    hcpdir = os.path.abspath(hcpdir)
    outdir = os.path.abspath(outdir)

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
        "aparc+aseg",
        "brainmask_fs",
        "fsaverage_LR32k",
        "ribbon",
        "wmparc",
        "xfms",
    ]

    if participant_ids is None:
        subject_folders = sorted(glob.glob(os.path.join(hcpdir, "*")))
        subject_folders = [
            subject_folder for subject_folder in subject_folders if os.path.isdir(subject_folder)
        ]
        participant_ids = [os.path.basename(subject_folder) for subject_folder in subject_folders]
        all_subject_ids = []
        for subject_id in participant_ids:
            subject_id = subject_id.split(".")[0]
            if subject_id not in all_subject_ids and subject_id not in EXCLUDE_LIST:
                all_subject_ids.append(subject_id)

            participant_ids = all_subject_ids

        if len(participant_ids) == 0:
            raise ValueError(f"No subject found in {hcpdir}")

        if len(participant_ids) > 0:
            for subject_id in participant_ids:
                convert_hcp_to_fmriprep_single_subject(
                    in_dir=hcpdir,
                    out_dir=outdir,
                    sub_id=subject_id,
                )

    else:
        convert_hcp_to_fmriprep_single_subject(
            in_dir=hcpdir,
            out_dir=outdir,
            sub_id=participant_ids,
        )

    return participant_ids


def convert_hcp_to_fmriprep_single_subject(in_dir, out_dir, sub_id):
    """Do the internal work for hcp2fmriprep."""
    assert isinstance(in_dir, str)
    assert os.path.isdir(in_dir)
    assert isinstance(out_dir, str)
    assert isinstance(sub_id, str)

    # make new directory for anat and func
    if not sub_id.startswith("sub-"):
        sub_id = f"sub-{sub_id}"

    subject_dir_fmriprep = os.path.join(out_dir, sub_id)
    anat_dir_fmriprep = os.path.join(subject_dir_fmriprep, "anat")
    func_dir_fmriprep = os.path.join(subject_dir_fmriprep, "func")
    os.makedirs(anat_dir_fmriprep, exist_ok=True)
    os.makedirs(func_dir_fmriprep, exist_ok=True)

    # A dictionary of mappings from HCP derivatives to fMRIPrep derivatives.
    # Values will be lists, to allow one-to-many mappings.
    copy_dictionary = {}

    # get old files
    t1w_orig = os.path.join(in_dir, "T1w_restore.nii.gz")
    brainmask_orig = os.path.join(in_dir, "brainmask_fs.nii.gz")
    ribbon_orig = os.path.join(in_dir, "ribbon.nii.gz")
    dseg_orig = os.path.join(in_dir, "aparc+aseg.nii.gz")

    fsaverage_dir_orig = os.path.join(in_dir, "fsaverage_LR32k")

    # NOTE: Why glob? Do we not know the full filenames? Are there multiple files?
    rh_midthickness_orig = glob.glob(
        os.path.join(fsaverage_dir_orig, "*R.midthickness.32k_fs_LR.surf.gii")
    )[0]
    lh_midthickness_orig = glob.glob(
        os.path.join(fsaverage_dir_orig, "*L.midthickness.32k_fs_LR.surf.gii")
    )[0]
    rh_inflated_orig = glob.glob(
        os.path.join(fsaverage_dir_orig, "*R.inflated.32k_fs_LR.surf.gii")
    )[0]
    lh_inflated_orig = glob.glob(
        os.path.join(fsaverage_dir_orig, "*L.inflated.32k_fs_LR.surf.gii")
    )[0]

    rh_pial_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*R.pial.32k_fs_LR.surf.gii"))[0]
    lh_pial_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*L.pial.32k_fs_LR.surf.gii"))[0]

    rh_wm_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*R.white.32k_fs_LR.surf.gii"))[0]
    lh_wm_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*L.white.32k_fs_LR.surf.gii"))[0]

    # to fmriprep directory
    t1w_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_desc-preproc_T1w.nii.gz")
    copy_dictionary[t1w_orig] = [t1w_fmriprep]
    dseg_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_dseg.nii.gz")
    copy_dictionary[dseg_orig] = [dseg_fmriprep]
    ribbon_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_desc-ribbon_T1w.nii.gz")
    copy_dictionary[ribbon_orig] = [ribbon_fmriprep]
    brainmask_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_desc-brain_mask.nii.gz")
    copy_dictionary[brainmask_orig] = [brainmask_fmriprep]
    t1w_to_template_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.txt",
    )
    copy_dictionary[t1w_orig].append(t1w_to_template_fmriprep)
    template_to_t1w_fmriprep = os.path.join(
        anat_dir_fmriprep,
        f"{sub_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.txt",
    )
    copy_dictionary[t1w_orig].append(template_to_t1w_fmriprep)

    rh_midthickness_fmriprep = os.path.join(
        anat_dir_fmriprep, f"{sub_id}_hemi-R_midthickness.surf.gii"
    )
    copy_dictionary[rh_midthickness_orig] = [rh_midthickness_fmriprep]
    lh_midthickness_fmriprep = os.path.join(
        anat_dir_fmriprep, f"{sub_id}_hemi-L_midthickness.surf.gii"
    )
    copy_dictionary[lh_midthickness_orig] = [lh_midthickness_fmriprep]
    rh_inflated_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_hemi-R_inflated.surf.gii")
    copy_dictionary[rh_inflated_orig] = [rh_inflated_fmriprep]
    lh_inflated_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_hemi-L_inflated.surf.gii")
    copy_dictionary[lh_inflated_orig] = [lh_inflated_fmriprep]

    rh_pial_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_hemi-R_pial.surf.gii")
    copy_dictionary[rh_pial_orig] = [rh_pial_fmriprep]
    lh_pial_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_hemi-L_pial.surf.gii")
    copy_dictionary[lh_pial_orig] = [lh_pial_fmriprep]

    rh_wm_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_hemi-R_smoothwm.surf.gii")
    copy_dictionary[rh_wm_orig] = [rh_wm_fmriprep]
    lh_wm_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_hemi-L_smoothwm.surf.gii")
    copy_dictionary[lh_wm_orig] = [lh_wm_fmriprep]

    print("finished converting anat files")

    # get the task files
    subject_task_folders = sorted(glob.glob(os.path.join(in_dir, "Results", "*")))
    subject_task_folders = [task for task in subject_task_folders if task.endswith(["RL", "LR"])]

    csf_mask = pkgrf("xcp_d", "/data/masks/csf.nii.gz")
    wm_mask = pkgrf("xcp_d", "/data/masks/wm.nii.gz")
    for subject_task_folder in subject_task_folders:
        _, task_name, acq_label = os.path.basename(subject_task_folder).split("_")
        filenamex = os.path.basename(subject_task_folder)

        # Find original task files
        bold_nifti_orig = os.path.join(subject_task_folder, f"{filenamex}.nii.gz")
        brainmask_orig_temp = os.path.join(subject_task_folder, "brainmask_fs.2.nii.gz")
        boldref_orig = os.path.join(subject_task_folder, "SBRef_dc.nii.gz")
        bold_cifti_orig = os.path.join(
            subject_task_folder, f"{filenamex}_Atlas_MSMAll.dtseries.nii"
        )

        # Construct fMRIPrep task filenames
        boldref_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_id}_task-{task_name}_acq-{acq_label}_space-MNI152NLin6Asym_boldref.nii.gz",
        )
        copy_dictionary[boldref_orig] = [boldref_fmriprep]

        bold_cifti_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_id}_task-{task_name}_acq-{acq_label}_space-fsLR_den-91k_bold.dtseries.nii",
        )
        copy_dictionary[bold_cifti_orig] = [bold_cifti_fmriprep]

        # Extract metadata for JSON files
        TR = nb.load(bold_nifti_orig).header.get_zooms()[-1]  # repetition time
        bold_nifti_json_dict = {
            "RepetitionTime": float(TR),
            "TaskName": task_name,
        }
        bold_nifti_json_fmriprep = os.path.join(
            func_dir_fmriprep,
            (
                f"{sub_id}_task-{task_name}_acq-{acq_label}_"
                "space-MNI152NLin6Asym_desc-preproc_bold.json"
            ),
        )
        writejson(bold_nifti_json_dict, bold_nifti_json_fmriprep)

        bold_cifti_json_dict = {
            "grayordinates": "91k",
            "space": "HCP grayordinates",
            "surface": "fsLR",
            "surface_density": "32k",
            "volume": "MNI152NLin6Asym",
        }
        bold_cifti_json_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{sub_id}_task-{task_name}_acq-{acq_label}_space-fsLR_den-91k_bold.dtseries.json",
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
        mvreg["rot_x"] = mvreg["rot_x"] * np.pi / 180
        mvreg["rot_y"] = mvreg["rot_y"] * np.pi / 180
        mvreg["rot_z"] = mvreg["rot_z"] * np.pi / 180
        mvreg["rot_x_derivative1"] = mvreg["rot_x_derivative1"] * np.pi / 180
        mvreg["rot_y_derivative1"] = mvreg["rot_y_derivative1"] * np.pi / 180
        mvreg["rot_z_derivative1"] = mvreg["rot_z_derivative1"] * np.pi / 180

        # get derivatives and powers
        mvreg["trans_x_power2"] = mvreg["trans_x"] ** 2
        mvreg["trans_x_derivative1_power2"] = mvreg["trans_x_derivative1"] ** 2
        mvreg["rot_x_power2"] = mvreg["rot_x"] ** 2
        mvreg["rot_x_derivative1_power2"] = mvreg["rot_x_derivative1"] ** 2
        mvreg["trans_y_power2"] = mvreg["trans_y"] ** 2
        mvreg["trans_y_derivative1_power2"] = mvreg["trans_y_derivative1"] ** 2
        mvreg["rot_y_power2"] = mvreg["rot_y"] ** 2
        mvreg["rot_y_derivative1_power2"] = mvreg["rot_y_derivative1"] ** 2
        mvreg["trans_z_power2"] = mvreg["trans_z"] ** 2
        mvreg["trans_z_derivative1_power2"] = mvreg["trans_z_derivative1"] ** 2
        mvreg["rot_z_power2"] = mvreg["rot_z"] ** 2
        mvreg["rot_z_derivative1_power2"] = mvreg["rot_z_derivative1"] ** 2

        # use masks: brain,csf and wm mask to extract timeseries
        gsreg = extractreg(mask=brainmask_orig_temp, nifti=bold_nifti_orig)
        csfreg = extractreg(mask=csf_mask, nifti=bold_nifti_orig)
        wmreg = extractreg(mask=wm_mask, nifti=bold_nifti_orig)

        rsmd = np.loadtxt(os.path.join(subject_task_folder, "Movement_AbsoluteRMS.txt"))
        brainreg = pd.DataFrame(
            {"global_signal": gsreg, "white_matter": wmreg, "csf": csfreg, "rmsd": rsmd}
        )

        # get derivatives and powers
        regressors = pd.concat([mvreg, brainreg], axis=1)
        regressors["global_signal_derivative1"] = pd.DataFrame(
            np.diff(regressors["global_signal"].to_numpy(), prepend=0)
        )
        regressors["global_signal_derivative1_power2"] = (
            regressors["global_signal_derivative1"] ** 2
        )
        regressors["global_signal_power2"] = regressors["global_signal"] ** 2

        regressors["white_matter_derivative1"] = pd.DataFrame(
            np.diff(regressors["white_matter"].to_numpy(), prepend=0)
        )
        regressors["white_matter_derivative1_power2"] = regressors["white_matter_derivative1"] ** 2
        regressors["white_matter_power2"] = regressors["white_matter"] ** 2
        regressors["csf_power2"] = regressors["csf"] ** 2
        regressors["csf_derivative1"] = pd.DataFrame(
            np.diff(regressors["csf"].to_numpy(), prepend=0)
        )
        regressors["csf_derivative1_power2"] = regressors["csf_derivative1"] ** 2

        # write out the json
        regressors_file_base = (
            f"{sub_id}_task-{task_name}_acq-{acq_label}_desc-confounds_timeseries"
        )
        regressors_tsv_fmriprep = os.path.join(
            func_dir_fmriprep,
            f"{regressors_file_base}.tsv",
        )
        regressors.to_csv(regressors_tsv_fmriprep, index=False, sep="\t")

        # NOTE: Is this JSON any good?
        regressors_json_fmriprep = os.path.join(func_dir_fmriprep, f"{regressors_file_base}.json")
        regressors.to_json(regressors_json_fmriprep)

    # Copy HCP files to fMRIPrep folder
    for file_orig, files_fmriprep in copy_dictionary.items():
        if not isinstance(files_fmriprep):
            raise ValueError(
                f"Entry for {file_orig} should be a list, but is a {type(files_fmriprep)}"
            )

        for file_fmriprep in files_fmriprep:
            copyfileobj_example(file_orig, file_fmriprep)

    # Write the dataset description out last
    # TODO: Add "unknown" version to dictionary.
    dataset_description_dict = {
        "Name": "HCP",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {"Name": "HCP"},
        ],
    }
    dataset_description_fmriprep = os.path.join(out_dir, "dataset_description.json")
    writejson(dataset_description_dict, dataset_description_fmriprep)

    # Write out the mapping from HCP to fMRIPrep
    mapping_fmriprep = os.path.join(subject_dir_fmriprep, "file_mapping.json")
    writejson(mapping_fmriprep, copy_dictionary)
