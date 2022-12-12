# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions for converting DCAN-format derivatives to fMRIPrep format."""
import glob
import json
import logging
import os
import re

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.input_data import NiftiMasker

from xcp_d.utils.filemanip import ensure_list

LOGGER = logging.getLogger("dcan")


def dcan2fmriprep(in_dir, out_dir, participant_ids=None):
    """Loop over all subjects in dcan dir and convert to fmriprep format."""
    LOGGER.warning("This is an experimental function.")
    in_dir = os.path.abspath(in_dir)
    out_dir = os.path.abspath(out_dir)

    if participant_ids is None:
        subject_folders = sorted(glob.glob(os.path.join(in_dir, "sub*")))
        subject_folders = [
            subject_folder for subject_folder in subject_folders if os.path.isdir(subject_folder)
        ]
        participant_ids = [os.path.basename(subject_folder) for subject_folder in subject_folders]
        if len(participant_ids) == 0:
            raise ValueError(f"No subject found in {in_dir}")

    else:
        participant_ids = ensure_list(participant_ids)

    for subject_id in participant_ids:
        convert_dcan_to_fmriprep_single_subject(
            in_dir=in_dir,
            out_dir=out_dir,
            sub_id=subject_id,
        )

    return participant_ids


def convert_dcan_to_fmriprep_single_subject(in_dir, out_dir, sub_id):
    """Convert dcan data to fmriprep format."""
    assert isinstance(in_dir, str)
    assert os.path.isdir(in_dir)
    assert isinstance(out_dir, str)
    assert isinstance(sub_id, str)

    subject_dir_fmriprep = os.path.join(out_dir, sub_id)

    # get session ids
    session_folders = sorted(glob.glob(os.path.join(in_dir, sub_id, "s*")))
    session_folders = [
        os.path.basename(ses_dir) for ses_dir in session_folders if os.path.isdir(ses_dir)
    ]
    # NOTE: Why split ses- out if you add it right back in?
    session_ids = [ses_dir.split("ses-")[1] for ses_dir in session_folders]
    session_ids = [f"ses-{session_id}" for session_id in session_ids]

    # A dictionary of mappings from HCP derivatives to fMRIPrep derivatives.
    # Values will be lists, to allow one-to-many mappings.
    copy_dictionary = {}

    for ses_id in session_ids:
        session_dir_fmriprep = os.path.join(subject_dir_fmriprep, ses_id)

        anat_dir_orig = os.path.join(in_dir, sub_id, ses_id, "files", "MNINonLinear")
        anat_dir_fmriprep = os.path.join(session_dir_fmriprep, "anat")
        os.makedirs(anat_dir_fmriprep, exist_ok=True)

        # NOTE: Why *was* this set to the *first* session only? (I fixed it)
        # AFAICT, this would copy the first session's files from DCAN into *every*
        # session of the output directory.
        func_dir_orig = os.path.join(anat_dir_orig, "Results")
        func_dir_fmriprep = os.path.join(session_dir_fmriprep, "func")
        os.makedirs(func_dir_fmriprep, exist_ok=True)

        # Collect anatomical files to copy
        t1w_orig = os.path.join(anat_dir_orig, "T1w.nii.gz")
        t1w_fmriprep = os.path.join(
            anat_dir_fmriprep, f"{sub_id}_{ses_id}_desc-preproc_T1w.nii.gz"
        )
        copy_dictionary[t1w_orig] = [t1w_fmriprep]

        # NOTE: We're using the T1w image as a transform. This doesn't make sense.
        t1w_to_template_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5",
        )
        copy_dictionary[t1w_orig].append(t1w_to_template_fmriprep)

        # NOTE: We're using the T1w image as a transform. This doesn't make sense.
        template_to_t1w_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5",
        )
        copy_dictionary[t1w_orig].append(template_to_t1w_fmriprep)

        brainmask_orig = os.path.join(anat_dir_orig, "brainmask_fs.nii.gz")
        brainmask_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_desc-brain_mask.nii.gz",
        )
        copy_dictionary[brainmask_orig] = [brainmask_fmriprep]

        ribbon_orig = os.path.join(anat_dir_orig, "ribbon_orig.nii.gz")
        ribbon_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_desc-ribbon_orig.nii.gz",
        )
        copy_dictionary[ribbon_orig] = [ribbon_fmriprep]

        dseg_orig = os.path.join(anat_dir_orig, "aparc+aseg.nii.gz")
        dseg_fmriprep = os.path.join(anat_dir_fmriprep, f"{sub_id}_{ses_id}_dseg.nii.gz")
        copy_dictionary[dseg_orig] = [dseg_fmriprep]

        fsaverage_dir_orig = os.path.join(anat_dir_orig, "fsaverage_LR32k")

        # NOTE: Why glob? Do we not know the full filenames? Are there multiple files?
        rh_midthickness_orig = glob.glob(
            os.path.join(fsaverage_dir_orig, "*R.midthickness.32k_fs_LR.surf.gii")
        )[0]
        rh_midthickness_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_hemi-R_midthickness.surf.gii",
        )
        copy_dictionary[rh_midthickness_orig] = [rh_midthickness_fmriprep]

        lh_midthickness_orig = glob.glob(
            os.path.join(fsaverage_dir_orig, "*L.midthickness.32k_fs_LR.surf.gii")
        )[0]
        lh_midthickness_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_hemi-L_midthickness.surf.gii",
        )
        copy_dictionary[lh_midthickness_orig] = [lh_midthickness_fmriprep]

        rh_inflated_orig = glob.glob(
            os.path.join(fsaverage_dir_orig, "*R.inflated.32k_fs_LR.surf.gii")
        )[0]
        rh_inflated_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_hemi-R_inflated.surf.gii",
        )
        copy_dictionary[rh_inflated_orig] = [rh_inflated_fmriprep]

        lh_inflated_orig = glob.glob(
            os.path.join(fsaverage_dir_orig, "*L.inflated.32k_fs_LR.surf.gii")
        )[0]
        lh_inflated_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_hemi-L_inflated.surf.gii",
        )
        copy_dictionary[lh_inflated_orig] = [lh_inflated_fmriprep]

        rh_pial_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*R.pial.32k_fs_LR.surf.gii"))[0]
        rh_pial_fmriprep = os.path.join(
            anat_dir_fmriprep, f"{sub_id}_{ses_id}_hemi-R_pial.surf.gii"
        )
        copy_dictionary[rh_pial_orig] = [rh_pial_fmriprep]

        lh_pial_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*L.pial.32k_fs_LR.surf.gii"))[0]
        lh_pial_fmriprep = os.path.join(
            anat_dir_fmriprep, f"{sub_id}_{ses_id}_hemi-L_pial.surf.gii"
        )
        copy_dictionary[lh_pial_orig] = [lh_pial_fmriprep]

        rh_wm_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*R.white.32k_fs_LR.surf.gii"))[0]
        rh_wm_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_hemi-R_smoothwm.surf.gii",
        )
        copy_dictionary[rh_wm_orig] = [rh_wm_fmriprep]

        lh_wm_orig = glob.glob(os.path.join(fsaverage_dir_orig, "*L.white.32k_fs_LR.surf.gii"))[0]
        lh_wm_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{sub_id}_{ses_id}_hemi-L_smoothwm.surf.gii",
        )
        copy_dictionary[lh_wm_orig] = [lh_wm_fmriprep]

        print("finished collecting anat files")

        # get masks and transforms
        # NOTE: Why glob? Do we not know the full filenames? Are there multiple files?
        wmmask = glob.glob(os.path.join(anat_dir_orig, "wm_2mm_*_mask_eroded.nii.gz"))[0]
        csfmask = glob.glob(os.path.join(anat_dir_orig, "vent_2mm_*_mask_eroded.nii.gz"))[0]

        # NOTE: We're using the white matter mask as a transform. This doesn't make sense.
        t1w_to_native_orig = wmmask
        copy_dictionary[t1w_to_native_orig] = []

        # Collect functional files to copy
        task_dirs_orig = sorted(glob.glob(os.path.join(func_dir_orig, "task-*")))
        task_dirs_orig = [task_dir for task_dir in task_dirs_orig if os.path.isdir(task_dir)]
        task_names = [os.path.basename(task_dir).split("-")[1] for task_dir in task_dirs_orig]

        for task_name in task_names:
            task_id = f"task-{task_name}"
            task_dir_orig = os.path.join(func_dir_orig, task_id)

            # NOTE: Why are there regular expressions?
            taskname = re.split(r"(\d+)", task_name)[0]
            run_id = "run-" + str(int(re.split(r"(\d+)", task_name)[1]))

            # Find original task files
            brainmask_orig_temp = os.path.join(task_dir_orig, "brainmask_fs.2.0.nii.gz")

            sbref_orig = os.path.join(task_dir_orig, f"{task_id}_SBRef.nii.gz")
            boldref_fmriprep = os.path.join(
                func_dir_fmriprep,
                f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_space-MNI152NLin6Asym_boldref.nii.gz",
            )
            copy_dictionary[sbref_orig] = [boldref_fmriprep]

            bold_nifti_orig = os.path.join(task_dir_orig, f"{task_id}.nii.gz")
            bold_nifti_fmriprep = os.path.join(
                func_dir_fmriprep,
                (
                    f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_"
                    "space-MNI152NLin6Asym_desc-preproc_bold.nii.gz"
                ),
            )
            copy_dictionary[bold_nifti_orig] = [bold_nifti_fmriprep]

            bold_cifti_orig = os.path.join(task_dir_orig, f"{task_id}_Atlas.dtseries.nii")
            bold_cifti_fmriprep = os.path.join(
                func_dir_fmriprep,
                f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_space-fsLR_den-91k_bold.dtseries.nii",
            )
            copy_dictionary[bold_cifti_orig] = [bold_cifti_fmriprep]

            # NOTE: We're using the white matter mask as a transform. This doesn't make sense.
            native_to_t1w_fmriprep = os.path.join(
                func_dir_fmriprep,
                (
                    f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_"
                    "from-scanner_to-T1w_mode-image_xfm.txt"
                ),
            )
            copy_dictionary[t1w_to_native_orig].append(native_to_t1w_fmriprep)

            t1w_to_native_fmriprep = os.path.join(
                func_dir_fmriprep,
                (
                    f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_"
                    "from-T1w_to-scanner_mode-image_xfm.txt"
                ),
            )
            copy_dictionary[t1w_to_native_orig].append(t1w_to_native_fmriprep)

            # Extract metadata for JSON files
            TR = nb.load(bold_nifti_orig).header.get_zooms()[-1]  # repetition time
            bold_nifti_json_dict = {
                "RepetitionTime": np.float(TR),
                "TaskName": taskname,
            }

            bold_cifti_json_dict = {
                "grayordinates": "91k",
                "space": "HCP grayordinates",
                "surface": "fsLR",
                "surface_density": "32k",
                "bold_nifti_orig": "MNI152NLin6Asym",
            }

            bold_nifti_json_fmriprep = os.path.join(
                func_dir_fmriprep,
                (
                    f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_"
                    "space-MNI152NLin6Asym_desc-preproc_bold.json"
                ),
            )

            bold_cifti_json_fmriprep = os.path.join(
                func_dir_fmriprep,
                (
                    f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_"
                    "space-fsLR_den-91k_bold.dtseries.json"
                ),
            )

            writejson(bold_nifti_json_dict, bold_nifti_json_fmriprep)
            writejson(bold_cifti_json_dict, bold_cifti_json_fmriprep)

            # Create confound regressors
            mvreg = pd.read_csv(
                os.path.join(task_dir_orig, "Movement_Regressors.txt"),
                header=None,
                delimiter=r"\s+",
            )

            # Only use the first six columns
            mvreg = mvreg.iloc[:, 0:6]
            mvreg.columns = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

            # convert rotations from degrees to radians
            rot_columns = [c for c in mvreg.columns if c.startswith("rot")]
            for col in rot_columns:
                mvreg[col] = mvreg[col] * np.pi / 180

            # get derivatives of motion columns
            columns = mvreg.columns.tolist()
            for col in columns:
                mvreg[f"{col}_derivative1"] = mvreg[col].diff()

            # get powers
            columns = mvreg.columns.tolist()
            for col in columns:
                mvreg[f"{col}_power2"] = mvreg[col] ** 2

            # use masks: brain, csf, and wm mask to extract timeseries
            gsreg = extractreg(mask=brainmask_orig_temp, nifti=bold_nifti_orig)
            csfreg = extractreg(mask=csfmask, nifti=bold_nifti_orig)
            wmreg = extractreg(mask=wmmask, nifti=bold_nifti_orig)
            rsmd = np.loadtxt(os.path.join(task_dir_orig, "Movement_AbsoluteRMS.txt"))

            brainreg = pd.DataFrame(
                {"global_signal": gsreg, "white_matter": wmreg, "csf": csfreg, "rmsd": rsmd}
            )

            # get derivatives and powers
            brainreg["global_signal_derivative1"] = brainreg["global_signal"].diff()
            brainreg["white_matter_derivative1"] = brainreg["white_matter"].diff()
            brainreg["csf_derivative1"] = brainreg["csf"].diff()

            brainreg["global_signal_derivative1_power2"] = (
                brainreg["global_signal_derivative1"] ** 2
            )
            brainreg["global_signal_power2"] = brainreg["global_signal"] ** 2

            brainreg["white_matter_derivative1_power2"] = brainreg["white_matter_derivative1"] ** 2
            brainreg["white_matter_power2"] = brainreg["white_matter"] ** 2

            brainreg["csf_derivative1_power2"] = brainreg["csf_derivative1"] ** 2
            brainreg["csf_power2"] = brainreg["csf"] ** 2

            # Merge the two DataFrames
            regressors = pd.concat([mvreg, brainreg], axis=1)

            # write out the confounds
            regressors_file_base = (
                f"{sub_id}_{ses_id}_task-{task_name}_{run_id}_desc-confounds_timeseries"
            )
            regressors_tsv_fmriprep = os.path.join(
                func_dir_fmriprep,
                f"{regressors_file_base}.tsv",
            )
            regressors.to_csv(regressors_tsv_fmriprep, sep="\t", index=False)

            # NOTE: Is this JSON any good?
            regressors_json_fmriprep = os.path.join(
                func_dir_fmriprep,
                f"{regressors_file_base}.json",
            )
            writejson(bold_cifti_json_dict, regressors_json_fmriprep)

            # Make figures
            figdir = os.path.join(subject_dir_fmriprep, "figures")
            os.makedirs(figdir, exist_ok=True)
            bbref_fig_fmriprep = os.path.join(
                figdir,
                f"{sub_id}_{ses_id}_task-{taskname}_{run_id}_desc-bbregister_bold.svg",
            )
            bbref_fig_fmriprep = bbregplot(
                fixed_image=t1w_orig,
                moving_image=boldref_fmriprep,
                out_file=bbref_fig_fmriprep,
                contour=ribbon_orig,
            )

    # Copy HCP files to fMRIPrep folder
    for file_orig, files_fmriprep in copy_dictionary.items():
        if not isinstance(files_fmriprep):
            raise ValueError(
                f"Entry for {file_orig} should be a list, but is a {type(files_fmriprep)}"
            )

        for file_fmriprep in files_fmriprep:
            copyfileobj_example(file_orig, file_fmriprep)

    dataset_description_dict = {
        "Name": "ABCDDCAN",
        "BIDSVersion": "1.4.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "DCAN",
                "Version": "0.0.4",
                "CodeURL": "https://github.com/DCAN-Labs/abcd-hcp-pipeline",
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
    scans_tsv = os.path.join(subject_dir_fmriprep, f"{sub_id}_scans.tsv")
    scans_df.to_csv(scans_tsv, sep="\t", index=False)


def copyfileobj_example(src, dst):
    """Copy a file from source to dest.

    source and dest must be file-like objects,
    i.e. any object with a read or write method, like for example StringIO.
    """
    import filecmp
    import shutil

    if not os.path.exists(dst) or not filecmp.cmp(src, dst):
        shutil.copyfile(src, dst)


def extractreg(mask, nifti):
    """Extract mean signal within mask from NIFTI."""
    masker = NiftiMasker(mask_img=mask)
    signals = masker.fit_transform(nifti)
    return np.mean(signals, axis=1)


def writejson(data, outfile):
    """Write dictionary to JSON file."""
    with open(outfile, "w") as f:
        json.dump(data, f)
    return outfile


def bbregplot(fixed_image, moving_image, contour, out_file="report.svg"):
    """Plot bbref_fig_fmriprep results."""
    import numpy as np
    from nilearn.image import load_img, resample_img, threshold_img
    from niworkflows.viz.utils import compose_view, cuts_from_bbox, plot_registration

    fixed_image_nii = load_img(fixed_image)
    moving_image_nii = load_img(moving_image)
    moving_image_nii = resample_img(
        moving_image_nii, target_affine=np.eye(3), interpolation="nearest"
    )
    contour_nii = load_img(contour) if contour is not None else None

    mask_nii = threshold_img(fixed_image_nii, 1e-3)

    n_cuts = 7
    if contour_nii:
        cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
    else:
        cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

    compose_view(
        plot_registration(
            fixed_image_nii,
            "fixed-image",
            estimate_brightness=True,
            cuts=cuts,
            label="fixed",
            contour=contour_nii,
            compress="auto",
        ),
        plot_registration(
            moving_image_nii,
            "moving-image",
            estimate_brightness=True,
            cuts=cuts,
            label="moving",
            contour=contour_nii,
            compress="auto",
        ),
        out_file=out_file,
    )
    return out_file
