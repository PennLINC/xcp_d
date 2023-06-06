# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Functions to support ingestion of non-BIDS preprocessing derivatives."""
import json
import os

import numpy as np
from nilearn import image, maskers
from nipype import logging

LOGGER = logging.getLogger("nipype.utils")


def copy_file(src, dst):
    """Copy a file from source to dest.

    source and dest must be file-like objects,
    i.e. any object with a read or write method, like for example StringIO.
    """
    import filecmp
    import shutil

    if not os.path.exists(dst) or not filecmp.cmp(src, dst):
        shutil.copyfile(src, dst)


def collect_anatomical_files(anat_dir_orig, anat_dir_fmriprep, base_anatomical_ents):
    """Collect anatomical files from ABCD or HCP-YA derivatives."""
    ANAT_DICT = {
        # XXX: Why have T1w here and T1w_restore for HCP?
        "T1w.nii.gz": "desc-preproc_T1w.nii.gz",
        "brainmask_fs.nii.gz": "desc-brain_mask.nii.gz",
        "ribbon.nii.gz": "desc-ribbon_T1w.nii.gz",
        "aparc+aseg.nii.gz": "desc-aparcaseg_dseg.nii.gz",
    }
    copy_dictionary = {}

    for in_str, out_str in ANAT_DICT.items():
        anat_orig = os.path.join(anat_dir_orig, in_str)
        anat_fmriprep = os.path.join(anat_dir_fmriprep, f"{base_anatomical_ents}_{out_str}")
        if os.path.isfile(anat_orig):
            copy_dictionary[anat_orig] = [anat_fmriprep]

    return copy_dictionary


def collect_surfaces(anat_dir_orig, anat_dir_fmriprep, sub_id, subses_ents):
    """Collect surface files from ABCD or HCP-YA derivatives."""
    SURFACE_DICT = {
        "R.midthickness.32k_fs_LR.surf.gii": "hemi-R_desc-hcp_midthickness.surf.gii",
        "L.midthickness.32k_fs_LR.surf.gii": "hemi-L_desc-hcp_midthickness.surf.gii",
        "R.inflated.32k_fs_LR.surf.gii": "hemi-R_desc-hcp_inflated.surf.gii",
        "L.inflated.32k_fs_LR.surf.gii": "hemi-L_desc-hcp_inflated.surf.gii",
        "R.very_inflated.32k_fs_LR.surf.gii": "hemi-R_desc-hcp_vinflated.surf.gii",
        "L.very_inflated.32k_fs_LR.surf.gii": "hemi-L_desc-hcp_vinflated.surf.gii",
        "R.pial.32k_fs_LR.surf.gii": "hemi-R_pial.surf.gii",
        "L.pial.32k_fs_LR.surf.gii": "hemi-L_pial.surf.gii",
        "R.white.32k_fs_LR.surf.gii": "hemi-R_smoothwm.surf.gii",
        "L.white.32k_fs_LR.surf.gii": "hemi-L_smoothwm.surf.gii",
        "R.corrThickness.32k_fs_LR.shape.gii": "hemi-R_thickness.shape.gii",
        "L.corrThickness.32k_fs_LR.shape.gii": "hemi-L_thickness.shape.gii",
        "R.curvature.32k_fs_LR.shape.gii": "hemi-R_curv.shape.gii",
        "L.curvature.32k_fs_LR.shape.gii": "hemi-L_curv.shape.gii",
        "R.sulc.32k_fs_LR.shape.gii": "hemi-R_sulc.shape.gii",
        "L.sulc.32k_fs_LR.shape.gii": "hemi-L_sulc.shape.gii",
    }

    fsaverage_dir_orig = os.path.join(anat_dir_orig, "fsaverage_LR32k")
    copy_dictionary = {}
    for in_str, out_str in SURFACE_DICT.items():
        surf_orig = os.path.join(fsaverage_dir_orig, f"{sub_id}.{in_str}")
        surf_fmriprep = os.path.join(
            anat_dir_fmriprep,
            f"{subses_ents}_space-fsLR_den-32k_{out_str}",
        )
        if os.path.isfile(surf_orig):
            copy_dictionary[surf_orig] = [surf_fmriprep]

    return copy_dictionary


def collect_confounds(
    task_dir_orig,
    func_dir_fmriprep,
    base_task_ents,
    work_dir,
    bold_file,
    brainmask_file,
    csf_mask_file,
    wm_mask_file,
):
    """Create confound regressors."""
    import pandas as pd

    mvreg_file = os.path.join(task_dir_orig, "Movement_Regressors.txt")
    rmsd_file = os.path.join(task_dir_orig, "Movement_AbsoluteRMS.txt")

    mvreg = pd.read_csv(mvreg_file, header=None, delimiter=r"\s+")

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

    # Use dummy column for framewise displacement, which will be recalculated by XCP-D.
    mvreg["framewise_displacement"] = 0

    # use masks: brain, csf, and wm mask to extract timeseries
    mean_gs = extract_mean_signal(
        mask=brainmask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    mean_csf = extract_mean_signal(
        mask=csf_mask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    mean_wm = extract_mean_signal(
        mask=wm_mask_file,
        nifti=bold_file,
        work_dir=work_dir,
    )
    rsmd = np.loadtxt(rmsd_file)

    brainreg = pd.DataFrame(
        {"global_signal": mean_gs, "white_matter": mean_wm, "csf": mean_csf, "rmsd": rsmd}
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
    confounds_df = pd.concat([mvreg, brainreg], axis=1)

    # write out the confounds
    regressors_tsv_fmriprep = os.path.join(
        func_dir_fmriprep,
        f"{base_task_ents}_desc-confounds_timeseries.tsv",
    )
    confounds_df.to_csv(regressors_tsv_fmriprep, sep="\t", index=False)

    # NOTE: Is this JSON any good?
    regressors_json_fmriprep = os.path.join(
        func_dir_fmriprep,
        f"{base_task_ents}_desc-confounds_timeseries.json",
    )
    confounds_df.to_json(regressors_json_fmriprep)


def extract_mean_signal(mask, nifti, work_dir):
    """Extract mean signal within mask from NIFTI."""
    assert os.path.isfile(mask), f"File DNE: {mask}"
    assert os.path.isfile(nifti), f"File DNE: {nifti}"
    masker = maskers.NiftiMasker(mask_img=mask, memory=work_dir, memory_level=5)
    signals = masker.fit_transform(nifti)
    return np.mean(signals, axis=1)


def write_json(data, outfile):
    """Write dictionary to JSON file."""
    with open(outfile, "w") as f:
        json.dump(data, f)
    return outfile


def plot_bbreg(fixed_image, moving_image, contour, out_file="report.svg"):
    """Plot bbref_fig_fmriprep results."""
    import numpy as np
    from niworkflows.viz.utils import compose_view, cuts_from_bbox, plot_registration

    fixed_image_nii = image.load_img(fixed_image)
    moving_image_nii = image.load_img(moving_image)
    moving_image_nii = image.resample_img(
        moving_image_nii, target_affine=np.eye(3), interpolation="nearest"
    )
    contour_nii = image.load_img(contour) if contour is not None else None

    mask_nii = image.threshold_img(fixed_image_nii, 1e-3)

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


def copy_files_in_dict(copy_dictionary):
    """Copy files in dictionary."""
    for file_orig, files_fmriprep in copy_dictionary.items():
        if not isinstance(files_fmriprep, list):
            raise ValueError(
                f"Entry for {file_orig} should be a list, but is a {type(files_fmriprep)}"
            )

        if len(files_fmriprep) > 1:
            LOGGER.warning(f"File used for more than one output: {file_orig}")

        for file_fmriprep in files_fmriprep:
            copy_file(file_orig, file_fmriprep)
