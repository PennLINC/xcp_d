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


def hcp2fmriprep(hcpdir, outdir, sub_id=None):
    """Convert HCP-format data to fMRIPrep format."""
    LOGGER.warning("This is an experimental function and has not been tested yet.")
    hcpdir = os.path.abspath(hcpdir)
    outdir = os.path.abspath(outdir)
    if sub_id is None:
        sub_idir = glob.glob(hcpdir + "/*")
        sub_id = [os.path.basename(j) for j in sub_idir]
        subjects = []
        x_list = [
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
        for item in sub_id:
            item = item.split(".")[0]
            if item not in subjects and item not in x_list:
                subjects.append(item)
            sub_id = subjects
        if len(sub_id) == 0:
            raise ValueError(f"No subject found in {hcpdir}")
        if len(sub_id) > 0:
            for j in sub_id:
                hcpfmriprepx(hcp_dir=hcpdir, out_dir=outdir, sub_id=j)
    else:
        hcpfmriprepx(hcp_dir=hcpdir, out_dir=outdir, sub_id=str(sub_id))

    return sub_id


def hcpfmriprepx(hcp_dir, out_dir, sub_id):
    """Do the internal work for hcp2fmriprep."""
    anat_dirx = hcp_dir
    # make new directory for anat and func

    anatdir = out_dir + "/sub-" + sub_id + "/anat/"
    funcdir = out_dir + "/sub-" + sub_id + "/func/"
    os.makedirs(anatdir, exist_ok=True)
    os.makedirs(funcdir, exist_ok=True)

    # get old files
    tw1 = anat_dirx + "/T1w_restore.nii.gz"
    brainmask = anat_dirx + "/brainmask_fs.nii.gz"
    ribbon = anat_dirx + "/ribbon.nii.gz"
    segm = anat_dirx + "/aparc+aseg.nii.gz"

    midR = glob.glob(anat_dirx + "/fsaverage_LR32k/*R.midthickness.32k_fs_LR.surf.gii")[0]
    midL = glob.glob(anat_dirx + "/fsaverage_LR32k/*L.midthickness.32k_fs_LR.surf.gii")[0]
    infR = glob.glob(anat_dirx + "/fsaverage_LR32k/*R.inflated.32k_fs_LR.surf.gii")[0]
    infL = glob.glob(anat_dirx + "/fsaverage_LR32k/*L.inflated.32k_fs_LR.surf.gii")[0]

    pialR = glob.glob(anat_dirx + "/fsaverage_LR32k/*R.pial.32k_fs_LR.surf.gii")[0]
    pialL = glob.glob(anat_dirx + "/fsaverage_LR32k/*L.pial.32k_fs_LR.surf.gii")[0]

    whiteR = glob.glob(anat_dirx + "/fsaverage_LR32k/*R.white.32k_fs_LR.surf.gii")[0]
    whiteL = glob.glob(anat_dirx + "/fsaverage_LR32k/*L.white.32k_fs_LR.surf.gii")[0]

    hcpfiles = [
        tw1,
        segm,
        ribbon,
        brainmask,
        tw1,
        tw1,
        midL,
        midR,
        pialL,
        pialR,
        whiteL,
        whiteR,
        infL,
        infR,
    ]

    # to fmriprep directory
    if "sub" not in sub_id:
        subid = "sub-" + sub_id
    t1wim = anatdir + subid + "_desc-preproc_T1w.nii.gz"
    t1seg = anatdir + subid + "_dseg.nii.gz"
    t1ribbon = anatdir + subid + "_desc-ribbon_T1w.nii.gz"
    t1brainm = anatdir + subid + "_desc-brain_mask.nii.gz"
    regfile1 = anatdir + subid + "_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.txt"
    regfile2 = anatdir + subid + "_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.txt"

    lMid = anatdir + subid + "_hemi-L_midthickness.surf.gii"
    rMid = anatdir + subid + "_hemi-R_midthickness.surf.gii"

    lpial = anatdir + subid + "_hemi-L_pial.surf.gii"
    rpial = anatdir + subid + "_hemi-R_pial.surf.gii"

    lwhite = anatdir + subid + "_hemi-L_smoothwm.surf.gii"
    rwhite = anatdir + subid + "_hemi-R_smoothwm.surf.gii"

    linf = anatdir + subid + "_hemi-L_inflated.surf.gii"
    rinf = anatdir + subid + "_hemi-R_inflated.surf.gii"
    newanatfiles = [
        t1wim,
        t1seg,
        t1ribbon,
        t1brainm,
        regfile1,
        regfile2,
        lMid,
        rMid,
        lpial,
        rpial,
        lwhite,
        rwhite,
        linf,
        rinf,
    ]

    for i, j in zip(hcpfiles, newanatfiles):
        copyfileobj_example(i, j)
    print("finished converting anat files")
    # get the task files

    taskx = glob.glob(hcp_dir + "/Results/*")
    tasklistx = []
    print(hcp_dir)
    for j in taskx:
        if j.endswith("RL") or j.endswith("LR"):
            tasklistx.append(j)

    csf_mask = pkgrf("xcp_d", "/data/masks/csf.nii.gz")
    wm_mask = pkgrf("xcp_d", "/data/masks/wm.nii.gz")
    for k in tasklistx:
        idx = os.path.basename(k).split("_")
        filenamex = os.path.basename(k)

        # create confound regressors

        mvreg = pd.read_csv(k + "/Movement_Regressors.txt", header=None, delimiter=r"\s+")
        mvreg = mvreg.iloc[:, :]
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
        # convert rot to rad
        mvreg["rot_x"] = mvreg["rot_x"] * np.pi / 180
        mvreg["rot_y"] = mvreg["rot_y"] * np.pi / 180
        mvreg["rot_z"] = mvreg["rot_z"] * np.pi / 180
        mvreg["rot_x_derivative1"] = mvreg["rot_x_derivative1"] * np.pi / 180
        mvreg["rot_y_derivative1"] = mvreg["rot_y_derivative1"] * np.pi / 180
        mvreg["rot_z_derivative1"] = mvreg["rot_z_derivative1"] * np.pi / 180
        mvreg["trans_x_power2"] = mvreg["trans_x"] ** 2

        # get derivatives and powers
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
        bolddata = k + "/" + filenamex + ".nii.gz"
        brainmask = k + "/brainmask_fs.2.nii.gz"
        gsreg = extractreg(mask=brainmask, nifti=bolddata)
        csfreg = extractreg(mask=csf_mask, nifti=bolddata)
        wmreg = extractreg(mask=wm_mask, nifti=bolddata)

        rsmd = np.loadtxt(k + "/Movement_AbsoluteRMS.txt")
        brainreg = pd.DataFrame(
            {"global_signal": gsreg, "white_matter": wmreg, "csf": csfreg, "rmsd": rsmd}
        )

        # get derivatives and powers
        regressors = pd.concat([mvreg, brainreg], axis=1)
        regressors["global_signal_derivative1"] = pd.DataFrame(
            np.diff(regressors["global_signal"].tonumpy(), prepend=0)
        )
        regressors["global_signal_derivative1_power2"] = (
            regressors["global_signal_derivative1"] ** 2
        )

        regressors["white_matter_derivative1"] = pd.DataFrame(
            np.diff(regressors["white_matter"].tonumpy(), prepend=0)
        )
        regressors["white_matter_derivative1_power2"] = regressors["white_matter_derivative1"] ** 2

        regressors["csf_derivative1"] = pd.DataFrame(
            np.diff(regressors["csf"].tonumpy(), prepend=0)
        )
        regressors["csf_derivative1_power2"] = regressors["csf_derivative1"] ** 2

        # write out the json
        # jsonreg = pd.DataFrame({'LR': [1, 2, 3]})  # just a fake json
        regressors.to_csv(
            funcdir
            + "/sub-"
            + sub_id
            + "_task-"
            + idx[1]
            + "_acq-"
            + idx[2]
            + "_desc-confounds_timeseries.tsv",
            index=False,
            sep="\t",
        )
        regressors.to_json(
            funcdir
            + "/sub-"
            + sub_id
            + "_task-"
            + idx[1]
            + "_acq-"
            + idx[2]
            + "_desc-confounds_timeseries.json"
        )

        # functional files
        hcp_ref = k + "/" + filenamex + "_SBRef.nii.gz"
        prep_ref = (
            funcdir
            + "/sub-"
            + sub_id
            + "_task-"
            + idx[1]
            + "_acq-"
            + idx[2]
            + "_space-MNI152NLin6Asym_boldref.nii.gz"
        )

        # create/copy  cifti
        ciftip = k + "/" + filenamex + "_Atlas_MSMAll.dtseries.nii"
        ciftib = (
            funcdir
            + "/sub-"
            + sub_id
            + "_task-"
            + idx[1]
            + "_acq-"
            + idx[2]
            + "_space-fsLR_den-91k_bold.dtseries.nii"
        )

        niftip = k + "/" + filenamex + ".nii.gz"
        TR = nb.load(niftip).header.get_zooms()[-1]  # repetition time

        jsontis = {"RepetitionTime": np.float(TR), "TaskName": idx[1]}

        json2 = {
            "grayordinates": "91k",
            "space": "HCP grayordinates",
            "surface": "fsLR",
            "surface_density": "32k",
            "volume": "MNI152NLin6Asym",
        }

        boldjson = (
            funcdir
            + "/sub-"
            + sub_id
            + "_task-"
            + idx[1]
            + "_acq-"
            + idx[2]
            + "_space-MNI152NLin6Asym_desc-preproc_bold.json"
        )
        ddjsonfile = out_dir + "/dataset_description.json"
        ciftijson = (
            funcdir
            + "/sub-"
            + sub_id
            + "_task-"
            + idx[1]
            + "_acq-"
            + idx[2]
            + "_space-fsLR_den-91k_bold.dtseries.json"
        )

        ddjson = {"Name": "HCP", "DatasetType": "derivative", "GeneratedBy": [{"Name": "HCP"}]}

        writejson(jsontis, boldjson)
        writejson(json2, ciftijson)
        writejson(ddjson, ddjsonfile)

        # get the files
        rawfiles = [hcp_ref, ciftip]
        newfiles = [prep_ref, ciftib]

        for kk, jj in zip(rawfiles, newfiles):
            copyfileobj_example(kk, jj)
            print("finished converting" + kk)
