# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import json
import glob
import shutil
import filecmp
import numpy as np
import pandas as pd
import nibabel as nb
from nilearn.input_data import NiftiMasker
from pkg_resources import resource_filename as pkgrf


def hcp2fmriprep(hcpdir, outdir, sub_id=None):
    hcpdir = os.path.abspath(hcpdir)
    outdir = os.path.abspath(outdir)
    if sub_id is None:
        sub_idir = glob.glob(hcpdir + '/*')
        sub_id = [os.path.basename(j) for j in sub_idir]
        if len(sub_id) == 0:
            raise ValueError('No subject found in %s' % hcpdir)
        elif len(sub_id) > 0:
            for j in sub_id:
                hcpfmriprepx(hcp_dir=hcpdir, out_dir=outdir, sub_id=j)
    else:
        hcpfmriprepx(hcp_dir=hcpdir, out_dir=outdir, sub_id=str(sub_id))

    return sub_id


def hcpfmriprepx(hcp_dir, out_dir, subid):
    sub_id = 'sub-' + subid
    anat_dirx = hcp_dir + '/' + subid + '/T1w/'

    # make new directory for anat and func
    anatdir = out_dir + '/sub-' + subid + '/anat/'
    funcdir = out_dir + '/sub-' + subid + '/func/'
    os.makedirs(anatdir, exist_ok=True)
    os.makedirs(funcdir, exist_ok=True)

    # get old files
    tw1 = anat_dirx + '/T1w_acpc_dc_restore.nii.gz'
    brainmask = anat_dirx + '/brainmask_fs.nii.gz'
    ribbon = anat_dirx + '/ribbon.nii.gz'
    segm = anat_dirx + '/aparc+aseg.nii.gz'

    midR = glob.glob(anat_dirx + '/fsaverage_LR32k/*R.midthickness.32k_fs_LR.surf.gii')[0]
    midL = glob.glob(anat_dirx + '/fsaverage_LR32k/*L.midthickness.32k_fs_LR.surf.gii')[0]
    infR = glob.glob(anat_dirx + '/fsaverage_LR32k/*R.inflated.32k_fs_LR.surf.gii')[0]
    infL = glob.glob(anat_dirx + '/fsaverage_LR32k/*L.inflated.32k_fs_LR.surf.gii')[0]

    pialR = glob.glob(anat_dirx + '/fsaverage_LR32k/*R.pial.32k_fs_LR.surf.gii')[0]
    pialL = glob.glob(anat_dirx + '/fsaverage_LR32k/*L.pial.32k_fs_LR.surf.gii')[0]

    whiteR = glob.glob(anat_dirx + '/fsaverage_LR32k/*R.white.32k_fs_LR.surf.gii')[0]
    whiteL = glob.glob(anat_dirx + '/fsaverage_LR32k/*L.white.32k_fs_LR.surf.gii')[0]

    hcpfiles = [tw1, segm, ribbon, brainmask, tw1, tw1,
                midL, midR, pialL, pialR, whiteL, whiteR, infL, infR]

    # to fmriprep directory
    t1wim = anatdir + sub_id + '_desc-preproc_T1w.nii.gz'
    t1seg = anatdir + sub_id + '_dseg.nii.gz'
    t1ribbon = anatdir + sub_id + '_desc-ribbon_T1w.nii.gz'
    t1brainm = anatdir + sub_id + '_desc-brain_mask.nii.gz'
    regfile1 = anatdir + sub_id + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
    regfile2 = anatdir + sub_id + '_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'

    lMid = anatdir + sub_id + '_hemi-L_midthickness.surf.gii'
    rMid = anatdir + sub_id + '_hemi-R_midthickness.surf.gii'

    lpial = anatdir + sub_id + '_hemi-L_pial.surf.gii'
    rpial = anatdir + sub_id + '_hemi-R_pial.surf.gii'

    lwhite = anatdir + sub_id + '_hemi-L_smoothwm.surf.gii'
    rwhite = anatdir + sub_id + '_hemi-R_smoothwm.surf.gii'

    linf = anatdir + sub_id + '_hemi-L_inflated.surf.gii'
    rinf = anatdir + sub_id + '_hemi-R_inflated.surf.gii'
    newanatfiles = [t1wim, t1seg, t1ribbon, t1brainm, regfile1, regfile2, lMid, rMid, lpial, rpial,
                    lwhite, rwhite, linf, rinf]

    for i, j in zip(hcpfiles, newanatfiles):
        copyfileobj_example(i, j)

    # get the task files
    taskx = glob.glob(hcp_dir + '/' + subid + '/MNINonLinear/Results/*')
    tasklistx = []
    for j in taskx:
        if j.endswith('RL') or j.endswith('LR'):
            tasklistx.append(j)

    csf_mask = pkgrf('xcp_d', 'masks/csf.nii.gz')
    wm_mask = pkgrf('xcp_d', 'masks/wm.nii.gz')

    for k in tasklistx:
        idx = os.path.basename(k).split('_')
        filenamex = os.path.basename(k)

        # create confound regressors

        mvreg = pd.read_csv(k + '/Movement_Regressors.txt', header=None, delimiter=r"\s+")
        mvreg = mvreg.iloc[:, 0:6]
        mvreg.columns = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        # convert rot to rad
        mvreg['rot_x'] = mvreg['rot_x']*np.pi/180
        mvreg['rot_y'] = mvreg['rot_y']*np.pi/180
        mvreg['rot_z'] = mvreg['rot_z']*np.pi/180

        # use masks: brain,csf and wm mask to extract timeseries
        bolddata = k + '/' + filenamex + '.nii.gz'
        brainmask = k + '/brainmask_fs.2.nii.gz'
        gsreg = extractreg(mask=brainmask, nifti=bolddata)
        csfreg = extractreg(mask=csf_mask, nifti=bolddata)
        wmreg = extractreg(mask=wm_mask, nifti=bolddata)

        rsmd = np.loadtxt(k + '/Movement_AbsoluteRMS.txt')
        brainreg = pd.DataFrame(
            {'global_signal': gsreg, 'white_matter': wmreg, 'csf': csfreg, 'rmsd': rsmd})
        regressors = pd.concat([mvreg, brainreg], axis=1)

        # write out the json
        # jsonreg = pd.DataFrame({'LR': [1, 2, 3]})  # just a fake json
        regressors.to_csv(funcdir+'/sub-'+subid+'_task-' +
                          idx[1]+'_acq-'+idx[2]+'_desc-confounds_timeseries.tsv',
                          index=False, sep='\t')
        regressors.to_json(funcdir+'/sub-' + subid +
                           '_task-' + idx[1]+'_acq-' +
                           idx[2]+'_desc-confounds_timeseries.json')

        # functional files
        hcp_ref = k + '/' + filenamex + '_SBRef.nii.gz'
        prep_ref = funcdir+'/sub-'+subid+'_task-' + \
            idx[1]+'_acq-'+idx[2]+'_space-MNI152NLin6Asym_boldref.nii.gz'

        # create/copy  cifti
        ciftip = k + '/' + filenamex + '_Atlas_MSMAll.dtseries.nii'
        ciftib = funcdir+'/sub-'+subid+'_task-' + \
            idx[1]+'_acq-'+idx[2]+'_space-fsLR_den-91k_bold.dtseries.nii'

        niftip = k + '/' + filenamex + '.nii.gz'
        tr = nb.load(niftip).header.get_zooms()[-1]   # repetition time

        jsontis = {"RepetitionTime": np.float(tr), "TaskName": idx[1]}

        json2 = {"grayordinates": "91k", "space": "HCP grayordinates",
                 "surface": "fsLR", "surface_density": "32k", "volume": "MNI152NLin6Asym"}

        boldjson = funcdir+'/sub-'+subid+'_task-' + \
            idx[1]+'_acq-'+idx[2]+'_space-MNI152NLin6Asym_desc-preproc_bold.json'
        ciftijson = funcdir+'/sub-'+subid+'_task-' + \
            idx[1]+'_acq-'+idx[2]+'_space-fsLR_den-91k_bold.dtseries.json'

        writejson(jsontis, boldjson)
        writejson(json2, ciftijson)

        # get the files
        rawfiles = [hcp_ref, ciftip]
        newfiles = [prep_ref, ciftib]

        for kk, jj in zip(rawfiles, newfiles):
            copyfileobj_example(kk, jj)


def copyfileobj_example(src, dst):
    if not os.path.exists(dst) or not filecmp.cmp(src, dst):
        shutil.copyfile(src, dst)


def symlinkfiles(source, dest):
    # Beware, this example does not handle any edge cases!
    with open(source, 'rb') as src, open(dest, 'wb') as dst:
        copyfileobj_example(src, dst)


def extractreg(mask, nifti):
    masker = NiftiMasker(mask_img=mask)
    signals = masker.fit_transform(nifti)
    return np.mean(signals, axis=1)


def writejson(data, outfile):
    with open(outfile, 'w') as f:
        json.dump(data, f)
    return outfile
