# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os,json,glob,re
import numpy as np 
import pandas as pd
import nibabel as nb 
from nilearn.input_data import NiftiMasker


def dcan2fmriprep(dcan_dir,out_dir,sub_id):
    """
    dcan2fmriprep(dcan_dir,out_dir)
    """
    # get session id if available 

    sess =glob.glob(dcan_dir+'/'+sub_id+'/s*')
    ses_id = []
    ses_id = [ j.split('ses-')[1] for j in sess]
    # anat dirx 
    
    
    for ses in ses_id:
        anat_dirx = dcan_dir+'/' + sub_id + '/ses-' +sess + '/files/MNINonLinear/'
        anatdir = out_dir +'/' + sub_id + '/ses-'+ses+ '/anat/'
        os.makedirs(anatdir,exist_ok=True)

        tw1 = anat_dirx +'/T1w.nii.gz'
        brainmask  = anat_dirx + '/brainmask_fs.nii.gz'
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
         
        dcanimages = [tw1,segm,ribbon, brainmask,tw1,tw1,midL,midR,pialL,pialR,whiteL,whiteR,infL,infR]
        
        t1wim = anatdir + sub_id + '_' + sess + '_desc-preproc_T1w.nii.gz'
        t1seg = anatdir + sub_id + '_' + sess + '_dseg.nii.gz'
        t1ribbon = anatdir + sub_id + '_' + sess + '_desc-ribbon_T1w.nii.gz'
        t1brainm =  anatdir + sub_id + '_' + sess + '_desc-brain_mask.nii.gz'
        regfile1  =  anatdir + sub_id + '_' + sess + '_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5'
        regfile2  =  anatdir + sub_id + '_' + sess + '_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'

        lMid = anatdir + sub_id + '_' + sess + '_hemi-L_midthickness.surf.gii'
        rMid = anatdir + sub_id + '_' + sess + '_hemi-R_midthickness.surf.gii'

        lpial = anatdir + sub_id + '_' + sess + '_hemi-L_pial.surf.gii'
        rpial = anatdir + sub_id + '_' + sess + '_hemi-R_pial.surf.gii'

        lwhite = anatdir + sub_id + '_' + sess + '_hemi-L_smoothwm.surf.gii'
        rwhite = anatdir + sub_id + '_' + sess + '_hemi-R_smoothwm.surf.gii'

        linf = anatdir + sub_id + '_' + sess + '_hemi-L_inflated.surf.gii'
        rinf = anatdir + sub_id + '_' + sess + '_hemi-R_inflated.surf.gii'

        newanatfiles =[t1wim,t1seg,t1ribbon,t1brainm,regfile1,regfile2,lMid,rMid,lpial,rpial,
                     lwhite,rwhite,linf,rinf]
        
        for i,j in zip(dcanimages,newanatfiles):
            symlinkfiles(i,j)
        

        # get masks and transforms 

        wmmask =glob.glob(anat_dirx + '/wm_2mm_*_mask_eroded.nii.gz')[0]
        csfmask =glob.glob(anat_dirx + '/vent_2mm_*_mask_eroded.nii.gz')[0]
        tw1tonative = anat_dirx +'xfms/T1w_to_MNI_0GenericAffine.mat'

        # get task and idx  run 01 
        func_dirx  = dcan_dir +'/' + sub_id + '/ses-' +ses_id[0] + '/files/MNINonLinear/Results/'
        taskd = glob.glob(func_dirx + 'task-*')
        taskid=[]
        for k in taskd:
            if not os.path.isfile(k):
                taskid.append(os.path.basename(k).split('-')[1])


        

        func_dir = out_dir +'/' + sub_id + '/ses-'+ses+ '/func/' 
        os.makedirs(func_dir,exist_ok=True)
        for ttt in taskd:
            taskname = re.split(r'(\d+)', ttt)[0]
            run_id = re.split(r'(\d+)', ttt)[1] 


   
    

    



def symlinkfiles( src, dest):
    if os.path.exists(dest): 
        os.remove(dest)
        os.symlink(src,dest)
    else:
        os.symlink(src,dest)
    return dest 

def extractreg(mask,nifti):
    masker=NiftiMasker(mask_img=mask)
    signals = masker.fit_transform(nifti)
    return np.mean(signals,axis=1)

def writejson(data,outfile):
    with open(outfile,'w') as f:
        json.dump(data,f)
    return outfile 
