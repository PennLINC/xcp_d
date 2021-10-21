
.. include:: links.rst

-------------------
Outputs of XCP_ABCD
-------------------

The  ``xcp_abcd`` outputs are written out in BIDS format and consist of three main parts. 

1. Summary reports: There are two summary reports - an executive summary per session and a participant summary::
       
       xcp_abcd/sub-${sub-id}_ses-${ses-id}_executive_summary.html
       xcp_abcd/sub-${sub-id}.html

2. Anatomical outputs: Anatomical outputs conists of anatomical preprocessed T1w/T2w  and segmentation images in MNI spaces::
       
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_space-MNI152NLin6Asym_desc-preproc_dseg.nii.gz


   If there are gifti files in fMRIPrep output, the gifti files are resmapled to standard sapce::
        
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_den-32k_hemi-L_${surface}.surf.gii
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_den-32k_hemi-R_${surface}.surf.gii
    
3. Functional outputs: Functional outputs consist of processed/denoised BOLD data, timeseries, functional connectivity matrices, and resting-state derivatives.
   
   a. Denoised or residual BOLD data::

       # nifti 
       xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_desc-residual_bold.nii.gz
       xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_desc-residual_bold.json

       # cifti 
       xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_den-91k_desc-residual_bold.dtseries.nii
       xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_den-91k_desc-residual_bold.json
    
      The json/sidecar contains paramters of the data and processing steps. 

       .. code-block:: json-object
     
        {
              "Freq Band": [ 0.009, 0.08],
              "RepetitionTime": 2.0,
              "compression": true,
              "dummy vols": 0,
              "nuissance parameters": "27P",
              }

   b. Functional timeseries and connectivity matrices::

        #nifti
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_atlas-${atlasname}_desc-timeseries_bold.tsv
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_atlas-${atlasname}_desc-connectivity_bold.tsv
        
        #cifti
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_atlas-${atlasname}_den-91k_bold.pconn.nii
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_atlas-${atlasname}_den-91k_bold.ptseries.nii
       
   c. Resting-state derivatives (ReHo and ALFF)::
        
        # nifti
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_desc-reho_bold.nii.gz
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_desc-alff_bold.nii.gz

        # cifti
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_den-32k_hemi-L_desc-reho_bold.func.gii
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_den-32k_hemi-R_desc-reho_bold.func.gii
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_den-91k_desc-alff_bold.dtseries.nii
    
   d. Other outputs inlcude quality control and framewise  displacement::

        # nifti
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_desc-qc_bold.csv
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_desc-framewisedisplacement_bold.tsv
        
        # cifti
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_desc-qc_bold.csv
        xcp_abcd/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_desc-framewisedisplacement_den-91k_bold.tsv
        
        
       


