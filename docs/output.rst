
.. include:: links.rst

----------------
Outputs of xcp_d
----------------

The  ``xcp_d`` outputs are written out in BIDS format and consist of three main parts.

#.   Summary reports:
     There are two summary reports - an executive summary per session and a participant summary::

          xcp_d/sub-${sub-id}_ses-${ses-id}_executive_summary.html
          xcp_d/sub-${sub-id}.html

#.   Anatomical outputs:
     Anatomical outputs conists of anatomical preprocessed T1w/T2w and segmentation images in MNI spaces::

          xcp_d/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz
          xcp_d/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_space-MNI152NLin6Asym_desc-preproc_dseg.nii.gz


     If there are gifti files in fMRIPrep output, the gifti files are resampled to standard space::

          xcp_d/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_den-32k_hemi-L_${surface}.surf.gii
          xcp_d/sub-${sub-id}/ses-${ses-id}/anat/sub-${sub-id}_ses-${ses-id}_den-32k_hemi-R_${surface}.surf.gii

#.   Functional outputs:
     Functional outputs consist of processed/denoised BOLD data, timeseries, functional connectivity matrices, and resting-state derivatives.

     #.   Denoised or residual BOLD data::

               # Nifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_desc-residual_bold.nii.gz
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_desc-residual_bold.json

               # Cifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_den-91k_desc-residual_bold.dtseries.nii
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_den-91k_desc-residual_bold.json

          The json/sidecar contains paramters of the data and processing steps.

          .. code-block:: json

               {
                    "Freq Band": [0.009, 0.08],
                    "RepetitionTime": 2.0,
                    "compression": true,
                    "dummy vols": 0,
                    "nuisance parameters": "27P",
               }

     #.   Functional timeseries and connectivity matrices::

               # Nifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_atlas-${atlasname}_desc-timeseries_bold.tsv
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_atlas-${atlasname}_desc-connectivity_bold.tsv

               # Cifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_atlas-${atlasname}_den-91k_timeseries.tsv
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_atlas-${atlasname}_den-91k_conmat.tsv

     #.   Resting-state derivatives (ReHo and ALFF)::

               # Nifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_desc-reho_bold.nii.gz
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_desc-alff_bold.nii.gz

               # Cifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_den-32k_hemi-L_desc-reho_bold.func.gii
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_den-32k_hemi-R_desc-reho_bold.func.gii
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_den-91k_desc-alff_bold.dtseries.nii

     #.   Other outputs include quality control and framewise displacement::

               # Nifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_desc-qc_bold.csv
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-${space}_desc-framewisedisplacement_bold.tsv

               # Cifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_desc-qc_bold.csv
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_run-${run}_space-fsLR_desc-framewisedisplacement_den-91k_bold.tsv

     #.   DCAN style scrubbing file.
          This file is in hdf5 format (readable by h5py), and contains binary scrubbing masks from 0.0 to 1mm FD in 0.01 steps.
          At each step the following variables are present::

               # Nifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-${space}_desc-framewisedisplacement_bold-DCAN.hdf5

               # Cifti
               xcp_d/sub-${sub-id}/ses-${ses-id}/func/sub-${sub-id}_ses-${ses-id}_task-${taskname}_space-fsLR_desc-framewisedisplacement-bold-DCAN.hdf5

          These files have the following keys:

          #.   FD_threshold: a number >= 0 that represents the FD threshold used to calculate the metrics in this list
          #.   frame_removal: a binary vector/array the same length as the number of frames in the concatenated time series, indicates whether a frame is removed (1) or not (0)
          #.   format_string (legacy): a string that denotes how the frames were excluded -- uses a notation devised by Avi Snyder
          #.   total_frame_count: a whole number that represents the total number of frames in the concatenated series
          #.   remaining_frame_count: a whole number that represents the number of remaining frames in the concatenated series
          #.   remaining_seconds: a whole number that represents the amount of time remaining after thresholding
          #.   remaining_frame_mean_FD: a number >= 0 that represents the mean FD of the remaining frames
