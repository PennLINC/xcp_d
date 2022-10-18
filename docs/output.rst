
.. include:: links.rst

----------------
Outputs of xcp_d
----------------

The  ``xcp_d`` outputs are written out in BIDS format and consist of three main parts.

.. admonition:: A note on BIDS compliance

     ``xcp_d`` attempts to follow the BIDS specification as best as possible.
     However, many ``xcp_d`` derivatives are not currently covered by the specification.
     In those instances, we attempt to follow recommendations from existing BIDS Extension Proposals (BEPs),
     which are in-progress proposals to add new features to BIDS.

     Three BEPs that are of particular use in ``xcp_d`` are
     `BEP012: Functional preprocessing derivatives <https://github.com/bids-standard/bids-specification/pull/519>`_,
     `BEP017: BIDS connectivity matrix data schema <https://docs.google.com/document/d/1ugBdUF6dhElXdj3u9vw0iWjE6f_Bibsro3ah7sRV0GA/edit?usp=sharing>`_,
     and
     `BEPXXX: Atlas Specification <https://docs.google.com/document/d/1RxW4cARr3-EiBEcXjLpSIVidvnUSHE7yJCUY91i5TfM/edit?usp=sharing>`_
     (currently unnumbered).

     In cases where a derivative type is not covered by an existing BEP,
     we have simply attempted to follow the general principles of BIDS.

     If you discover a problem with the BIDS compliance of ``xcp_d``'s derivatives, please open an issue in the ``xcp_d`` repository.

1. Summary reports: There are two summary reports - an executive summary per session and a participant summary::

       xcp_d/sub-<label>[_ses-<label>]_executive_summary.html
       xcp_d/sub-<label>.html

2. Anatomical outputs: Anatomical outputs consist of anatomical preprocessed T1w/T2w and segmentation images in MNI spaces::

        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz
        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-MNI152NLin6Asym_dseg.nii.gz

   If the ``--warp-surfaces-native2std`` option is selected, and reconstructed surfaces are available in the preprocessed dataset,
   then these surfaces will be warped to fsLR space at 32k density::

        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_midthickness.surf.gii
        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_inflated.surf.gii
        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_vinflated.surf.gii
        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-fsLR_den-32k_hemi-<L|R>_midthickness.surf.gii
        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-fsLR_den-32k_hemi-<L|R>_pial.surf.gii
        xcp_d/sub-<label>/[ses-<label>/]anat/<source_entities>_space-fsLR_den-32k_hemi-<L|R>_smoothwm.surf.gii

3. Functional outputs: Functional outputs consist of processed/denoised BOLD data, timeseries, functional connectivity matrices, and resting-state derivatives.

   a. Denoised or residual BOLD data::

       # Nifti
       xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_desc-denoised_bold.nii.gz
       xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_desc-denoised_bold.json

       # Cifti
       xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii
       xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_den-91k_desc-denoised_bold.json

      The json/sidecar contains parameters of the data and processing steps.

       .. code-block:: json-object

        {
              "Freq Band": [ 0.009, 0.08],
              "RepetitionTime": 2.0,
              "compression": true,
              "dummy vols": 0,
              "nuissance parameters": "27P",
              }

   b. Functional timeseries and connectivity matrices::

        #Nifti
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_atlas-<label>_timeseries.tsv
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_atlas-<label>_measure-pearsoncorrelation_conmat.tsv

        #Cifti
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_atlas-<label>_den-91k_timeseries.ptseries.nii
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_atlas-<label>_den-91k_measure-pearsoncorrelation_conmat.pconn.nii

   c. Resting-state derivatives (Regional Homogeneity and ALFF)::

        # Nifti
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_reho.nii.gz
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_alff.nii.gz

        # Cifti
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_den-32k_hemi-L_reho.func.gii
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_den-32k_hemi-R_reho.func.gii
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_den-91k_alff.dtseries.nii

   d. Other outputs include quality control and framewise  displacement::

        # Nifti
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_qc.csv
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_desc-framewisedisplacement_motion.tsv

        # Cifti
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_qc.csv
        xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_desc-framewisedisplacement_motion.tsv

      The ``desc-framewisedisplacement_motion.tsv`` is a tab-delimited file with one column: "framewise_displacement".

   e. DCAN style scrubbing file.
      This file is in hdf5 format (readable by h5py), and contains binary scrubbing masks from 0.0 to 1mm FD in 0.01 steps.
      At each step the following variables are present::

       # Nifti
       xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-<label>_desc-framewisedisplacement_bold-DCAN.hdf5

       # Cifti
       xcp_d/sub-<label>/[ses-<label>/]func/<source_entities>_space-fsLR_desc-framewisedisplacement-bold-DCAN.hdf5

       These files have the following keys:
       1. FD_threshold: a number >= 0 that represents the FD threshold used to calculate the metrics in this list
       2. frame_removal: a binary vector/array the same length as the number of frames in the concatenated time series, indicates whether a frame is removed (1) or not (0)
       3. format_string (legacy): a string that denotes how the frames were excluded -- uses a notation devised by Avi Snyder
       4. total_frame_count: a whole number that represents the total number of frames in the concatenated series
       5. remaining_frame_count: a whole number that represents the number of remaining frames in the concatenated series
       6. remaining_seconds: a whole number that represents the amount of time remaining after thresholding
       7. remaining_frame_mean_FD: a number >= 0 that represents the mean FD of the remaining frames
