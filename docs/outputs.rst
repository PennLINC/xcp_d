
.. include:: links.rst

##################
Outputs of *XCP-D*
##################

The  ``xcp_d`` outputs are written out in BIDS format and consist of three main parts.

.. admonition:: A note on BIDS compliance

   ``xcp_d`` attempts to follow the BIDS specification as best as possible.
   However, many ``xcp_d`` derivatives are not currently covered by the specification.
   In those instances, we attempt to follow recommendations from existing BIDS Extension Proposals
   (BEPs), which are in-progress proposals to add new features to BIDS.

   Three BEPs that are of particular use in ``xcp_d`` are
   `BEP012: Functional preprocessing derivatives <https://github.com/bids-standard/bids-specification/pull/519>`_,
   `BEP017: BIDS connectivity matrix data schema <https://docs.google.com/document/d/1ugBdUF6dhElXdj3u9vw0iWjE6f_Bibsro3ah7sRV0GA/edit?usp=sharing>`_,
   and
   `BEPXXX: Atlas Specification <https://docs.google.com/document/d/1RxW4cARr3-EiBEcXjLpSIVidvnUSHE7yJCUY91i5TfM/edit?usp=sharing>`_
   (currently unnumbered).

   In cases where a derivative type is not covered by an existing BEP,
   we have simply attempted to follow the general principles of BIDS.

   If you discover a problem with the BIDS compliance of ``xcp_d``'s derivatives, please open an
   issue in the ``xcp_d`` repository.


***************
Summary Reports
***************

There are two summary reports - a Nipreps-style participant summary and an executive summary per
session (if ``--dcan-qc`` is used).
The executive summary is based on the DCAN lab's
`ExecutiveSummary tool <https://github.com/DCAN-Labs/ExecutiveSummary>`_.

.. code-block::

   xcp_d/
      sub-<label>.html
      sub-<label>[_ses-<label>]_executive_summary.html


******************
Anatomical Outputs
******************

Anatomical outputs consist of anatomical preprocessed T1w/T2w and segmentation images in MNI space.

.. code-block::

   xcp_d/
      sub-<label>/[ses-<label>/]
         anat/
            <source_entities>_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz
            <source_entities>_space-MNI152NLin6Asym_desc-preproc_T2w.nii.gz
            <source_entities>_space-MNI152NLin6Asym_dseg.nii.gz

If the ``--warp-surfaces-native2std`` option is selected, and reconstructed surfaces are available
in the preprocessed dataset, then these surfaces will be warped to fsLR space at 32k density.

.. code-block::

   xcp_d/
      sub-<label>/[ses-<label>/]
         anat/
            <source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_midthickness.surf.gii
            <source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_inflated.surf.gii
            <source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_vinflated.surf.gii
            <source_entities>_space-fsLR_den-32k_hemi-<L|R>_midthickness.surf.gii
            <source_entities>_space-fsLR_den-32k_hemi-<L|R>_pial.surf.gii
            <source_entities>_space-fsLR_den-32k_hemi-<L|R>_smoothwm.surf.gii


******************
Functional Outputs
******************

Functional outputs consist of processed/denoised BOLD data, timeseries,
functional connectivity matrices, and resting-state derivatives.

.. important::

   Prior to version 0.4.0, the denoised data outputted by ``xcp_d`` was interpolated,
   meaning that high-motion volumes were replaced with interpolated data prior to temporal
   filtering.
   **This was a bug.**
   From 0.4.0 on, we have started to only write out the censored version of the denoised data,
   with high-motion volumes completely removed.
   This extends to the parcellated time series and correlation matrices as well.


Denoised or residual BOLD data
==============================

.. code-block::

   xcp_d/
      sub-<label>/[ses-<label>/]
         func/
            # Nifti
            <source_entities>_space-<label>_desc-denoised_bold.nii.gz
            <source_entities>_space-<label>_desc-denoised_bold.json
            <source_entities>_space-<label>_desc-denoisedSmoothed_bold.nii.gz
            <source_entities>_space-<label>_desc-denoisedSmoothed_bold.json
            <source_entities>_space-<label>_desc-interpolated_bold.nii.gz
            <source_entities>_space-<label>_desc-interpolated_bold.json

            # Cifti
            <source_entities>_space-fsLR_den-91k_desc-denoised_bold.dtseries.nii
            <source_entities>_space-fsLR_den-91k_desc-denoised_bold.json
            <source_entities>_space-fsLR_den-91k_desc-denoisedSmoothed_bold.dtseries.nii
            <source_entities>_space-fsLR_den-91k_desc-denoisedSmoothed_bold.json
            <source_entities>_space-fsLR_den-91k_desc-interpolated_bold.dtseries.nii
            <source_entities>_space-fsLR_den-91k_desc-interpolated_bold.json

.. important::

   The smoothed denoised BOLD files will only be generated if smoothing is enabled with the
   ``--smoothing parameter``.

.. important::

   The interpolated denoised BOLD files (``desc-interpolated``) should NOT be used for analyses.
   These files are only generated if ``--dcan-qc`` is used, and primarily exist for
   compatibility with DCAN-specific analysis tools.

The json/sidecar contains parameters of the data and processing steps.

   .. code-block:: json-object

      {
         "Freq Band": [0.01, 0.08],
         "RepetitionTime": 2.0,
         "compression": true,
         "dummy vols": 0,
         "nuisance parameters": "27P",
      }


Functional timeseries and connectivity matrices
===============================================

This includes the atlases used to extract the timeseries.

.. code-block::

   xcp_d/
      # Nifti
      space-<label>_atlas-<label>_dseg.nii.gz

      # Cifti
      space-<label>_atlas-<label>_dseg.dlabel.nii

      sub-<label>/[ses-<label>/]
         func/
            # Nifti
            <source_entities>_space-<label>_atlas-<label>_coverage.tsv
            <source_entities>_space-<label>_atlas-<label>_timeseries.tsv
            <source_entities>_space-<label>_atlas-<label>_measure-pearsoncorrelation_conmat.tsv

            # Cifti
            <source_entities>_space-fsLR_atlas-<label>_den-91k_coverage.tsv
            <source_entities>_space-fsLR_atlas-<label>_den-91k_coverage.pscalar.nii
            <source_entities>_space-fsLR_atlas-<label>_den-91k_timeseries.tsv
            <source_entities>_space-fsLR_atlas-<label>_den-91k_timeseries.ptseries.nii
            <source_entities>_space-fsLR_atlas-<label>_den-91k_measure-pearsoncorrelation_conmat.tsv
            <source_entities>_space-fsLR_atlas-<label>_den-91k_measure-pearsoncorrelation_conmat.pconn.nii


Resting-state metric derivatives (Regional Homogeneity and ALFF)
================================================================

.. code-block::

   xcp_d/
      sub-<label>/[ses-<label>/]
         func/
            # Nifti
            <source_entities>_space-<label>_reho.nii.gz
            <source_entities>_space-<label>_alff.nii.gz
            <source_entities>_space-<label>_desc-smooth_alff.nii.gz

            # Cifti
            <source_entities>_space-fsLR_den-91k_reho.dscalar.nii
            <source_entities>_space-fsLR_den-91k_alff.dscalar.nii
            <source_entities>_space-fsLR_den-91k_desc-smooth_alff.dscalar.nii

.. important::
      The smoothed ALFF image will only be generated if smoothing is enabled
      (i.e., with the ``--smoothing parameter``).

.. important::
      ALFF images will not be generated if bandpass filtering is disabled
      (i.e., with the ``--disable-bandpass-filtering`` parameter),
      or if high-motion outlier censoring is enabled
      (i.e., ``--fd-thresh`` is greater than zero).


Other outputs include quality control, framewise displacement, and confounds files
==================================================================================

.. code-block::

   xcp_d/
      sub-<label>/[ses-<label>/]
         func/
            # Nifti
            <source_entities>_space-<label>_desc-linc_qc.csv
            <source_entities>[_desc-filtered]_motion.tsv
            <source_entities>[_desc-filtered]_motion.json
            <source_entities>_outliers.tsv
            <source_entities>_outliers.json
            <source_entities>_design.tsv

            # Cifti
            <source_entities>_space-fsLR_desc-linc_qc.csv
            <source_entities>[_desc-filtered]_motion.tsv
            <source_entities>[_desc-filtered]_motion.json
            <source_entities>_outliers.tsv
            <source_entities>_outliers.json
            <source_entities>_design.tsv

``[desc-filtered]_motion.tsv`` is a tab-delimited file with seven columns:
one for each of the six filtered motion parameters, as well as "framewise_displacement".
If no motion filtering was applied, this file will not have the ``desc-filtered`` entity.
This file includes the high-motion volumes that are removed in most other derivatives.

``outliers.tsv`` is a tab-delimited file with one column: "framewise_displacement".
The "framewise_displacement" column contains zeros for low-motion volumes, and ones for
high-motion outliers.
This file includes the high-motion volumes that are removed in most other derivatives.

``design.tsv`` is a tab-delimited file with one column for each nuisance regressor,
including an intercept column, a linear trend column, and one-hot encoded regressors indicating
each of the high-motion outlier volumes.
This file includes the high-motion volumes that are removed in most other derivatives.


DCAN style scrubbing file (if ``--dcan-qc`` is used)
====================================================

This file is in hdf5 format (readable by h5py), and contains binary scrubbing masks from 0.0
to 1mm FD in 0.01 steps.

.. code-block::

   xcp_d/
      sub-<label>/[ses-<label>/]
         func/
            # Nifti
            <source_entities>_desc-dcan_qc.hdf5

            # Cifti
            <source_entities>_desc-dcan_qc.hdf5

These files have the following keys:

1. ``FD_threshold``: a number >= 0 that represents the FD threshold used to calculate the
   metrics in this list
2. ``frame_removal``: a binary vector/array the same length as the number of frames in the
   concatenated time series, indicates whether a frame is removed (1) or not (0)
3. ``format_string`` (legacy): a string that denotes how the frames were excluded -- uses a
   notation devised by Avi Snyder
4. ``total_frame_count``: a whole number that represents the total number of frames in the
   concatenated series
5. ``remaining_frame_count``: a whole number that represents the number of remaining frames in
   the concatenated series
6. ``remaining_seconds``: a whole number that represents the amount of time remaining after
   thresholding
7. ``remaining_frame_mean_FD``: a number >= 0 that represents the mean FD of the remaining frames
