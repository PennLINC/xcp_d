
.. include:: links.rst

----------------
General Workflow
----------------

Input data
-----------
The default inputs to `XCP-D` are the outputs of  `fMRIPrep` and `Nibabies`.  The `xcp_d` can aslo process `ABCD-DCAN` and minimal processing of `HCP` data which  require `input-type` flag ( 'dcan'  and 'hcp' for `ABCD-DCAN` and `HCP` respectively).


Processesing Steps
------------------

0. Data is read in. See :ref:`Running XCP-D:Inputs` for information on input dataset structures.

1. Skip volumes [Optional]: ``xcp_d`` allows the first N number of volumes to be skipped or deleted before processing. These volumes are usually refered to as dummy scans. It can be added to the command line with ``-d X`` where X is in seconds. For example, if your TR is .72, and you want to remove the first 2 volumes, you should enter: ``-d 1.44``. Most default scanning sequences include dummy volumes that are not reconstructed. However, some users still perfer to remove the first reconstructed few volumes.


2. Confound regressors selection: The confound regressors configurations in the table below are implemented in ``xcp_d`` with 36P as the default. In addition to the standard confound regressors selected from fMRIPrep outputs, a custom confound timeseries can be added as described below in:ref:`Running XCP-D:Custom Confounds`. If you are passing custom confound regressors, and you want none of the regressors here, the option will be ``custom``.

   .. list-table:: Confound
   
    * - Pipelines
      - Six Motion Estimates
      - White Matter
      - CSF
      - Global Signal
      - ACompCor
      - AROMA
    * - 24P 
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - 
      - 
      -
      -
      -
    * - 27P 
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X
      - X
      - X
      -
      -
    * - 36P 
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      -
      -
    * - acompcor_gsr 
      -  X, dX
      - 
      - 
      - X
      - 10 com, 5WM,5CSF
      -
    * - acompcor 
      - X, dX 
      - 
      - 
      - 
      - 10 com, 5WM,5CSF
      -
    * - aroma_gsr
      - X, dX 
      - X
      - X
      - X
      - 
      - X
    * - aroma 
      - X, dX 
      - X
      - X
      - 
      - 
      - X


   For more information about confound regresssors selection, please refer to `Ciric et. al. 2017`_ 

   After the selection of confound regressors, the respiratory effects can optionally be filtered out from the motion estimates with band-stop filter to improve fMRI data quality. Please refer to `Fair et. al. 2020`_ for more information. These band-stop parameters are age specific (see table below) and can  be added to the command line arguments (see :ref:`Running XCP-D:Command Structure`).

   .. list-table:: Respiratory Filter 


    * - Age Range
      - Cutoff Range 
        (Breaths per Minute)
    * - < 1 year 
      - 30 to  60
    * - 1 to 2 years 
      - 25 - 50
    * - 2 - 6 years
      - 20 - 35 
    * - 6-12 years  
      - 15 - 25 
    * - 12 - 18 years
      - 12 - 20 
    * - 19 - 65 years
      - 12 - 18 
    * - 65 - 80 years
      - 12 - 28 
    * - > 80 years
      - 10 - 30 

3. Despiking [Optional]: Despiking is a process in which large spikes in the BOLD times series are truncated. Despiking reduces/limits the amplitude or magnitude of the large spikes but preserves those data points with an imputed reduced amplitude. Despiking is  done before  regression and filtering to minimize the impact of spike. Despiking is applied to whole volumes  and data, and different from temporal censoring. It can be added to the command line arguments with ``--despike``.

4. Temporal Censoring: Temporal Censoring is a process in which data points with excessive motion outliers are identified/flagged. The censored data points are removed from the data before regression. This is effective for removing spurious sources of connectivity in fMRI data but must be applied very carefully. The Framewise displacement (FD) threshold  is used to identify the censored volumes or outliers which are obtained from the FMRIPREP nuissance matrix. The default FD threshold implemented in ``xcp_d`` is 0.3 mm.  This can be modifeid in the commmand line argument with ``--fd-threshold X`` where X is the FD threshold in mm. Any volume with FD above the threshold will be flagged as an outlier before the regession. Please refer to `Satterthwaite et al. 2013`_ and `Power et. al. 2012`_ for more information. There is also a process of  masking out the non-contiguous segments of data points between outliers. The number of contiguous volumes required to survive masking is set to 5 by default and and can be modifeid by the users in the command line arguments; here, if there are 1-4 volumes between volumes that are masked for censoring, those 1-4 volumes are also masked for censoring.

5. Confound Regression: At this stage, the BOLD data is denoised by regressing the confound regressors from the BOLD data. If there are any volumes or timepoints flagged, as outliers during censoring step, these volumes are excluded from the regression. In addition, if there are custom confound regressors, these are combined with the confound regressors selected, if any, in step 2.

6. Bandpass filtering: ``XCP-D`` implements a Butterworth bandpass filter to filter BOLD signal after regression. The bandpass filter parameters are set to 0.009 to 0.08 Hz with order of 2 by default and can be modified in the command line. If there are any flagged volumes or timepoints during Temporal Censoring, these volumes are interpolated before bandpass filtering. 

7. Functional timeseries and connectivity matrices: ``XCP-D`` implements  a module that extracts voxelwise timeseries with the brain atlases. The local mean timeseries within each brain atlas's region  of interest (ROI) is extracted. Currently, static connectivity is estimated using the Pearson correlation between all ROIs for a particular atlas. The following atlases are implemented in  ``XCP-D``:
     
  a. Schaefer 100,200,300,400,500,600,700,800,900,1000
  b. Glasser 360 
  c. Gordon 333 
  d. Tian Subcortical Atlas `Tian et al <https://www.nature.com/articles/s41593-020-00711-6.epdf?sharing_token=Fzk9fg_oTs49l2_4GcFHvtRgN0jAjWel9jnR3ZoTv0OcoEh_rWSSGTYcOuTVFJlvyoz7cKiJgYmHRlYIGzAnNt5tMyMZIXn3xdgdMC_wzDAONIDh5m0cUiLGzNChnEK_AHqVJl2Qrno8-hzk8CanTnXjGX3rRfZX3WXgTLew1oE%3D>`_

8.  Resting-state derivatives: For each BOLD data, the resting-state derivatives are computed. These includes regional homogeneity (ReHo) and amplitude of low-frequency fluctuation (ALFF).  

9. Residual BOLD and resting-state derivatives smoothing: A smoothing kernel of 6mm is implemented as default for smoothing residual BOLD, ReHo and ALFF. Kernel size can be modified in the command line arguments. 

10. Quality control. The quality control (QC) in ``XCP-D`` estimates the quality of BOLD data before and after regression and also estimates BOLD-T1w coregistration and BOLD-Template normalization qualites. The QC metrics include: 

      a. Motion parameters summary: mean FD, mean and maximum RMS 
      b. Mean DVARs before and after regression and its relationship to FD 
      c. BOLD-T1w coregistration quality - Dice, Jaccard, Coverage and Cross-correlation indices
      d. BOLD-Template normalization quality - Dice, Jaccard, Coverage and Cross-correlation indices



Outputs
-------

``XCP-D`` generates the following ouputs following the successful execution: 

1. Executive summary: The executive summary reveals some very important information of BOLD data before and after regression. 
It also shows some key files to allow a quick QC of the image processing results of a single session of a single subject.

2. Visual QA (quality assessment) reports: ``XCP-D`` generates one HTML per subject, per session (if applicable), that allows the user to conduct a thorough visual assessment 
of processed data. This also includes QC measures. 

3. Processed BOLD data: residual BOLD for each subject and session, functional timeseries and connectvity matrices, and  resting-state derivatives. 
   
4. Anatomical data. The anatomical data (processed T1w processed and segmentation files) are copied from fMRIPrep. If both images are not in MNI2006 space, they are resamspled to MNI space. The surfaces (Gifti files) in each subject are also remsapled to standardard space (fsLR-32K). 

See Outputs_ for details about xcp_d outputs. 
