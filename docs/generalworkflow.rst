
.. include:: links.rst

----------------
General Worklow
----------------



Inputs data
-----------
The main input to `xpc_abcd` is the outputs of  `fMRIPrep`. The optional input is the custom task or physiological timeseries that users may want to
regress from the  BOLD data. This custom timeseries can be arranged as described in `Task Regression`_. 


Processesing Steps
------------------

1. [Optional] The ``xcp_abcd`` allows the first N number of volumes to be skipped or deleted before processing.
   These volumes are usually refer to as dummy scans. It can be added to command line with ``-d X`` where X in seconds


2. Confound regressors selection. The  confound regressors configrations in the table below are implemented in ``xcp_abcd`` with 27P as the default. 
   In addition to the standard confound regressors selected from fMRIPrep outputs, the custom timeseries can be added as described in `Task Regression`_.
   

   .. list-table:: Confound
   
    * - Pipelines
      - Six Motion Estimates
      - White Matter
      - CSF
      - Global Signal
      - tcompcor
      - acompcor
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
    * - tcompcor 
      -  
      - 
      - 
      -
      - 5 comps
      -
    * - acompcor 
      - X, dX 
      - 
      - 
      -
      -  
      - 5 comps

   For more  information about confound regresssors selection, please refer to `Ciric et. al. 2017`_ 

   [optional] After the selection of confound regressors, the respiratory effects can be filtered 
   out from the motion estimates with band-stop filter to improve fMRI data quality. Pls refer to `Fair et. al. 2020`_ 
   for more information. These band-stop parameters are age specific (see Table below) and can  be added to the command line arguments (see `Usage`_ )

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

3. [Optional] Despiking: Despiking is a process in which large spikes in the BOLD times series are truncated. 
   Despiking reduces/limits the amplitude or magnitude of the large spikes but preserves those data 
   points with an imputed reduced amplitude. Despiking is  done before  regression and filtering 
   to minimize the impact of spike. It  be added to command line arguments with ``--despike``.

4. Temporal Censoring: Temporal Censoring is a process in which data points with excessive motion outliers are identified/flagged. 
   The censored data points are removed from the data before regression. This is effective for removing spurious sources of connectivity in fMRI data
   but must be applied very carefully. However, the  Framewise displacement (FD) threshold  is used to identify the censored volumes or outliers 
   which is obtained from FMRIPREP nuissance matrix. 
   The default FD threshold implemented in ``xcp_abcd`` is 0.3 mm.  This can be modifeid in the commmand line argument with ``--fd-threshold X`` where X is the FD threshold in mm.
   Any volume with FD above the threshold will be flagged as outliers before the regession.
   Please refer to `Satterthwaite et al. 2013`_ and `Power et. al. 2012`_ for more information. 

   There is also a process of  masking out the non-contiguous segments of data points between outliers. The number of contiguous volumes required to survive masking is set to 5 by default and 
   and can be modifeid by the users in the commnad line argument.

5. Confound Regression: At this stage, the BOLD data is denoised by regressing the confound regressors from the BOLD data. If there is any volumes or timepoints flagged 
   as outliers during censoring step, these volumes are excluded from the regression. In addtion, if there is custom confound regressors,
   this is combined with the confound regressors selected in step 2. 

6. Bandpass fitering: The ``xcp_abcd`` implemented butterworth bandpass filter to filter BOLD signal after regression. 
   The bandpass filter parameters are set to  0.009 to 0.08 Hz with order of 2  by default and cam be modified in the command line. 
   if there is any flagged volumes or timepoints during censoring step, these volumes are interpolated  before bandpass filtering. 

7. Functional timeseries and connectivity matrixces.  The ``xcp_abcd`` implemented  a module that extract voxelwise timeseries with
   thebrain atlases. The local mean timeseries within each brain atlas's region  of interest(ROI) is extracted. Currently, static connectivity is estimated using 
   the Pearson correlation between all ROIs for a particular atlas. The following atlases are implemented in  ``xcp_abcd``:
     
  a. Schaefer 200 and 400
  b. Glasser 360 
  c. Gordon 360 
  d. subcortital atlas  


8.  Resting-state derivatives: For each BOLD data, the resting-state derivates are computed. These includes regional homogeneity (ReHo) 
    and amplitude of low-frequency fluctuation (ALFF).  

9. Residual BOLD and resting-state derivatives smoothing. A smoothing kernel of 5mm is implemented as default for smoothing residual BOLD, ReHo 
   and ALFF. Kernel size can be modified in the command line argument. 

10. Quality control. The quality control(QC) in ``xcp_abcd`` estimates the quality of BOLD data before and after regression and also estimate BOLD-T1w coregistration  and
     BOLD-Template normalization qualites. The QC metrics include: 

      a. motion parameters summary: mean FD, mean and maximum RMS 
      b. Mean DVARs before and after regression and its relationship to FD 
      c. BOLD-T1w coregistration quality - Dice, Jaccard, Coverage and Cross-correlation indices
      d. BOLD-Template normalization quality - Dice, Jaccard, Coverage and Cross-correlation indices



Outputs
-------

``xcp_abcd`` generatex the following ouputs following the successful execution: 

1. Excutive summary:  Execuive summary revealed some very importnat information of BOLD data before and after regression. 
It also show some key files to allow a quick QC of the image processing results of a single session of a single subject.

2. Visual QA (quality assessment) reports: one HTML per subject, per session (if applicable), that allows the user to conduct a thorough visual assessment 
of processed data. This also includes QC measures. 

3. Processed BOLD data: the residual BOLD for each subject and session, functional timeseries and connectvity matrices, and  resting-state derivatives. 
   
4. Anatomical data. The anatomical data( processed T1w processed and segmentation files ) are copied from fMRIPrep. If both images are not in MNI2006 space, they are resamspled to MNI space
The surfaces (Gifti files) in subjects are also remsapled to standardard space (fsLR-32K). 

See `Outputs`_ for details about xcp_abcd  outputs. 