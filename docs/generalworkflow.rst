.. include:: links.rst

---------------
General Worklow
---------------



Inputs data
-----------
The main input to `xpc_abcd` is the outputs of  `fMRIPrep`. The optional input is custom task or physiological timeseries that users may want to
regress from the  bold data. This custom timeseries can be arrange as described in `Task Regression`_. 

 
Processesing Steps
------------------

1. [Optional] The pipeline allows to implement the first N number of volumes to skip before processing.
   These volumes are usaully refer to as dummy scans. It can be added to command line wiht 
   ``-d X`` where X in seconds


2. Confound regressors selection. The  confound regressors in the table below are implemented in ``xcp_abcd`` with 27P ad default. 
   In addition to the confound regressors, the custom timeseries can be added as described in `Task Regression`_.

   .. list-table:: Confound table
       :widths: 10 10 10 10 10 10 10 
       :header-rows: 1
       * - Pipelines
         - Six Motion
         - White matter
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
  
  
    For more information see `Ciric et. al. 2017`_

    [Optional] Before regression, the respiratory effects can be filtered out from  the motion estimates as described in `Fair et. al. 2020`_.
    The respiratory effects are age specific  and parmaters are selected based on the age: 

    
    .. list-table:: Respiratory artifatcs
       :widths: 10 10 
       :header-rows: 1
       * - Age Range
         - Cutoff Range (Breaths per Minute)
       * - < 1 year 
         - 30 - 60
       * - 1-2 years 
         - 25 - 50
       * -  2 - 6 years
         -  20 - 35 
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


3. [Optional] Despiking is a process in which large spikes in the BOLD times series are truncated. Despiking reduces/limits the amplitude or 
   magnitude of the large spikes but preserves those data points with an imputed reduced amplitude. Despiking is encouraged to be done before 
   filtering and regression to minimize the impact of spike. Despiking is very effective because it is a voxelwise operation, and no volume is deleted/removed.


outputs