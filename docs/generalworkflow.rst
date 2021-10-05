
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

2. Confound regressors selection. Confound 
   

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

3. [Optional] Despiking is a process in which large spikes in the BOLD times series are truncated. Despiking reduces/limits the amplitude or 
   magnitude of the large spikes but preserves those data points with an imputed reduced amplitude. Despiking is encouraged to be done before 
   filtering and regression to minimize the impact of spike. Despiking is very effective because it is a voxelwise operation, and no volume is deleted/removed.

worklow 


outputs