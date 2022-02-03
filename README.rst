# XCP-D

*XCP-D*: A Robust Postprocessing Pipeline of  fMRI data
===========================================================

.. image:: https://zenodo.org/badge/309485627.svg
   :target: https://zenodo.org/badge/latestdoi/309485627

This pipeline is developed by the `Satterthwaite lab at the University of Pennysilvania
<https://www.satterthwaitelab.com/>`_  and `Developmental Cognition and Neuroimaaging lab  at the University of Minnesota 
<https://innovation.umn.edu/developmental-cognition-and-neuroimaging-lab/>`_ for 
open-source software distribution.

About
------
XCP-D paves the final section of the reproducible and scalable route from the MRI scanner to functional connectivity data in the hands of neuroscientists. We developed XCP-D to extend the BIDS and NiPrep apparatus to the point where data is most commonly consumed and analyzed by neuroscientists studying functional connectivity. Thus, with the development of XCP-D, data can be automatically preprocessed and analyzed in BIDS format, using NiPrep-style containerized code, all the way from the from the scanner to functional connectivity matrices.

XCP-D picks up right where _fMRIPrep ends, directly consuming the outputs of fMRIPrep. XCP-D leverages the BIDS and NiPrep frameworks to automatically generate denoised BOLD images, parcellated time series, functional connectivity matrices, and quality assessment reports. 

1. `fMRIprep <https://fmriprep.org>`_
2. `NiBabies <https://nibabies.readthedocs.io>`_
3. `DCAN <https://github.com/DCAN-Labs/abcd-hcp-pipeline>`_
4. `Minimal preprocessed HCP data <https://www.humanconnectome.org/study/hcp-lifespan-development/data-releases>`_


See the `documentation <https://xcp-abcd.readthedocs.io/en>`_ for more details.


.. image:: https://raw.githubusercontent.com/pennlinc/xcp_d/main/docs/_static/schematic_land-01.png

