# XCP-D

*XCP-D*: A Robust Postprocessing Pipeline of  fMRI data
===========================================================

.. image:: https://zenodo.org/badge/309485627.svg
   :target: https://zenodo.org/badge/latestdoi/309485627

This pipeline is developed by the `Satterthwaite lab at the University of Pennslyvania
<https://www.satterthwaitelab.com/>`_ (**XCP**\; eXtensible Connectivity Pipeline)  and `Developmental Cognition and Neuroimaging lab at the University of Minnesota 
<https://innovation.umn.edu/developmental-cognition-and-neuroimaging-lab/>`_ (**-D**\CAN)for 
open-source software distribution.

About
------
XCP-D paves the final section of the reproducible and scalable route from the MRI scanner to functional connectivity data in the hands of neuroscientists. We developed XCP-D to extend the BIDS and NiPrep apparatus to the point where data is most commonly consumed and analyzed by neuroscientists studying functional connectivity. Thus, with the development of XCP-D, data can be automatically preprocessed and analyzed in BIDS format, using NiPrep-style containerized code, all the way from the from the scanner to functional connectivity matrices.

XCP-D picks up right where `fMRIprep <https://fmriprep.org>`_ ends, directly consuming the outputs of fMRIPrep. XCP-D leverages the BIDS and NiPrep frameworks to automatically generate denoised BOLD images, parcellated time series, functional connectivity matrices, and quality assessment reports. XCP-D can also process outputs from: `NiBabies <https://nibabies.readthedocs.io>`_, `DCAN <https://github.com/DCAN-Labs/abcd-hcp-pipeline>`_ and `Minimal preprocessed HCP data <https://www.humanconnectome.org/study/hcp-lifespan-development/data-releases>`_

.. image:: https://raw.githubusercontent.com/pennlinc/xcp_d/main/docs/_static/schematic_land-01.png

See the `documentation <https://xcp-abcd.readthedocs.io/en>`_ for more details.
