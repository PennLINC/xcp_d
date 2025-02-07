.. include:: links.rst

##########################
Frequently Asked Questions
##########################

.. contents:: Table of Contents
   :local:
   :depth: 1

*************************************
Should I use *XCP-D* for my analysis?
*************************************

This is a complicated question.
*XCP-D* is designed for resting-state or pseudo-resting-state functional connectivity analyses.
It also produces a number of resting-state-specific derivatives, includng ALFF and ReHo.

If your goal is just to denoise your resting-state data, *XCP-D* may be overkill.
You can perform simple denoising with tools like Nilearn or FSL.

*XCP-D* is not designed as a general-purpose postprocessing pipeline.
It is really only appropriate for certain analyses,
and other postprocessing/analysis tools are better suited for many types of data/analysis.

*XCP-D* derivatives are not particularly useful for task-dependent functional connectivity analyses,
such as psychophysiological interactions (PPIs) or beta series analyses.
It is also not suitable for general task-based analyses, such as standard task GLMs,
as we recommend including nuisance regressors in the GLM step,
rather than denoising data prior to the GLM.


***************************************************
How do you decide what steps to include in *XCP-D*?
***************************************************

We have tried to follow current best practices in the field for processing and denoising
resting-state data.
While there are always new methods being developed, we have limited *XCP-D* to methods that
have been validated and are commonly used in the field.
If there is a method you would like to see included in *XCP-D*, please open an issue on GitHub,
but please be aware that we may not be willing to include all methods.


*******************************************************
Can I use XCP-D on my data if it is not in BIDS format?
*******************************************************

No. *XCP-D* is designed to work with BIDS-formatted data.
Since *XCP-D* uses preprocessed data (i.e., data from a BIDS Derivatives dataset) and
the BIDS Derivatives dataset structure is not finalized, we largely follow the conventions
of NiPreps BIDS apps (especially fMRIPrep) for the input data.
