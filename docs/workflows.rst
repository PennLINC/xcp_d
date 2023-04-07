
.. include:: links.rst

###########################
Processing Pipeline Details
###########################


**********
Input data
**********

The default inputs to XCP-D are the outputs of ``fMRIPrep`` (``--input-type fmriprep``) and
``Nibabies`` (``--input-type nibabies``).
XCP-D can also postprocess ``HCP`` data (``--input-type hcp``).


****************
Processing Steps
****************

See :ref:`usage_inputs` for information on input dataset structures.


Anatomical processing
=====================
:func:`~xcp_d.workflows.anatomical.init_postprocess_anat_wf`

XCP-D performs minimal postprocessing on anatomical derivatives from the preprocessing pipeline.
This includes applying existing transforms to preprocessed T1w and T2w volumes,
in order to warp them from native T1w space to the target standard space,
while retaining the original resolution.


Surface normalization
---------------------
:func:`~xcp_d.workflows.anatomical.init_warp_surfaces_to_template_wf`

If the ``--warp-surfaces-native2std`` is used, then fsnative surface files from the preprocessing
derivatives will be warped to fsLR-32k space.

.. important::

   This step will only succeed if FreeSurfer derivatives are also available.


Confound regressor selection
============================
:func:`~xcp_d.workflows.postprocessing.init_prepare_confounds_wf`,
:func:`~xcp_d.utils.confounds.consolidate_confounds`

The confound regressor configurations in the table below are implemented in XCP-D,
with ``36P`` as the default.
In addition to the standard confound regressors selected from fMRIPrep outputs,
custom confounds can be added as described in :ref:`usage_custom_confounds`.
If you want to use custom confounds, without any of the nuisance regressors described here,
use ``--nuisance-regressors custom``.

.. list-table:: Confound

   *  - Pipelines
      - Six Motion Estimates
      - White Matter
      - CSF
      - Global Signal
      - ACompCor
      - AROMA
      - Linear Trend
      - Intercept
   *  - 24P
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      -
      -
      -
      -
      -
      - X
      - X
   *  - 27P
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X
      - X
      - X
      -
      -
      - X
      - X
   *  - 36P
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      - X, X\ :sup:`2`, dX, dX\ :sup:`2`
      -
      -
      - X
      - X
   *  - acompcor_gsr
      -  X, dX
      -
      -
      - X
      - 10 com, 5WM, 5CSF
      -
      - X
      - X
   *  - acompcor
      - X, dX
      -
      -
      -
      - 10 com, 5WM, 5CSF
      -
      - X
      - X
   *  - aroma_gsr
      - X, dX
      - X
      - X
      - X
      -
      - X
      - X
      - X
   *  - aroma
      - X, dX
      - X
      - X
      -
      -
      - X
      - X
      - X

For more information about confound regressor selection, please refer to :footcite:t:`benchmarkp`.

.. warning::

   In XCP-D versions prior to 0.3.1, the selected AROMA confounds were incorrect.
   We strongly advise users of these versions not to use the ``aroma`` or ``aroma_gsr``
   options.


Dummy scan removal [OPTIONAL]
=============================
:func:`~xcp_d.workflows.postprocessing.init_prepare_confounds_wf`,
:class:`~xcp_d.interfaces.censoring.RemoveDummyVolumes`

XCP-D allows the first *N* volumes to be removed before processing.
These volumes are usually refered to as dummy volumes.
Most default scanning sequences include dummy volumes that are not reconstructed.
However, some users still prefer to remove the first few reconstructed volumes.

Users may provide the number of volumes directly with the ``--dummy-scans <INT>`` parameter,
or they may rely on the preprocessing pipeline's estimated non-steady-state volume indices with
``--dummy-scans auto``.


Identification of high-motion outlier volumes
=============================================
:func:`~xcp_d.workflows.postprocessing.init_prepare_confounds_wf`,
:class:`~xcp_d.interfaces.censoring.FlagMotionOutliers`

XCP-D uses framewise displacement to identify high-motion outlier volumes.
These outlier volumes are removed from the BOLD data prior to denoising.

The threshold used to identify outlier volumes can be set with the ``--fd-thresh`` parameter.

.. important::
   If a BOLD run does not have enough low-motion data, then the post-processing workflow
   will automatically stop early, and no derivatives for that run will be written out.


Despiking [OPTIONAL]
====================
:func:`~xcp_d.workflows.postprocessing.init_despike_wf`

Despiking is a process in which large spikes in the BOLD times series are truncated.
Despiking reduces/limits the amplitude or magnitude of the large spikes but preserves those
data points with an imputed reduced amplitude.
This is done before regression and filtering, in order to minimize the impact of large amplitude
changes in the data.
It can be added to the command line arguments with ``--despike``.


Motion parameter filtering [OPTIONAL]
-------------------------------------
:func:`~xcp_d.workflows.postprocessing.init_prepare_confounds_wf`,
:class:`~xcp_d.interfaces.censoring.FlagMotionOutliers`,
:func:`~xcp_d.utils.confounds.load_motion`

Motion parameters may be contaminated with respiratory effects :footcite:p:`power2019distinctions`.
In order to address this issue, XCP-D optionally allows users to specify a band-stop or low-pass
filter to remove respiration-related signals from the motion parameters, prior to framewise
displacement calculation.
Please refer to :footcite:t:`fair2020correction` and :footcite:t:`gratton2020removal` for
more information.

.. important::
   Please note that the filtered motion parameters are **only** used to flag high-motion outliers.
   They will not be used in the confound regression step.

The two options for the motion-filtering parameter are "notch" (the band-stop filter) and
"lp" (the low-pass filter).

The cutoff points for either the notch filter
(the beginning and end of the frequency band to remove)
or the low-pass filter (the highest frequency to retain) can be set by the user
(see :ref:`usage_cli`), and may depend on the age of the participant.

Below are some recommendations for cutoff values when using the notch filter.

.. list-table:: Respiratory Filter

   *  - Age Range
      - Cutoff Range
        (Breaths per Minute)
   *  - < 1 year
      - 30 to  60
   *  - 1 to 2 years
      - 25 - 50
   *  - 2 - 6 years
      - 20 - 35
   *  - 6-12 years
      - 15 - 25
   *  - 12 - 18 years
      - 12 - 20
   *  - 19 - 65 years
      - 12 - 18
   *  - 65 - 80 years
      - 12 - 28
   *  - > 80 years
      - 10 - 30

If using the low-pass filter for single-band data, a recommended cutoff is 6 BPM (i.e., 0.1 Hertz),
per :footcite:t:`gratton2020removal`.


Framewise displacement calculation and thresholding
---------------------------------------------------
:func:`~xcp_d.workflows.postprocessing.init_prepare_confounds_wf`,
:class:`~xcp_d.interfaces.censoring.FlagMotionOutliers`,
:func:`~xcp_d.utils.modified_data.compute_fd`

Framewise displacement is then calculated according to the formula from Power et al. (CITE).
Two parameters that impact FD calculation and thresholding are
(1) the head radius used to convert rotation degrees to millimeters and
(2) the framewise displacement threshold.
The former may be set with the ``--head-radius`` parameter, which also has an "auto" option,
in which a brain mask from the preprocessing derivatives is loaded and
(treating the brain as a sphere) the radius is directly calculated
(see :func:`~xcp_d.utils.utils.estimate_brain_radius`).
The latter is set with the ``--fd-thresh`` parameter.

In this step, volumes with a framewise displacement value over the ``--fd-thresh`` parameter will
be flagged as "high motion outliers".
These volumes will later be removed from the denoised data.


Denoising
=========
:class:`~xcp_d.interfaces.nilearn.DenoiseNifti`, :class:`~xcp_d.interfaces.nilearn.DenoiseCifti`

Temporal censoring
------------------

Prior to confound regression, high-motion volumes will be removed from the BOLD data.
These volumes will also be removed from the nuisance regressors.
Please refer to :footcite:t:`power2012spurious` for more information.


Confound regression
-------------------

Prior to confound regression, all nuisance regressors, except the intercept regressor, will be
mean-centered.

.. admonition:: Handling of signal regressors

   In some cases, nuisance regressors share variance with signal regressors, in which case
   additional processing must be done before regression.
   One example of this is denoising using components from a spatial independent components
   analysis.
   With spatial ICA, each component's spatial weights are orthogonal to all other components,
   but the time series for the component may be correlated with other components.
   In common ICA-based denoising methods, such as AROMA or ME-ICA with tedana,
   components are classified as either "noise" or "signal".
   However, the "noise" components will often share variance with the "signal" components,
   so simply regressing the noise components out of the BOLD data,
   without considering the signal components, may remove signal of interest.

   To address this issue, XCP-D will look for signal regressors in the selected confounds.
   If any signal regressors are detected
   (i.e., if any columns in the confounds file have a ``signal__`` prefix),
   then the noise regressors will be orthogonalized with respect to the signal regressors,
   to produce "pure evil" regressors.

   This is done automatically for XCP-D's built-in nuisance strategies which include AROMA
   components, but users must manually add the ``signal__`` prefix to any signal regressors in
   their custom confounds files, if they choose to use them.

After censoring, mean-centering, and potential orthogonalization,
confound regression will be performed with a linear least-squares approach.
The parameter estimates for each of the confounds will be retained, along with the residuals from
the regression.
The residuals from regressing the censored BOLD data on the censored confounds will be referred to
as the ``denoised BOLD``.

Additionally, the parameter estimates from each of the censored confounds will be used calculate
residuals using the *full* (i.e., uncensored) confounds and BOLD data.
The residuals from this step will be referred to as the ``uncensored, denoised BOLD``.
The ``uncensored, denoised BOLD`` is later used for DCAN QC figures, but it is not written out to
the output directory.


Interpolation
-------------

An interpolated version of the ``denoised BOLD`` is then created by filling in the high-motion
outlier volumes with cubic spline interpolated data, as implemented in ``Nilearn``.
The resulting ``interpolated, denoised BOLD`` is primarily used for bandpass filtering.


Bandpass filtering [OPTIONAL]
-----------------------------

The ``interpolated, denoised BOLD`` is then bandpass filtered using a Butterworth filter.
The resulting ``filtered, interpolated, denoised BOLD`` will only be written out to the output
directory if the ``--dcan-qc`` flag is used, as users **should not** use interpolated data
directly.

Bandpass filtering can be disabled with the ``--disable-bandpass-filter`` flag.


Re-censoring
------------
:class:`~xcp_d.interfaces.censoring.Censor`

After bandpass filtering, high motion volumes are removed from the
``filtered, interpolated, denoised BOLD`` once again, to produce ``filtered, denoised BOLD``.
This is the primary output of XCP-D.


Resting-state derivative generation
===================================

For each BOLD run, resting-state derivatives are generated.
These include regional homogeneity (ReHo) and amplitude of low-frequency fluctuation (ALFF).

ALFF
----
:func:`~xcp_d.workflows.restingstate.init_alff_wf`

ALFF will only be calculated if the bandpass filter is enabled
(i.e., if the ``--disable-bandpass-filter`` flag is not used)
and censoring is disabled
(i.e., if ``--fd-thresh`` is set to a value less than or equal to zero).

Smoothed ALFF derivatives will also be generated if the ``--smoothing`` flag is used.


ReHo
----
:func:`~xcp_d.workflows.restingstate.init_reho_nifti_wf`,
:func:`~xcp_d.workflows.restingstate.init_reho_cifti_wf`


Parcellation and functional connectivity estimation
===================================================
:func:`~xcp_d.workflows.connectivity.init_functional_connectivity_nifti_wf`,
:func:`~xcp_d.workflows.connectivity.init_functional_connectivity_cifti_wf`

The ``filtered, denoised BOLD`` is fed into a functional connectivity workflow,
which extracts parcel-wise time series from the BOLD using several atlases:

   a.  Schaefer 100,200,300,400,500,600,700,800,900,1000
   b.  Glasser 360
   c.  Gordon 333
   d.  Tian Subcortical Atlas :footcite:p:`tian2020topographic`

The resulting parcellated time series for each atlas is then used to generate static functional
connectivity matrices, as measured with Pearson correlation coefficients.

For CIFTI data, both tab-delimited text file (TSV) and CIFTI versions of the parcellated time
series and correlation matrices are written out.


Smoothing [OPTIONAL]
====================
:func:`~xcp_d.workflows.postprocessing.init_resd_smoothing_wf`

The ``filtered, denoised BOLD`` may optionally be smoothed with a Gaussian kernel.
This smoothing kernel is set with the ``--smoothing`` parameter.


Concatenation of functional derivatives [OPTIONAL]
==================================================
:func:`~xcp_d.workflows.concatenation.init_concatenate_data_wf`

If the ``--combineruns`` flag is included, then BOLD runs will be grouped by task and concatenated.
Several concatenated derivatives will be generated, including the ``filtered, denoised BOLD``,
the ``smoothed, filtered, denoised BOLD``, the temporal mask, and the filtered motion parameters.

.. important::
   If a run does not have enough low-motion data and is skipped, then the concatenation workflow
   will not include that run.

.. important::
   If a set of related runs do not have enough low-motion data, then the concatenation workflow
   will automatically stop early, and no concatenated derivatives for that set of runs will be
   written out.


Quality control
===============
:func:`~xcp_d.workflows.plotting.init_qc_report_wf`

The quality control (QC) in ``XCP-D`` estimates the quality of BOLD data before and after
regression and also estimates BOLD-T1w coregistration and BOLD-Template normalization
qualites.
The QC metrics include the following:

   a. Motion parameters summary: mean FD, mean and maximum RMS
   b. Mean DVARs before and after regression and its relationship to FD
   c. BOLD-T1w coregistration quality - Dice similarity index, Coverage and Pearson correlation
   d. BOLD-Template normalization quality - Dice similarity index, Coverage and Pearson correlation


*******
Outputs
*******

XCP-D generates four main types of outputs for every subject.

First, XCP-D generates an HTML "executive summary" that displays relevant information about the
anatomical data and the BOLD data before and after regression.
The anatomical image viewer allows the user to see the segmentation overlaid on the anatomical
image.
Next, for each session, the user can see the segmentation registered onto the BOLD images.
Alongside this image, pre and post regression "carpet" plot is alongside DVARS, FD, the global
signal.
The number of volumes remaining at various FD thresholds are shown.

Second, XCP-D generates an HTML "report" for each subject and session.
The report contains a Processing Summary with QC values, with the BOLD volume space, the TR,
mean FD, mean RMSD, and mean and maximum RMS,
the correlation between DVARS and FD before and after processing, and the number of volumes
censored.
Next, pre and post regression "carpet" plots are alongside DVARS and FD.
An About section that notes the release version of XCP-D, a Methods section that can be copied and
pasted into the user's paper,
which is customized based on command line options, and an Error section, which will read
"No errors to report!" if no errors are found.

Third, XCP-D outputs processed BOLD data, including denoised unsmoothed and smoothed timeseries in
MNI152NLin2009cAsym and fsLR-32k spaces, parcellated time series, functional connectivity matrices,
and ALFF and ReHo (smoothed and unsmoothed).

Fourth, the anatomical data (processed T1w processed and segmentation files) are copied from
fMRIPrep.
If both images are not in MNI152NLin6Asym space, they are resampled to MNI space.
The fMRIPrep surfaces (gifti files) in each subject are also resampled to standard space
(fsLR-32K).

See :doc:`outputs` for details about XCP-D outputs.

**********
References
**********

.. footbibliography::
