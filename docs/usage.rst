.. include:: links.rst

=============
Running XCP-D
=============

.. warning::

   XCP-D may not run correctly on **M1 chips**.

.. _usage_inputs:

Execution and Input Formats
===========================

The *XCP-D* workflow takes `fMRIPRep`, `NiBabies`, `DCAN` and `HCP` outputs in the form of BIDS derivatives.
In these examples, we use an fmriprep output directory.

The outputs are required to include at least anatomical and functional outputs with at least one preprocessed BOLD image.
Additionally, each of theseshould be in directories that can be parsed by the BIDS online validator
(even if it is not BIDS valid - we do not require BIDS valid directories).
The directories must also include a valid `dataset_description.json`.

The exact command to run in *xcp_d* depends on the `installation`_ method and data that needs to be processed.
We start first with the *bare-metal* :ref:`installation_manually_prepared_environment` installation,
as the command line is simpler.
``xcp_d`` can be executed on the command line, processesing fMRIPrep outputs, using the following command-line structure, for example:

.. code-block:: bash

   xcp_d <fmriprep_dir> <outputdir> --cifti --despike  --head_radius 40 -w /wkdir --smoothing 6

However, we strongly recommend using :ref:`installation_container_technologies`.
Here, the command-line will be composed of a preamble to configure the container execution,
followed by the ``xcp_d`` command-line options as if you were running it on a *bare-metal* installation.

.. _usage_cli:

Command-Line Arguments
======================

.. argparse::
   :ref: xcp_d.cli.run.get_parser
   :prog: xcp_d
   :nodefault:
   :nodefaultconst:


.. _run_docker:

Running ``xcp_d`` via Docker containers
========================================

If you are running ``xcp_d`` locally, we recommend Docker.
See :ref:`installation_container_technologies` for installation instructions.

In order to run Docker smoothly, it is best to prevent permissions issues associated with the root file system.
Running Docker as user on the host will ensure the ownership of files written during the container execution.

A Docker container can be created using the following command:

.. code-block:: bash

   docker run --rm -it \
      -v /dset/derivatives/fmriprep:/fmriprep:ro \
      -v /tmp/wkdir:/work:rw \
      -v /dset/derivatives:/out:rw \
      -v /dset/derivatives/freesurfer:/freesurfer:ro \  # Necessary for fMRIPrep versions <22.0.2
      pennlinc/xcp_d:latest \
      /fmriprep /out participant \
      --cifti --despike --head_radius 40 -w /work --smoothing 6

.. _run_singularity:

Running ``xcp_d`` via Singularity containers
============================================

If you are computing on an :abbr:`HPC (High-Performance Computing)`, we recommend using Singularity.
See :ref:`installation_container_technologies` for installation instructions.

If the data to be preprocessed is also on the HPC or a personal computer, you are ready to run *xcp_d*.

.. code-block:: bash

    singularity run --cleanenv xcp_d.simg \
        path/to/data/fmri_dir  \
        path/to/output/dir \
        --participant-label label

Relevant aspects of the ``$HOME`` directory within the container
------------------------------------------------------------------

By default, Singularity will bind the user's ``$HOME`` directory on the host
into the ``/home/$USER`` directory (or equivalent) in the container.
Most of the time, it will also redefine the ``$HOME`` environment variable and
update it to point to the corresponding mount point in ``/home/$USER``.
However, these defaults can be overwritten in your system.
It is recommended that you check your settings with your system's administrator.
If your Singularity installation allows it, you can work around the ``$HOME``
specification, combining the bind mounts argument (``-B``) with the home overwrite
argument (``--home``) as follows:

.. code-block:: bash

    singularity run -B $HOME:/home/xcp \
        --home /home/xcp \
        --cleanenv xcp_d.simg \
        <xcp_d arguments>

Therefore, once a user specifies the container options and the image to be run,
the command line options are the same as the *bare-metal* installation.

.. _usage_custom_confounds:

Custom Confounds
================

XCP-D can include custom confounds in its denoising.
Here, you can supply your confounds, and optionally add these to a confound strategy already supported in XCP-D.

To add custom confounds to your workflow, use the ``--custom-confounds`` parameter,
and provide a folder containing the custom confounds files for all of the subjects, sessions, and tasks you plan to post-process.

The individual confounds files should be tab-delimited, with one column for each regressor,
and one row for each volume in the data being denoised.

If you want to regress task-related signals out of your data, you can use the custom confounds option to do it.

Task Regression
---------------

Here we document how to include task effects as confounds.

.. tip::
   The basic approach to task regression is to convolve your task regressors with an HRF,
   then save those regressors to a custom confounds file.

.. warning::
   This method is still under development.

We recommend using a tool like Nilearn to generate convolved regressors from BIDS events files.
See `this example <https://nilearn.github.io/stable/auto_examples/04_glm_first_level/plot_design_matrix.html#create-design-matrices>`_.

.. code-block:: python

   import numpy as np
   from nilearn.glm.first_level import make_first_level_design_matrix

   N_VOLUMES = 200
   TR = 0.8
   frame_times = np.arange(N_VOLUMES) * TR
   events_df = pd.read_table("sub-X_ses-Y_task-Z_run-01_events.tsv")

   task_confounds = make_first_level_design_matrix(
      frame_times,
      events_df,
      drift_model=None,
      add_regs=None,
      hrf_model="spm",
   )

   # The design matrix will include a constant column, which we should drop
   task_confounds.drop(columns="constant")

   task_confounds.to_csv(
      "/path/to/custom_confounds/sub-X_ses-Y_task-Z_run-01_desc-confounds_timeseries.tsv",
      sep="\t",
      index=False,
   )

Then, when you run XCP-D, you can use the flag ``--custom-confounds /path/to/custom_confounds``.

Command Line XCP-D with Custom Confounds
````````````````````````````````````````

Last, supply the ``${subid}_${sesid}_task-${taskid}_desc-custom_timeseries.tsv`` file to xcp_d with ``-c`` option.
``-c`` should point to the directory where this file exists, rather than to the file itself;
``xcp_d`` will identify the correct file based on the subid, sesid, and taskid.
You can simultaneously perform additional confound regression by including, for example, ``-p 36P`` to the call.

.. code-block:: bash

   singularity run --cleanenv -B /my/project/directory:/mnt xcpabcd_latest.simg \
      /mnt/input/fmriprep \
      /mnt/output/directory \
      participant \
      --despike --lower-bpf 0.01 --upper-bpf 0.08 \
      --participant_label $subid -p 36P -f 10 \
      -t emotionid -c /mnt/taskarray_file_dir

Custom Parcellations
====================
While XCP-D comes with many built in parcellations, we understand that many users will want to use custom parcellations.
We suggest running XCP-D with the ``-cifti`` option (assuming you have cifti files),
and then using the Human Connectome Project wb_command to generate the time series:

.. code-block:: bash

   wb_command \
      -cifti-parcellate \
      {SUB}_ses-{SESSION}_task-{TASK}_run-{RUN}_space-fsLR_den-91k_desc-residual_bold.dtseries.nii \
      your_parcels.dlabel \
      {SUB}_ses-{SESSION}_task-{TASK}_run-{RUN}_space-fsLR_den-91k_desc-residual_bold.ptseries.nii

After this, if one wishes to have a connectivity matrix:

.. code-block:: bash

   wb_command \
      -cifti-correlation \
      {SUB}_ses-{SESSION}_task-{TASK}_run-{RUN}_space-fsLR_den-91k_desc-residual_bold.ptseries.nii \
      {SUB}_ses-{SESSION}_task-{TASK}_run-{RUN}_space-fsLR_den-91k_desc-residual_bold.pconn.nii

More information can be found at the HCP `documentation <https://www.humanconnectome.org/software/workbench-command>`_

Troubleshooting
===============

Logs and crashfiles are outputted into the ``<output dir>/xcp_d/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://xcp-d.readthedocs.io/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/PennLINC/xcp_d/issues.

If you have a question about using ``xcp_d``, please create a new topic on `NeuroStars <https://neurostars.org>`_ with the `"xcp_d" tag <https://neurostars.org/tag/xcp_d>`_.
The ``xcp_d`` developers follow NeuroStars, and will be able to answer your question there.
