.. include:: links.rst

===============
Running XCP-D
===============

Inputs
===============
The *XCP-D* workflow takes `fMRIPRep`, `NiBabies`, `abcd-hcp-pipeline` and `HCP` outputs in the form of BIDS derivatives. The outputs are required to include at least anatomical and functional outputs with at least one preprocessed BOLD image. In these examples, we use an fmriprep output directory.

General Command Structure
------------------
The exact command to run in *xcp_d* depends on the Installation_ method and data that needs to be processed. We start first with the the *bare-metal* :ref:`Manually Prepared Environment (Python 3.8+)` installation, as the command line is simpler. ``xcp_d`` can be executed on the command line, processesing fMRIPrep outputs, using the following command-line structure, for example:
::
   $ xcp_d <fmriprep_dir> <outputdir> --cifti --despike  --head_radius 40 -w /wkdir --smoothing 6

However, we strongly recommend using :any:`container-tech`. Here, the command-line will be composed of a preamble to configure the container execution followed by the ``xcp_d`` command-line options as if you were running it on a *bare-metal* installation.

Docker
===============
If you are computing locally, we recommend Docker. See :any:`docker-install` for installation questions.
::
   $ docker run --rm -it \
   -v /fmriprepdata:/data/ \
   -v /tmp/wkdir:/wkdir \
   -v /tmp:/scrth \
   -v /tmp/xcpd_ciftiF/:/out \
   pennlinc/xcp_d:latest \
   /data/fmriprep /out \
   --cifti --despike  --head_radius 40 -w /wkdir --smoothing 6

Singularity
===============
If you are computing on a :abbr:`HPC (High-Performance Computing)`, we recommend Singularity. See :any:`singularity-install` for installation questions.
::

    $ singularity run --cleanenv xcp_d.simg \
      path/to/data/fmri_dir  path/to/output/dir \
      --participant-label label


**Relevant aspects of the** ``$HOME`` **directory within the container**.
By default, Singularity will bind the user's ``$HOME`` directory on the host
into the ``/home/$USER`` directory (or equivalent) in the container.
Most of the time, it will also redefine the ``$HOME`` environment variable and
update it to point to the corresponding mount point in ``/home/$USER``.
However, these defaults can be overwritten in your system.
It is recommended that you check your settings with your system's administrator.
If your Singularity installation allows it, you can work around the ``$HOME``
specification, combining the bind mounts argument (``-B``) with the home overwrite
argument (``--home``) as follows: ::

    $ singularity run -B $HOME:/home/xcp --home /home/xcp \
          --cleanenv xcp_d.simg <xcp_d arguments>


Therefore, once a user specifies the container options and the image to be run, the command line options are the same as the *bare-metal* installation.

Command-Line Arguments
===============
.. argparse::
   :ref: xcp_d.cli.run.get_parser
   :prog: xcp_d
   :nodefault:
   :nodefaultconst:

Troubleshooting
===============
Logs and crashfiles are outputted into the
``<output dir>/xcp_d/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://xcp-abcd.readthedocs.io/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/PennLINC/xcp_d/issues.
