.. include:: links.rst

.. _Usage:

Usage Notes
===========

XCP-D Execution 
------------------
The *xcp_d* workflow takes `fMRIPRep`, `NiBabies`, `abcd-hcp-pipeline` and `HCP` outputs in the form of BIDS derivatives.  
The outputs  are required to include at least anatomical and functional outputs with at least one preprocessed BOLD image. 

The exact command to run in *xcp_d* depends on the Installation_ method and data that needs to be processed

Using the *bare-metal* installation, ``xcp_d`` can be executed on the command line, processesing fMRIPrep outputs, using the following command-line structure
::
   $ xcp_d <fmri_pdir> <outputdir> <options>

However, we strongly recommend using a container infrastructure. Here, the command-line will be composed of a preamble to configure the container execution followed by the ``xcp_d`` command-line options as if you were running it on a *bare-metal* installation.

The command-line structure above is then modified as follows:
::
  $<container_command_and_options> <xcp_d_container_image> <fmri_dir> <outputdir> <options>

Therefore, once a user specifies the container options and the image to be run, the command line is the same as for the ordinary installation, but dropping the ``xcp_d`` executable name.

The basic command of *xcp_d* is:
::

    xcp_d inputpdir output  # for nifti 

    xcp_d inputpdir output  --cifti  # for cifti

The `abcd-hcp-pipeline` and `HCP` outputs are  not in the form of bids derivatives and required to be specified in the command line :: 

    xcp_d inputpdir output  --input-type dcan  # for abcd-hcp-pipeline
    xcp_d inputpdir output  --input-type hcp  # for HCP

It is advisable process abcd-hcp-pipeline outputs by participant by adding `--participant-label` to the command line.

Command-Line Arguments
----------------------
.. argparse::
   :ref: xcp_d.cli.run.get_parser
   :prog: xcp_d
   :nodefault:
   :nodefaultconst:

Troubleshooting
---------------
Logs and crashfiles are outputted into the
``<output dir>/xcp_d/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://xcp-abcd.readthedocs.io/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/PennLINC/xcp_d/issues.
