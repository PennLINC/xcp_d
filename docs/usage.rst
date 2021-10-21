.. include:: links.rst

.. _Usage :

Usage Notes
===========



XCP_ABCD Execution 
------------------
The *xcp_abcd* workflow takes  fMRIPRep outputs in the form of BIDS derivatives.  
The fMRIPRep outputs  are required to include at least anatomical and functional outputs 
with at least one preprocessed BOLD image. 

The exact command to run in *xcp_abcd* depends on the Installation_ method and 
data that needs to be processed
The basic command of *xcp_abcd* is as follow
Example: ::

    xcp_abcd fmriprepdir output participant  # for nifti 

    xcp_abcd fmriprepdir output participant  --cifti  # for cifti


Command-Line Arguments
----------------------
.. argparse::
   :ref: xcp_abcd.cli.run.get_parser
   :prog: xcp_abcd
   :nodefault:
   :nodefaultconst:
Troubleshooting
---------------
Logs and crashfiles are outputted into the
``<output dir>/xcp_abcd/sub-<participant_label>/log`` directory.
Information on how to customize and understand these files can be found on the
`nipype debugging <http://nipype.readthedocs.io/en/latest/users/debug.html>`_
page.

**Support and communication**.
The documentation of this project is found here: https://xcp-abcd.readthedocs.io/.

All bugs, concerns and enhancement requests for this software can be submitted here:
https://github.com/PennLINC/xcp_abcd/issues.
