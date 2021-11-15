.. include:: links.rst

------------
Installation
------------

There are two ways to get *xcp_abcd* installed:

* within a `Manually Prepared Environment (Python 3.8+)`_,
   or
* with container technologies (RECOMMENDED), such as :ref:`run_docker`
  or :ref:`run_singularity`.

After the setup of the environment, ``xcp_abcd`` can be executed on the 
command line. The command-line options are documented in the :ref:`usage`
section.

``xcp_abcd`` processes the fMRIPrep outputs using the following command-line structure::
   $ xcp_abcd <fmriprepdir> <outputdir> <analysis_level> <options>


With either singularity or docker, the command-line will be composed of a 
preamble to configure the container execution followed by the ``xcp_abcd``
command-line options.

The command-line structure above is then modified as follows::
  $<container_command_and_options> <container_image> \
      <fmriprepdir> <outputdir> <analysis_level> <options>

Therefore, once a user specifies the container options and the image to be run
the command line is the same as for the ordinary installation, but dropping
the ``xcp_abcd`` executable name.

Container technologies: Docker and Singularity
==============================================
Container technologies are operating-system-level virtualization methods to run Linux systems
using the host's Linux kernel.
This is a lightweight approach to virtualization, as compared to virtual machines.


.. _docker_installation:

Docker (recommended for PC/laptop and commercial cloud)
-------------------------------------------------------
XCP_ABCD runs fast and easily with less memory requiremen on 
PC/laptop with Docker. The instructions (`Docker installation`_) are easier to follow.
Check  `docker <https://docs.docker.com/get-started/>`_ for more information about
installtion of docker and how to get started. The list of Docker images that are ready to 
use is found at the `Docker Hub`_,


Singularity (recommended for HPC)
---------------------------------

For security reasons, many :abbr:`HPCs (High-Performance Computing)`
do not allow Docker containers, but do allow Singularity_ containers.
The improved security for multi-tenant systems comes at the price of some limitations
and extra steps necessary for execution.


Manually Prepared Environment (Python 3.8+)
===========================================


.. warning::

   This method is not recommended! Please checkout container alternatives
   in :ref:`run_docker`, and :ref:`run_singularity`.

XCP_ABCD requires some `External Dependencies`_. These tools must be installed 
and their binaries available in the system's ``$PATH``.

On a functional Python 3.8 (or above) environment with ``pip`` installed,
*xcp_abcd* can be installed using the habitual command ::

    $ pip install git+https://github.com/pennlinc/xcp_abcd.git

Check your installation with the ``--version`` argument ::

    $ xcp_abcd --version


External Dependencies
---------------------

*XCP_ABCD* is written using Python 3.8 (or above), and is based on
nipype_.

*XCP_ABCD* requires some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``*XCP_ABCD*`` package:

- FSL_ (version 5.0.9 or higher)
- ANTs_ (version 2.2.0 - or higher)
- AFNI_ (version Debian-16.2.07)
- `bids-validator <https://github.com/bids-standard/bids-validator>`_ (version 1.6.0)
- `connectome-workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ (version Debian-1.3.2)
