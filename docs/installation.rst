.. include:: links.rst

.. _Installation :

Installation
============

There are two ways to install *XCP-D*:

* within a `Manually Prepared Environment (Python 3.8+)`_, also known as
  *bare-metal installation*; or
* using container technologies (RECOMMENDED).

The command-line options are documented in the :ref:`Usage` section.
After the setup of the environment, ``xcp_d`` can be executed on the 
command line, processesing fMRIPrep outputs, using the following command-line structure
::
   $ xcp_d <fmri_pdir> <outputdir> <options>

However, we strongly recommend using a container infrastructure. Here, the command-line will be composed of a preamble to configure the container execution followed by the ``xcp_d`` command-line options as if you were running it on a *bare-metal* installation.

The command-line structure above is then modified as follows:
::
  $<container_command_and_options> <xcp_d_container_image> <fmri_dir> <outputdir> <options>

Therefore, once a user specifies the container options and the image to be run, the command line is the same as for the ordinary installation, but dropping the ``xcp_d`` executable name.

Container technologies: Docker and Singularity
==============================================
*XCP-D* is a *NiPreps* application, and therefore follows some overarching principles of containerized execution drawn from the BIDS-Apps protocols. For detailed information of containerized execution of *NiPreps*, please visit the corresponding `Docker <https://www.nipreps.org/apps/docker/>`__ or `Singularity <https://www.nipreps.org/apps/singularity/>`__ subsections.
The *NiPreps* portal also containes `extended details of execution with the Docker wrapper <https://www.nipreps.org/apps/docker/#running-a-niprep-with-a-lightweight-wrapper>`__.

For security reasons, many :abbr:`HPCs (High-Performance Computing)` do not allow Docker containers, but do allow Singularity_ containers. The improved security for multi-tenant systems comes at the price of some limitations and extra steps necessary for execution.

Manually Prepared Environment (Python 3.8+)
===========================================
.. warning::

   This method is not recommended! Please use container alternatives
   in :ref:`run_docker`, and :ref:`run_singularity`.

``xcp_d`` requires some `External Dependencies`_. These tools must be installed and their binaries available in the system's ``$PATH``.

On a functional Python 3.8 (or above) environment with ``pip`` installed,
``xcp_d`` can be installed using the habitual command
::

    $ pip install git+https://github.com/pennlinc/xcp_d.git

Check your installation with the ``--version`` argument
::

    $ xcp_d --version


External Dependencies
---------------------

*XCP-D* is written using Python 3.8 (or above), and is based on
nipype_.

*XCP-D* requires some other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``xcp_d`` package:

- FSL_ (version 5.0.9 or higher)
- ANTs_ (version 2.2.0 - or higher)
- AFNI_ (version Debian-16.2.07)
- `bids-validator <https://github.com/bids-standard/bids-validator>`_ (version 1.6.0)
- `connectome-workbench <https://www.humanconnectome.org/software/connectome-workbench>`_ (version Debian-1.3.2)
