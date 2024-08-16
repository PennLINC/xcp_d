.. include:: links.rst

############
Installation
############

There are two ways to install *XCP-D*:

* using `Container Technologies`_ (RECOMMENDED)
* within a `Manually Prepared Environment (Python 3.10)`_, also known as *bare-metal installation*

.. _installation_container_technologies:

**********************
Container Technologies
**********************
*XCP-D* is ideally run via a Docker or Apptainer container. If you are running *XCP-D* locally,
we suggest Docker_.
However, for security reasons, many :abbr:`HPCs (High-Performance Computing)` do not allow Docker
containers, but do allow Apptainer_ containers.
The improved security for multi-tenant systems comes at the price of some limitations and extra
steps necessary for execution.


.. _installation_docker:

Docker Installation
===================
For every new version of *xcp_d* that is released, a corresponding Docker image is generated.

In order to run *xcp_d* via Docker images, the Docker Engine must be installed.

If you have used *xcp_d* via Docker in the past, you might need to pull down a more recent version
of the image: ::

    $ docker pull pennlinc/xcp_d:<latest-version>

The image can also be found here: https://hub.docker.com/r/pennlinc/xcp_d

*xcp_d* can be run interacting directly with the Docker Engine via the `docker run` command,
or through a lightweight wrapper that was created for convenience.


.. _installation_apptainer:

Apptainer Installation
========================

You can create an Apptainer image directly on the system using the following
command: ::

    $ apptainer build xcp_d-<version>.simg docker://pennlinc/xcp_d:<version>

where ``<version>`` should be replaced with the desired version of *xcp-d* that you want to
download.


.. _installation_manually_prepared_environment:

*******************************************
Manually Prepared Environment (Python 3.10)
*******************************************
.. warning::

   This method is not recommended! Please use container alternatives
   in :ref:`run_docker`, and :ref:`run_apptainer`.

XCP-D requires some `External Dependencies`_.
These tools must be installed and their binaries available in the system's ``$PATH``.

On a functional Python 3.10 environment with ``pip`` installed,
XCP-D can be installed using the habitual command
::

    $ pip install git+https://github.com/pennlinc/xcp_d.git

Check your installation with the ``--version`` argument
::

    $ xcp_d --version


*********************
External Dependencies
*********************

*XCP-D* is written using Python 3.10, is based on nipype_,
and requires some other neuroimaging software tools that are not handled by the Python's packaging
system (PyPi) used to deploy the XCP-D package:

-  ANTs_ (version 2.2.0 - or higher)
-  AFNI_ (version Debian-16.2.07)
-  `bids-validator <https://github.com/bids-standard/bids-validator>`_ (version 1.6.0)
-  `connectome-workbench <https://www.humanconnectome.org/software/connectome-workbench>`_
   (version Debian-1.3.2)
