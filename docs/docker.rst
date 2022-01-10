.. include:: links.rst

.. _run_docker:

Running *xcp_d* via Docker containers
========================================
For every new version of *xcp_d* that is released, a corresponding Docker
image is generated.

In order to run *xcp_d* via Docker images, the Docker Engine must be installed.

If you have used *xcp_d* via Docker in the past, you might need to pull down a
more recent version of the image: ::

    $ docker pull pennlinc/xcp_d:<latest-version>

The image can also be found here: https://registry.hub.docker.com/r/pennlinc/xcp_d

*xcp_d* can be run interacting directly with the Docker Engine via the `docker run`
command, or through a lightweight wrapper that was created for convenience.


Running *xcp_d* directly with the Docker Engine
--------------------------------------------------------------
**Running containers as a user**

In order to run docker smoothly, it is best to prevent permissions issues
associated with the root file system. Running docker as user on the host will
ensure the ownership of files written during the container execution.

A ``docker`` container can be created using the following command::

    $ docker run -ti --rm \
        -v path/to/data:/fmriprep_output:ro \
        -v path/to/output:/out \
        pennlinc/xcp_d:<latest-version> \
        /data /out/out 

See :ref:`usage` for more information.
