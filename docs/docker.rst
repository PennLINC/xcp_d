.. include:: links.rst

.. _run_docker:

Running *xcp_d* via Docker containers
========================================

In order to run docker smoothly, it is best to prevent permissions issues associated with the root file system. Running docker as user on the host will ensure the ownership of files written during the container execution.

A ``docker`` container can be created using the following command::

    $ docker run -ti --rm \
        -v path/to/data:/fmriprep_output:ro \
        -v path/to/output:/out \
        pennlinc/xcp_d:<latest-version> \
        /data /out/out 

See :ref:`usage` for more information.
