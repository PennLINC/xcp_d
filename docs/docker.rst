.. include:: links.rst

.. _run_docker:

Running *xcp_d* via Docker containers
========================================

In order to run docker smoothly, it is best to prevent permissions issues associated with the root file system. Running docker as user on the host will ensure the ownership of files written during the container execution.

A ``docker`` container can be created using the following command::

    $ docker run --rm -it \
   -v /fmriprepdata:/in/ \
   -v /tmp/wkdir:/wkdir/ \
   -v /tmp:/scrth/ \
   -v /tmp/xcpd_ciftiF/:/out. \
   pennlinc/xcp_d:latest \
   /in/ /out/ pariticipant \

See :ref:`usage` for more information.
