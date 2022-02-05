.. include:: links.rst


.. _run_singularity:

Running *xcp_d* via Singularity containers
=============================================

If the data to be preprocessed is also on the HPC or a personal computer, you are ready to run *xcp_d*. ::

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
