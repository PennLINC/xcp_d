.. include:: links.rst

##########################
Frequently Asked Questions
##########################


.. _upgrading:

*********
Upgrading
*********

When you run *XCP-D*, it checks whether a newer release is available and whether the version
you are running has been flagged.
If a newer release exists, *XCP-D* prints a message like::

    You are using XCP-D v26.1.0, and a newer version of XCP-D is available: v26.2.0.

If your version has been flagged, *XCP-D* prints a warning with the reason.
A flagged version has a known, severe flaw that can affect its outputs,
and we strongly discourage using it.

Both checks are informational only and never stop *XCP-D* from running.
They need internet access; if *XCP-D* cannot reach Docker Hub or GitHub,
it skips the corresponding check silently.
The latest release is looked up from the
`XCP-D image tags on Docker Hub <https://hub.docker.com/r/pennlinc/xcp_d/tags>`_
and cached in ``~/.cache/xcp_d/latest`` for 14 days.
The list of flagged versions lives in the
`.versions.json <https://github.com/PennLINC/xcp_d/blob/main/.versions.json>`_
file in the *XCP-D* repository.


When should I upgrade?
======================

-  **If your version is flagged**, upgrade, and reprocess any data that the flagged issue
   affects.
-  **If you are starting a new study or analysis**, use the latest release.
-  **If you are in the middle of a study**, keep using the same version for all of its
   participants.
   Different versions can produce different outputs, so mixing versions within one analysis
   adds variability unrelated to your data.
   Upgrade mid-study only if the new release fixes a bug that affects you,
   and then reprocess every participant with the new version.

To find out what changed between versions, see :doc:`changes`
or the `GitHub releases page <https://github.com/PennLINC/xcp_d/releases>`_.

Always record the *XCP-D* version you used.
It is also written to the ``dataset_description.json`` file in the output directory.


How do I upgrade?
=================

Pull the image for the new version, replacing ``<version>`` with the release you want.

With Docker::

    $ docker pull pennlinc/xcp_d:<version>

With Apptainer::

    $ apptainer build xcp_d-<version>.sif docker://pennlinc/xcp_d:<version>

For a bare-metal installation, reinstall from the release tag::

    $ pip install --upgrade git+https://github.com/PennLINC/xcp_d.git@<version>

Then confirm the version with ``xcp_d --version``.

Use a new output directory and a new working directory when you change versions.
A working directory from an older version may contain cached intermediate results,
which the new version could reuse instead of recomputing.
