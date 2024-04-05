.. include:: links.rst

#######################
Contributing to *XCP-D*
#######################

This guide describes the organization and preferred coding style for XCP-D,
for prospective code contributors.

.. important::

   We ask all users and contributors to follow *XCP-D*'s
   `Code of Conduct <https://github.com/PennLINC/xcp_d/blob/main/CODE_OF_CONDUCT.md>`_.

Development in Docker is encouraged, for the sake of consistency and portability.
By default, work should be built off of
`pennlinc/xcp_d:unstable <https://hub.docker.com/r/pennlinc/xcp_d/>`_,
which tracks the ``main`` branch,
or ``pennlinc/xcp_d:latest``,
which tracks the latest release
(see :doc:`installation` for the basic procedure for running).


***********
Style guide
***********

*XCP-D* is a Python library, primarily written as Nipype interfaces and workflows.
As such, we follow a specific coding style adapted to both Python generally and Nipype
specifically.

*XCP-D* uses a number of ``flake8`` extensions to evaluate coding style,
but the main ones are ``black`` and ``isort``.

.. tip::

   Please run ``isort`` and ``black`` on the ``XCP-D`` codebase before opening a pull request.

   Also run ``flake8`` to check for any additional style issues.


Workflow coding style
=====================

Workflows should largely be written just like other functions (which should go in ``xcp_d.utils``),
except for a couple of elements.

First, ``black`` tends to format workflow connections awkwardly, so contributors should add
``# fmt:skip`` after each workflow connection statement.

Second, workflow connections should occur shortly after the associated nodes are defined,
rather than all at once at the end of the function.

Here is an example of a basic workflow, written in the preferred style for *XCP-D*.

.. code-block:: python

   from nipype.interfaces import utility as niu
   from nipype.pipeline import engine as pe
   from niworkflows.engine.workflows import LiterateWorkflow as Workflow


   def init_example_workflow_wf(name="example_workflow_wf"):
      """Create an example workflow.

      Workflow Graph
         .. workflow::
            :graph2use: orig
            :simple_form: yes

            from xcp_d.workflows.example import init_example_workflow_wf

            wf = init_example_workflow_wf(name="example_workflow_wf")

      Parameters
      ----------
      name : str
         Name of the workflow. Default is "example_workflow_wf".

      Inputs
      ------
      a : str
         Input a.
      b : str
         Input b.

      Outputs
      -------
      a : str
         Input a.
      b : str
         Input b.
      """
      workflow = LiterateWorkflow(name=name)

      inputnode = pe.Node(niu.IdentityInterface(fields=["a", "b"]), name="inputnode")
      outputnode = pe.Node(niu.IdentityInterface(fields=["a", "b"]), name="outputnode")

      workflow.connect([
         (inputnode, outputnode, [
            ("a", "a"),
            ("b", "b"),
         ]),
      ])  # fmt:skip

      return workflow


*************************************************
Contributing to XCP-D without adding dependencies
*************************************************

In the most common case, you will want to modify *XCP-D*'s Python code without adding any
dependencies (Python or not) to the Docker image.
In this situation, you can use the ``unstable`` Docker image without having to build a new Docker
image yourself.

1. Pull the ``unstable`` Docker image.

   .. code-block:: bash

      docker pull pennlinc/xcp_d:unstable

2. Fork the XCP-D repository to your GitHub account.
   For more information on contributing via the fork-and-branch approach,
   see `GitHub's contributing guide
   <https://docs.github.com/en/get-started/quickstart/contributing-to-projects>`_.

3. Clone your forked repository to your local machine.

4. Create a branch to work on.
   **Make sure your branch is up to date with XCP-D's ``main`` branch before making any
   changes!**

5. Make changes to the codebase that you want to try out.

6. Test out your changes by running the Docker container.
   The trick here is to mount your modified version of *XCP-D* into the Docker container,
   overwriting the container's version.
   This way, your Docker container will run using your modified code,
   rather than the original version.

   You can do this by running the Docker image as described in :doc:`usage`,
   but adding in a mount point for your code:

   .. code-block:: bash

      docker run \
         -v /path/to/local/xcp_d:/usr/local/miniconda/lib/python3.10/site-packages/xcp_d \
         pennlinc/xcp_d:unstable \
         ...  # see the usage documentation for info on what else to include in this command

7. Push your changes to GitHub.

8. Open a pull request to PennLINC/XCP-D's ``main`` branch.
   Please follow `NiPreps contributing guidelines <https://www.nipreps.org/community/>`_
   when preparing a pull request.


Running tests
=============

While CircleCI will automatically run *XCP-D*'s tests for any open PRs,
we strongly recommend running at least some tests locally, to make sure your proposed changes work.

*XCP-D* has a file, ``xcp_d/tests/run_local_tests.py``, that builds Docker ``run`` commands to
run selected tests.
Please use that script to run some tests locally before opening your PR.

If all tests pass, this is a strong indication that your proposed changes do not introduce bugs.
However, we strongly recommend reviewing the output files- especially the HTML reports-
from the integration tests to see how your proposed changes affect the primary outputs from
*XCP-D*.


********************************
Adding or modifying dependencies
********************************

If you think *XCP-D* needs to use a library (Python or not) that is not installed in the Docker
image already, then you will need to build a new Docker image to test out your proposed changes.

*XCP-D* uses a "base Docker image" defined in https://github.com/PennLINC/xcpd_build.
We try to define the majority of non-Python requirements in that Docker image.
If you want to add or modify a non-Python dependency, then you will need to clone that repository
modify its Dockerfile, and build its Docker image to ensure that the new dependency installs
correctly.
Once that's done, you can open a pull request to the ``xcpd_build`` repository with your change.

.. tip::

   Given that this method requires contributing to two repositories, it's a good idea to link to
   the associated *XCP-D* issue in your ``xcpd_build`` PR.

For Python dependencies, you can update the requirements defined in *XCP-D*'s ``setup.cfg``
and rebuild the *XCP-D* Docker image locally to test out your change.
Once your change is working, you can open a pull request to the *XCP-D* repo.


*****************
Maintaining XCP-D
*****************


Making a Release
================

To make an *XCP-D* release, complete the following steps:

1.  Choose a new version tag, according to the semantic versioning standard.
2.  In GitHub's release tool, draft a new release.

   #. Create a new tag. Use semantic versioning terminology (e.g., ``1.0.0``).
      For pre-releases, use release candidate terminology (e.g., ``0.0.12rc1``).
   #. Define a release title. The release title should be the same as the new tag
      (e.g., ``1.0.0``).
   #. For pre-releases, select the "This is a pre-release" option.
   #. Select the "Generate release notes" button.
      This will create most of the necessary release notes based on XCP-D's config file.
   #. At the top of the release notes, add some information summarizing the release.

3. Once the release notes have been completed, open a new PR with the following changes:

   #. Copy the release notes over to CHANGES.md.
   #. Modify the CITATION.cff file, updating the version number and release date.

5. Once the new PR has been merged, you can publish the release.
   This will make the release on GitHub and will trigger a CircleCI job to push the new version to
   DockerHub.
