.. include:: links.rst

================
Developers - API
================

xcp_d-combineqc
---------------

.. argparse::
   :ref: xcp_d.cli.combineqc.get_parser
   :prog: xcp_d-combineqc


:mod:`xcp_d.workflow`: Workflows
--------------------------------

.. automodule:: xcp_d.workflow
   :no-members:
   :no-inherited-members:

.. currentmodule:: xcp_d

.. autosummary::
   :toctree: generated/
   :template: module.rst

   xcp_d.workflow.base
   xcp_d.workflow.bold
   xcp_d.workflow.cifti
   xcp_d.workflow.anatomical
   xcp_d.workflow.connectivity
   xcp_d.workflow.execsummary
   xcp_d.workflow.outputs
   xcp_d.workflow.plotting
   xcp_d.workflow.postprocessing
   xcp_d.workflow.restingstate

:mod:`xcp_d.interfaces`: Nipype Interfaces
------------------------------------------

.. automodule:: xcp_d.interfaces
   :no-members:
   :no-inherited-members:

.. currentmodule:: xcp_d

.. autosummary::
   :toctree: generated/
   :template: module.rst

   xcp_d.interfaces.ants
   xcp_d.interfaces.bids
   xcp_d.interfaces.c3
   xcp_d.interfaces.connectivity
   xcp_d.interfaces.filtering
   xcp_d.interfaces.layout_builder
   xcp_d.interfaces.nilearn
   xcp_d.interfaces.prepostcleaning
   xcp_d.interfaces.qc_plot
   xcp_d.interfaces.regression
   xcp_d.interfaces.report_core
   xcp_d.interfaces.report
   xcp_d.interfaces.resting_state
   xcp_d.interfaces.surfplotting
   xcp_d.interfaces.workbench

:mod:`xcp_d.utils`: Miscellaneous Utility Functions
---------------------------------------------------

.. automodule:: xcp_d.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: xcp_d

.. autosummary::
   :toctree: generated/
   :template: module.rst

   xcp_d.utils.atlas
   xcp_d.utils.bids
   xcp_d.utils.concatenation
   xcp_d.utils.confounds
   xcp_d.utils.dcan2fmriprep
   xcp_d.utils.hcp2fmriprep
   xcp_d.utils.doc
   xcp_d.utils.execsummary
   xcp_d.utils.fcon
   xcp_d.utils.filemanip
   xcp_d.utils.modified_data
   xcp_d.utils.plot
   xcp_d.utils.qcmetrics
   xcp_d.utils.sentry
   xcp_d.utils.utils
   xcp_d.utils.write_save
