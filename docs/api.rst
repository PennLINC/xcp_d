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

   workflow.base
   workflow.bold
   workflow.cifti
   workflow.anatomical
   workflow.connectivity
   workflow.execsummary
   workflow.outputs
   workflow.plotting
   workflow.postprocessing
   workflow.restingstate

:mod:`xcp_d.interfaces`: Nipype Interfaces
------------------------------------------

.. automodule:: xcp_d.interfaces
   :no-members:
   :no-inherited-members:

.. currentmodule:: xcp_d

.. autosummary::
   :toctree: generated/
   :template: module.rst

   interfaces.ants
   interfaces.bids
   interfaces.c3
   interfaces.connectivity
   interfaces.filtering
   interfaces.layout_builder
   interfaces.nilearn
   interfaces.prepostcleaning
   interfaces.qc_plot
   interfaces.regression
   interfaces.report_core
   interfaces.report
   interfaces.resting_state
   interfaces.surfplotting
   interfaces.workbench

:mod:`xcp_d.utils`: Miscellaneous Utility Functions
---------------------------------------------------

.. automodule:: xcp_d.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: xcp_d

.. autosummary::
   :toctree: generated/
   :template: module.rst

   utils.atlas
   utils.bids
   utils.concatenation
   utils.confounds
   utils.dcan2fmriprep
   utils.hcp2fmriprep
   utils.doc
   utils.execsummary
   utils.fcon
   utils.filemanip
   utils.modified_data
   utils.plot
   utils.qcmetrics
   utils.sentry
   utils.utils
   utils.write_save
