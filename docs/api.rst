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

.. currentmodule:: xcp_d.workflow

.. autosummary::
   :toctree: generated/
   :template: module.rst

   base
   bold
   cifti
   anatomical
   connectivity
   execsummary
   outputs
   plotting
   postprocessing
   restingstate

:mod:`xcp_d.interfaces`: Nipype Interfaces
------------------------------------------

.. automodule:: xcp_d.interfaces
   :no-members:
   :no-inherited-members:

.. currentmodule:: xcp_d.interfaces

.. autosummary::
   :toctree: generated/
   :template: module.rst

   ants
   bids
   c3
   connectivity
   filtering
   layout_builder
   nilearn
   prepostcleaning
   qc_plot
   regression
   report_core
   report
   resting_state
   surfplotting
   workbench

:mod:`xcp_d.utils`: Miscellaneous Utility Functions
---------------------------------------------------

.. automodule:: xcp_d.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: xcp_d.utils

.. autosummary::
   :toctree: generated/
   :template: module.rst

   atlas
   bids
   concatenation
   confounds
   dcan2fmriprep
   hcp2fmriprep
   doc
   execsummary
   fcon
   filemanip
   modified_data
   plot
   qcmetrics
   sentry
   utils
   write_save
