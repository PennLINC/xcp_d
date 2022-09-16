# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et
"""Nipype workflows for xcp_d."""
from xcp_d.workflow.connectivity import init_fcon_ts_wf, init_cifti_conts_wf, get_transformfile
from xcp_d.workflow.postprocessing import init_post_process_wf, init_resd_smoothing
from xcp_d.workflow.restingstate import init_compute_alff_wf, init_surface_reho_wf, init_3d_reho_wf
from xcp_d.workflow.bold import init_boldpostprocess_wf
from xcp_d.workflow.cifti import init_ciftipostprocess_wf
from xcp_d.workflow.base import init_xcpd_wf
from xcp_d.workflow.anatomical import init_anatomical_wf
from xcp_d.workflow.execsummary import init_execsummary_wf

__all__ = [
    'init_fcon_ts_wf', 'init_cifti_conts_wf', 'init_post_process_wf',
    'init_compute_alff_wf', 'init_surface_reho_wf', 'init_3d_reho_wf',
    'init_boldpostprocess_wf', 'init_ciftipostprocess_wf', 'init_xcpd_wf',
    'init_censoring_wf', 'init_resd_smoothing', 'get_transformfile',
    'init_anatomical_wf', 'init_execsummary_wf'
]
