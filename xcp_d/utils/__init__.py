# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from xcp_d.utils.write_save import (read_ndata, write_ndata, read_gii, write_gii,
                                    despikedatacifti)
from xcp_d.utils.plot import (plot_svg, compute_dvars, plotimage)
from xcp_d.utils.confounds import load_confound_matrix
from xcp_d.utils.fcon import (extract_timeseries_funct, compute_2d_reho, compute_alff,
                              mesh_adjacency)
from xcp_d.utils.cifticonnectivity import CiftiCorrelation
from xcp_d.utils.ciftiparcellation import CiftiParcellate
from xcp_d.utils.ciftiseparatemetric import CiftiSeparateMetric
from xcp_d.utils.ciftiresample import CiftiSurfaceResample
from xcp_d.utils.bids import (collect_participants, collect_data, select_registrationfile,
                              select_cifti_bold, extract_t1w_seg)
from xcp_d.utils.bids import DerivativesDataSink as bid_derivative
from xcp_d.utils.modified_data import (interpolate_masked_data,
                                       generate_mask, compute_FD)
from xcp_d.utils.sentry import sentry_setup

from xcp_d.utils.qcmetrics import regisQ

from xcp_d.utils.utils import (get_maskfiles, get_transformfile, get_transformfilex,
                               stringforparams, fwhm2sigma, get_customfile)

from xcp_d.utils.plot import (plot_svgx, plot_carpet, confoundplot)

from xcp_d.utils.execsummary import (surf2vol, get_regplot, plot_registrationx,
                                     generate_brain_sprite, ribbon_to_statmap)
from xcp_d.utils.dcan2fmriprep import dcan2fmriprep
from xcp_d.utils.hcp2fmriprep import hcp2fmriprep
from xcp_d.utils.restingstate import ReHoNamePatch, DespikePatch, ContrastEnhancement
from xcp_d.utils.concantenation import concatenatebold

__all__ = [
    'read_ndata',
    'write_ndata',
    'read_gii',
    'write_gii',
    'plot_svg',
    'compute_dvars',
    'load_confound_matrix',
    'CiftiCorrelation',
    'CiftiParcellate',
    'CiftiSeparateMetric',
    'collect_participants',
    'collect_data',
    'compute_2d_reho',
    'extract_timeseries_funct',
    'compute_alff',
    'mesh_adjacency',
    'interpolate_masked_data',
    'generate_mask',
    'compute_FD',
    'bid_derivative',
    'sentry_setup',
    'despikedatacifti',
    'regisQ',
    'get_maskfiles',
    'get_transformfile',
    'get_transformfilex',
    'stringforparams',
    'fwhm2sigma',
    'get_customfile',
    'select_registrationfile',
    'select_cifti_bold',
    'CiftiSurfaceResample',
    'plot_svgx',
    'plot_carpet',
    'confoundplot',
    'surf2vol',
    'get_regplot',
    'plot_registrationx',
    'generate_brain_sprite',
    'plotimage',
    'extract_t1w_seg',
    'ribbon_to_statmap',
    'dcan2fmriprep',
    'hcp2fmriprep',
    'ReHoNamePatch',
    'DespikePatch',
    'concatenatebold',
    'ContrastEnhancement',
]
