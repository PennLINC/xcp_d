from xcp_d.interfaces.filtering import FilteringData
from xcp_d.interfaces.regression import regress, ciftidespike
from xcp_d.interfaces.connectivity import (NiftiConnect, ApplyTransformsx, get_atlas_cifti,
                                           get_atlas_nifti, connectplot)
from xcp_d.interfaces.resting_state import ComputeAlff, SurfaceReho, BrainPlot

from xcp_d.interfaces.prepostcleaning import interpolate, CensorScrub, RemoveTR
from xcp_d.interfaces.qc_plot import computeqcplot
from xcp_d.interfaces.report import SubjectSummary, AboutSummary, FunctionalSummary
from xcp_d.interfaces.report_core import generate_reports
from xcp_d.interfaces.surfplotting import (SurftoVolume, BrainPlotx, PlotSVGData, RegPlot,
                                           PlotImage, RibbontoStatmap)
from xcp_d.interfaces.layout_builder import LayoutBuilder

__all__ = [
    'regress','FilteringData', 'NiftiConnect',
    'ComputeAlff', 'SurfaceReho', 'get_atlas_cifti', 'get_atlas_nifti',
    'ApplyTransformsx', 'interpolate', 'CensorScrub', 'RemoveTR',
    'computeqcplot', 'SubjectSummary', 'AboutSummary', 'FunctionalSummary',
    'generate_reports', 'ciftidespike', 'connectplot', 'BrainPlot',
    'SurftoVolume', 'BrainPlotx', 'PlotSVGData', 'RegPlot', 'PlotImage',
    'LayoutBuilder', 'RibbontoStatmap'
]
