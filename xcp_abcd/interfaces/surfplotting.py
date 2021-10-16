# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from ..utils import (surf2vol,get_regplot, 
           generate_brain_sprite,plot_svgx,plotimage, ribbon_to_statmap)
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from pkg_resources import resource_filename as pkgrf
from nipype.interfaces.base import traits, InputMultiObject, File
from nipype.interfaces.ants.resampling import ApplyTransforms, ApplyTransformsInputSpec
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)

LOGGER = logging.getLogger('nipype.interface')

class _plotimgInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='plot image')

class _plotimgOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='out image')

class PlotImage(SimpleInterface):
    """
    Python class to plot x,y, and z  of image data
    """
    input_spec = _plotimgInputSpec
    output_spec = _plotimgOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = fname_presuffix(
                self.inputs.in_file,
                suffix='_file.svg', newpath=runtime.cwd,
                use_ext=False)
        
        self._results['out_file'] = plotimage(self.inputs.in_file,
             self._results['out_file'])
             
        return runtime


    
class _surf2volInputSpec(BaseInterfaceInputSpec):
    template= File(exists=True,mandatory=True, desc="t1 image")
    left_surf = File(exists=True,mandatory=True, desc="left hemipshere")
    right_surf = File(exists=True,mandatory=True, desc="right hemipshere")
    scale = traits.Int(default_value=1, desc="scale factor for the surface")

class _surf2volOutputSpec(TraitedSpec):
    out_file = File(exists=True, manadatory=True,
                                  desc=" t1image")
  


class SurftoVolume(SimpleInterface):
    r"""
    this class converts the freesurfer/gifti surface to volume 
    using ras2vox transform 
    """
    input_spec = _surf2volInputSpec
    output_spec = _surf2volOutputSpec

    def _run_interface(self, runtime):

        self._results['out_file'] = fname_presuffix(
                self.inputs.template,
                suffix='mri_stats_map.nii.gz', newpath=runtime.cwd,
                use_ext=False)

        self._results['out_file']= surf2vol(
                                 template=self.inputs.template,
                                 left_surf=self.inputs.left_surf,
                                 right_surf=self.inputs.right_surf,
                                  scale=self.inputs.scale,
                                  filename=self._results['out_file'])
        return runtime


class _brainplotxInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="stats file")
    template = File(exists=True,mandatory=True, desc="mask file ")


class _brainplotxOutputSpec(TraitedSpec):
    out_html = File(exists=True, manadatory=True,
                                  desc="zscore html")

class BrainPlotx(SimpleInterface):
    r"""
    this class create brainsprite with overlay as stats image
    """
    input_spec = _brainplotxInputSpec
    output_spec = _brainplotxOutputSpec

    def _run_interface(self, runtime):
       
        self._results['out_html'] = fname_presuffix(
                'brainsprite_out_',
                suffix='file.html', newpath=runtime.cwd,
                use_ext=False,)

        self._results['out_html'] = generate_brain_sprite(
            template_image=self.inputs.template,stat_map=self.inputs.in_file,
            out_file=self._results['out_html'])

        return runtime


class _regplotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="brain file")
    overlay = File(exists=True,mandatory=True, desc="overlay ")
    n_cuts = traits.Int(default_value=3, desc="number of cuts")


class _regplotOutputSpec(TraitedSpec):
    out_file = File(exists=True, manadatory=True,
                                  desc="svg file")

class RegPlot(SimpleInterface):
    r"""
    abandoned 

    """
    input_spec = _regplotInputSpec
    output_spec = _regplotOutputSpec

    def _run_interface(self, runtime):
       
        self._results['out_file'] = fname_presuffix(
                'reg_plot_',
                suffix='file.svg', newpath=runtime.cwd,
                use_ext=False)

        self._results['out_file'] = get_regplot(
            brain=self.inputs.in_file,overlay=self.inputs.overlay,
            cuts=self.inputs.n_cuts,order=("x","y","z"),
            out_file=self._results['out_file'])

        return runtime


class _plotsvgInputSpec(BaseInterfaceInputSpec):
    rawdata = File(exists=True,mandatory=True, desc="raw data ")
    regdata = File(exists=True,mandatory=True, desc="data after regreesion data ")
    resddata = File(exists=True,mandatory=True, desc="resdiual data ")
    fd = File(exists=True,mandatory=True, desc="fd")
    mask = File(exists=False,mandatory=False, desc="mask file ")
    seg = File(exists=False,mandatory=False, desc="seg file ")
    tr = traits.Float(default_value=1, desc="TR")


class _plotsvgOutputSpec(TraitedSpec):
    before_process = File(exists=True, manadatory=True,desc="svg file before pro")
    after_process = File(exists=True, manadatory=True,desc="svg file before after")


class PlotSVGData(SimpleInterface):
    r"""
    this class plots of fd, dvars, carpet plots of bold data 
    before and after regression/filtering

    """
    input_spec = _plotsvgInputSpec
    output_spec = _plotsvgOutputSpec

    def _run_interface(self, runtime):
       
        self._results['before_process'] = fname_presuffix(
                'carpetplot_before_',
                suffix='file.svg', newpath=runtime.cwd,
                use_ext=False)

        self._results['after_process'] = fname_presuffix(
                'carpetplot_after_',
                suffix='file.svg', newpath=runtime.cwd,
                use_ext=False)

        self._results['before_process'], self._results['after_process'] = plot_svgx(
              rawdata=self.inputs.rawdata,regdata=self.inputs.regdata,resddata=self.inputs.resddata,
              tr=self.inputs.tr,mask=self.inputs.mask,fd=self.inputs.fd,seg=self.inputs.seg,
              filenameaf=self._results['after_process'],filenamebf=self._results['before_process']) 


        return runtime


class _ribbonstatmapInputSpec(BaseInterfaceInputSpec):
    ribbon = File(exists=True,mandatory=True, desc="ribbon ")
    ## other settings or files will be added later from T2 ##


class _ribbonstatmapOutputSpec(TraitedSpec):
    out_file = File(exists=True, manadatory=True,desc="ribbon > pial and white")



class RibbontoStatmap(SimpleInterface):
    r"""
    this class plots of fd, dvars, carpet plots of bold data 
    before and after regression/filtering

    """
    input_spec = _ribbonstatmapInputSpec
    output_spec = _ribbonstatmapOutputSpec

    def _run_interface(self, runtime):
       
        self._results['out_file'] = fname_presuffix(
                'pial_white_',
                suffix='.nii.gz', newpath=runtime.cwd,
                use_ext=False)

        self._results['out_file'] = ribbon_to_statmap(ribbon=self.inputs.ribbon,
                                                      outfile=self._results['out_file'])

        return runtime
