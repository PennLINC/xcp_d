# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling computation of reho and alff.
    .. testsetup::
    # will comeback
"""
from ..utils import (write_gii, read_gii, read_ndata, write_ndata)
from ..utils import (compute_2d_reho, compute_alff,mesh_adjacency)
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, Directory, isdefined,
    SimpleInterface
)
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
LOGGER = logging.getLogger('nipype.interface') 
import os 
import nibabel as nb 
from brainsprite import viewer_substitute
import tempfile
from pkg_resources import resource_filename as pkgrf

# compute 2D reho
class _surfaceRehoInputSpec(BaseInterfaceInputSpec):
    surf_bold = File(exists=True,mandatory=True, desc="left or right hemisphere gii ")
    surf_hemi = traits.Str(exists=True,mandatory=True, desc="L or R ")


class _surfaceRehoOutputSpec(TraitedSpec):
    surf_gii = File(exists=True, manadatory=True,
                                  desc=" lh hemisphere reho")

class surfaceReho(SimpleInterface):
    r"""
    surface reho computation 
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> surfaceRehowf = surfaceReho()
    >>> surfaceRehowf.inputs.surf_bold= rhhemi.func.gii
    >>> surfaceRehowf.inputs.surf_hemi = 'R'
    >>> surfaceRehowf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """
    input_spec = _surfaceRehoInputSpec
    output_spec = _surfaceRehoOutputSpec

    def _run_interface(self, runtime):
        
        # get the gifti
        data_matrix = read_gii(self.inputs.surf_bold)

        # get mesh adjacency matrix
        mesh_matrix = mesh_adjacency(self.inputs.surf_hemi)
        # compute reho
        reho_surf = compute_2d_reho(datat=data_matrix, adjacency_matrix=mesh_matrix)
        

        #write the output out
        self._results['surf_gii'] = fname_presuffix(
                self.inputs.surf_bold,
                suffix='.gii', newpath=runtime.cwd,
                use_ext=False)
        write_gii(datat=reho_surf,template=self.inputs.surf_bold,
            filename=self._results['surf_gii'],hemi=self.inputs.surf_hemi)
        return runtime


class _alffInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="nifti, cifti or gifti")
    tr = traits.Float(exists=True,mandatory=True, desc="repetition time")
    lowpass = traits.Float(exists=True,mandatory=True, 
                            default_value=0.10,desc="lowpass filter in Hz")
    highpass = traits.Float(exists=True,mandatory=True, 
                            default_value=0.01,desc="highpass filter in Hz")
    mask = File(exists=False, mandatory=False,
                          desc=" brain mask for nifti file")


class _alffOutputSpec(TraitedSpec):
    alff_out = File(exists=True, manadatory=True,
                                  desc=" alff")

class computealff(SimpleInterface):
    r"""
    ALFF computation 
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> computealffwf = computealff()
    >>> computealffwf.inputs.in_file = datafile
    >>> computealffwf.inputs.lowpass = 0.1
    >>> computealffwf.inputs.highpass = 0.01 
    >>> computealffwf.inputs.TR = TR
    >>> computealffwf.inputs.mask_file = mask
    >>> computealffwf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """
    input_spec = _alffInputSpec
    output_spec = _alffOutputSpec

    def _run_interface(self, runtime):
        
        # get the nifti/cifti into  matrix
        data_matrix = read_ndata(datafile=self.inputs.in_file, 
                    maskfile=self.inputs.mask)
        
      
        alff_mat = compute_alff(data_matrix=data_matrix,
                     low_pass=self.inputs.lowpass,
                     high_pass=self.inputs.highpass, 
                     TR=self.inputs.tr)

        # writeout the data
        if self.inputs.in_file.endswith('.dtseries.nii'):
            suffix='_alff.dtseries.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):
            suffix='_alff.nii.gz'

        #write the output out
        self._results['alff_out'] = fname_presuffix(
                self.inputs.in_file,
                suffix=suffix, newpath=runtime.cwd,
                use_ext=False,)
        write_ndata(data_matrix=alff_mat, template=self.inputs.in_file, 
                filename=self._results['alff_out'],mask=self.inputs.mask)

        return runtime





class _brainplotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="alff or reho")
    mask_file = File(exists=True,mandatory=True, desc="mask file ")


class _brainplotOutputSpec(TraitedSpec):
    nifti_html = File(exists=True, manadatory=True,
                                  desc="zscore html")

class brainplot(SimpleInterface):
    r"""
    coming

    """
    input_spec = _brainplotInputSpec
    output_spec = _brainplotOutputSpec

    def _run_interface(self, runtime):
        
        # convert nifti to zscore
        tempnifti = tempfile.mkdtemp() + '/zscore.nii.gz'
        
        tempnifti = zscore_nifti(img=self.inputs.in_file,mask=self.inputs.mask_file,
                    outputname=tempnifti)
                    
        temptlatehtml = pkgrf('xcp_abcd','data/transform/brainsprite_template.html')

        bsprite = viewer_substitute(threshold=0, opacity=0.5, title="zcore",
                         cut_coords=[0,0,0])

        bsprite.fit(tempnifti, bg_img=None)

        import tempita
        template = tempita.Template.from_filename(temptlatehtml, encoding="utf-8")
        viewer = bsprite.transform(template, javascript='js', html='html', library='bsprite')
        self._results['nifti_html'] = fname_presuffix(
                'zscore_nifti_',
                suffix='stat.html', newpath=runtime.cwd,
                use_ext=False,)

        viewer.save_as_html(self._results['nifti_html'])
        return runtime


def zscore_nifti(img,outputname,mask=None):
    """
    image and mask must be in the same space

    """

    img = nb.load(img)
    
    if mask:
        maskdata = nb.load(mask).get_fdata()
        imgdata  = img.get_fdata()
        meandata = imgdata[maskdata>0].mean()
        stddata  = imgdata[maskdata>0].std()
    else:
        imgdata  = img.get_fdata()
        meandata = imgdata[imgdata>0].mean()
        stddata  = imgdata[imgdata>0].std()
     
    zscore_fdata = (imgdata - meandata)/stddata 

    dataout = nb.Nifti1Image(zscore_fdata,affine=img.affine,header=img.header)
    dataout.to_filename(outputname)
    return outputname

