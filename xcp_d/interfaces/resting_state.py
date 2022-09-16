# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling computation of reho and alff.
    .. testsetup::
    # will comeback
"""
import os

import nibabel as nb
import numpy as np
import tempita
from brainsprite import viewer_substitute
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from pkg_resources import resource_filename as pkgrf

from xcp_d.utils import (
    compute_2d_reho,
    compute_alff,
    mesh_adjacency,
    read_gii,
    read_ndata,
    write_gii,
    write_ndata,
)
from xcp_d.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')


# compute 2D reho
class _surfaceRehoInputSpec(BaseInterfaceInputSpec):
    surf_bold = File(exists=True,
                     mandatory=True,
                     desc="left or right hemisphere gii ")
    surf_hemi = traits.Str(exists=True, mandatory=True, desc="L or R ")


class _surfaceRehoOutputSpec(TraitedSpec):
    surf_gii = File(exists=True, manadatory=True, desc=" lh hemisphere reho")


class surfaceReho(SimpleInterface):
    r"""
    surface reho computation
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    surfaceRehowf = surfaceReho()
    surfaceRehowf.inputs.surf_bold= rhhemi.func.gii
    surfaceRehowf.inputs.surf_hemi = 'R'
    surfaceRehowf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """
    input_spec = _surfaceRehoInputSpec
    output_spec = _surfaceRehoOutputSpec

    def _run_interface(self, runtime):

        # Read the gifti data
        data_matrix = read_gii(self.inputs.surf_bold)

        # Get the mesh adjacency matrix
        mesh_matrix = mesh_adjacency(self.inputs.surf_hemi)

        # Compute reho
        reho_surf = compute_2d_reho(datat=data_matrix,
                                    adjacency_matrix=mesh_matrix)

        # Write the output out
        self._results['surf_gii'] = fname_presuffix(self.inputs.surf_bold,
                                                    suffix='.shape.gii',
                                                    newpath=runtime.cwd,
                                                    use_ext=False)
        write_gii(datat=reho_surf,
                  template=self.inputs.surf_bold,
                  filename=self._results['surf_gii'],
                  hemi=self.inputs.surf_hemi)
        return runtime


class _alffInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="nifti, cifti or gifti")
    TR = traits.Float(exists=True, mandatory=True, desc="repetition time")
    lowpass = traits.Float(exists=True,
                           mandatory=True,
                           default_value=0.10,
                           desc="lowpass filter in Hz")
    highpass = traits.Float(exists=True,
                            mandatory=True,
                            default_value=0.01,
                            desc="highpass filter in Hz")
    mask = File(exists=False,
                mandatory=False,
                desc=" brain mask for nifti file")


class _alffOutputSpec(TraitedSpec):
    alff_out = File(exists=True, manadatory=True, desc=" alff")


class computealff(SimpleInterface):
    r"""
    ALFF computation
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    computealffwf = computealff()
    computealffwf.inputs.in_file = datafile
    computealffwf.inputs.lowpass = 0.1
    computealffwf.inputs.highpass = 0.01
    computealffwf.inputs.TR = TR
    computealffwf.inputs.mask_file = mask
    computealffwf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """
    input_spec = _alffInputSpec
    output_spec = _alffOutputSpec

    def _run_interface(self, runtime):

        # Get the nifti/cifti into matrix form
        data_matrix = read_ndata(datafile=self.inputs.in_file,
                                 maskfile=self.inputs.mask)
        # compute the ALFF
        alff_mat = compute_alff(data_matrix=data_matrix,
                                low_pass=self.inputs.lowpass,
                                high_pass=self.inputs.highpass,
                                TR=self.inputs.TR)

        # Write out the data

        if self.inputs.in_file.endswith('.dtseries.nii'):
            suffix = '_alff.dtseries.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):
            suffix = '_alff.nii.gz'

        self._results['alff_out'] = fname_presuffix(
            self.inputs.in_file,
            suffix=suffix,
            newpath=runtime.cwd,
            use_ext=False,
        )
        write_ndata(data_matrix=alff_mat,
                    template=self.inputs.in_file,
                    filename=self._results['alff_out'],
                    mask=self.inputs.mask)

        return runtime


class _brainplotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="alff or reho")
    mask_file = File(exists=True, mandatory=True, desc="mask file ")


class _brainplotOutputSpec(TraitedSpec):
    nifti_html = File(exists=True, manadatory=True, desc="zscore html")


class brainplot(SimpleInterface):
    r"""
    Convert to a z-score map

    """
    input_spec = _brainplotInputSpec
    output_spec = _brainplotOutputSpec

    def _run_interface(self, runtime):

        # create a file name
        z_score_nifti = os.path.split(os.path.abspath(
            self.inputs.in_file))[0] + '/zscore.nii.gz'

        # create a nifti with z-scores
        z_score_nifti = zscore_nifti(img=self.inputs.in_file,
                                     mask=self.inputs.mask_file,
                                     outputname=z_score_nifti)
        #  get the right template
        temptlatehtml = pkgrf('xcp_d',
                              'data/transform/brainsprite_template.html')
        # adjust settings for viewing in HTML
        bsprite = viewer_substitute(threshold=0,
                                    opacity=0.5,
                                    title="zcore",
                                    cut_coords=[0, 0, 0])

        bsprite.fit(z_score_nifti, bg_img=None)

        template = tempita.Template.from_filename(temptlatehtml,
                                                  encoding="utf-8")
        viewer = bsprite.transform(template,
                                   javascript='js',
                                   html='html',
                                   library='bsprite')
        # write the html out
        self._results['nifti_html'] = fname_presuffix(
            'zscore_nifti_',
            suffix='stat.html',
            newpath=runtime.cwd,
            use_ext=False,
        )

        viewer.save_as_html(self._results['nifti_html'])
        return runtime


def zscore_nifti(img, outputname, mask=None):
    """
    Turn image into z_score.
    Image and mask must be in the same space.

    """

    img = nb.load(img)

    if mask:
        # z-score the data
        maskdata = nb.load(mask).get_fdata()
        imgdata = img.get_fdata()
        meandata = imgdata[maskdata > 0].mean()
        stddata = imgdata[maskdata > 0].std()
        zscore_fdata = (imgdata - meandata) / stddata
        # values where the mask is less than 1 are set to 0
        zscore_fdata[maskdata < 1] = 0
    else:
        # z-score the data
        imgdata = img.get_fdata()
        meandata = imgdata[np.abs(imgdata) > 0].mean()
        stddata = imgdata[np.abs(imgdata) > 0].std()
        zscore_fdata = (imgdata - meandata) / stddata

    # turn image to nifti and write it out
    dataout = nb.Nifti1Image(zscore_fdata,
                             affine=img.affine,
                             header=img.header)
    dataout.to_filename(outputname)
    return outputname
