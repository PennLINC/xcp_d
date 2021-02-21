# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Handling functional connectvity.
    .. testsetup::
    # will comeback
"""
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
from ..utils import extract_timeseries_funct

# nifti functional connectivity

class _nifticonnectInputSpec(BaseInterfaceInputSpec):
    regressed_file = File(exists=True,mandatory=True, desc="regressed file")
    atlas = File(exists=True,mandatory=True, desc="atlas file")

class _nifticonnectOutputSpec(TraitedSpec):
    time_series_tsv = File(exists=True, manadatory=True,
                                  desc=" time series file")
    fcon_matrix_tsv = File(exists=True, manadatory=True,
                                  desc=" time series file")


class nifticonnect(SimpleInterface):
    r"""
    extract timeseries and compute connectvtioy matrices.
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> conect = nifticonnect()
    >>> conect.inputs.regressed_file = datafile
    >>> conf.inputs.atlas = atlas_file
    >>> conf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()
    
    """
    input_spec = _nifticonnectInputSpec
    output_spec = _nifticonnectOutputSpec
    
    def _run_interface(self, runtime):
     
        self._results['time_series_tsv'] = fname_presuffix(
                self.inputs.regressed_file,
                suffix='time_series.tsv', newpath=runtime.cwd,
                use_ext=False)
        self._results['fcon_matrix_tsv'] = fname_presuffix(
                self.inputs.regressed_file,
                suffix='fcon_matrix.tsv', newpath=runtime.cwd,
                use_ext=False)
    
        self._results['time_series_tsv'],self._results['fcon_matrix_tsv'] = extract_timeseries_funct( 
                                 in_file=self.inputs.regressed_file,
                                 atlas=self.inputs.atlas,
                                 timeseries=self._results['time_series_tsv'],
                                 fconmatrix=self._results['fcon_matrix_tsv'])
        return runtime

class _ApplyTransformsInputSpec(ApplyTransformsInputSpec):
    transforms = InputMultiObject(
        traits.Either(File(exists=True), 'identity'),
        argstr="%s",
        mandatory=True,
        desc="transform files",
    )
class ApplyTransformsx(ApplyTransforms):
    """
    ApplyTransforms  dfrom nipype as workflow
    """

    input_spec = _ApplyTransformsInputSpec

    def _run_interface(self, runtime):
        # Run normally
        self.inputs.output_image = fname_presuffix(
                self.inputs.input_image,
                suffix='_trans.nii.gz', newpath=runtime.cwd,
                use_ext=False)
        runtime = super(ApplyTransformsx, self)._run_interface(
            runtime)
        return runtime

def get_atlas_nifti(atlasname):
    r"""
    select atlas by name from xcp_abcd/data
    all atlases are in MNI dimension
    atlas list: 
      schaefer200x7
      schaefer400x7
      glasser360
      gordon360
    """
    if atlasname == 'schaefer200x7':
        atlasfile = pkgrf('xcp_abcd', 'data/niftiatlas/schaefer200x7/schaefer200x7MNI.nii.gz')
    elif atlasname == 'schaefer400x7':
        atlasfile = pkgrf('xcp_abcd', 'data/niftiatlas/schaefer400x17/schaefer400x17MNI.nii.gz')
    elif atlasname == 'glasser360':
        atlasfile = pkgrf('xcp_abcd', 'data/niftiatlas/glasser360/glasser360MNI.nii.gz')
    elif atlasname == 'gordon333':
        atlasfile = pkgrf('xcp_abcd', 'data/niftiatlas/gordon333/gordon333MNI.nii.gz')
    else:
        raise RuntimeError('atlas not available')
    return atlasfile


def get_atlas_cifti(atlasname):
    r"""
    select atlas by name from xcp_abcd/data
    all atlases are in 91K dimension
    atlas list: 
      schaefer200x7
      schaefer400x7
      glasser360
      gordon360
    """
    if atlasname == 'schaefer200x7':
        atlasfile = pkgrf('xcp_abcd', 'data/ciftiatlas/schaefer_space-fsLR_den-32k_desc-200Parcels7Networks_atlas.dlabel.nii')
    elif atlasname == 'schaefer400x7':
        atlasfile = pkgrf('xcp_abcd', 'data/ciftiatlas/schaefer_space-fsLR_den-32k_desc-400Parcels7Networks_atlas.dlabel.nii')
    elif atlasname == 'glasser360':
        atlasfile = pkgrf('xcp_abcd', 'data/ciftiatlas/glasser_space-fsLR_den-32k_desc-atlas.dlabel.nii')
    elif atlasname == 'gordon333':
        atlasfile = pkgrf('xcp_abcd', 'data/ciftiatlas/gordon_space-fsLR_den-32k_desc-atlas.dlabel.nii')
    else:
        raise RuntimeError('atlas not available')
    return atlasfile