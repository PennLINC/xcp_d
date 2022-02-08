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
    traits, TraitedSpec, BaseInterfaceInputSpec, File,SimpleInterface
)
LOGGER = logging.getLogger('nipype.interface')
from ..utils import extract_timeseries_funct
import matplotlib.pyplot as plt
from nilearn.plotting import plot_matrix
import nibabel as nb
import numpy as np

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
    select atlas by name from xcp_d/data
    all atlases are in MNI dimension
    atlas list:
      schaefer100x17
      schaefer200x17
      schaefer300x17
      schaefer400x17
      schaefer500x17
      schaefer600x17
      schaefer700x17
      schaefer800x17
      schaefer900x17
      schaefer1000x17
      glasser360
      gordon360
    """

    if atlasname[:8] == 'schaefer':
        if atlasname[8:12] == '1000': atlasfile = pkgrf('xcp_d', 'data/niftiatlas/Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_2mm.nii'.format(atlasname[8:11]))
        else: atlasfile = pkgrf('xcp_d', 'data/niftiatlas/Schaefer2018_{0}Parcels_17Networks_order_FSLMNI152_2mm.nii'.format(atlasname[8:11]))
    elif atlasname == 'glasser360':
        atlasfile = pkgrf('xcp_d', 'data/niftiatlas/glasser360/glasser360MNI.nii.gz')
    elif atlasname == 'gordon333':
        atlasfile = pkgrf('xcp_d', 'data/niftiatlas/gordon333/gordon333MNI.nii.gz')
    elif atlasname == 'tiansubcortical':
        atlasfile = pkgrf('xcp_d', 'data//niftiatlas/TianSubcortical/Tian_Subcortex_S3_3T.nii.gz')
    else:
        raise RuntimeError('atlas not available')
    return atlasfile


def get_atlas_cifti(atlasname):
    r"""
    select atlas by name from xcp_d/data
    all atlases are in 91K dimension
    atlas list:
      schaefer100x17
      schaefer200x17
      schaefer300x17
      schaefer400x17
      schaefer500x17
      schaefer600x17
      schaefer700x17
      schaefer800x17
      schaefer900x17
      schaefer1000x17
      glasser360
      gordon360
    """
    if atlasname[:8] == 'schaefer':
        if atlasname[8:12] == '1000': atlasfile = pkgrf('xcp_d', 'data/ciftiatlas/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii'.format(atlasname[8:11]))
        else: atlasfile = pkgrf('xcp_d', 'data/ciftiatlas/Schaefer2018_{0}Parcels_17Networks_order.dlabel.nii'.format(atlasname[8:11]))
    elif atlasname == 'glasser360':
        atlasfile = pkgrf('xcp_d', 'data/ciftiatlas/glasser_space-fsLR_den-32k_desc-atlas.dlabel.nii')
    elif atlasname == 'gordon333':
        atlasfile = pkgrf('xcp_d', 'data/ciftiatlas/gordon_space-fsLR_den-32k_desc-atlas.dlabel.nii')
    elif atlasname == 'tiansubcortical':
        atlasfile = pkgrf('xcp_d', 'data/ciftiatlas/Tian_Subcortex_S3_3T_32k.dlabel.nii')
    else:
        raise RuntimeError('atlas not available')
    return atlasfile

class _connectplotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,mandatory=True, desc="bold file")
    sc217_timeseries = File(exists=True,mandatory=True, desc="sc217 atlas")
    sc417_timeseries = File(exists=True,mandatory=True, desc="sc417 atlas")
    gd333_timeseries = File(exists=True,mandatory=True, desc="gordon atlas")
    gs360_timeseries = File(exists=True,mandatory=True, desc="glasser atlas")
    

class _connectplotOutputSpec(TraitedSpec):
    connectplot = File(exists=True, manadatory=True,)


class connectplot(SimpleInterface):
    r"""
    extract timeseries and compute connectvtioy matrices.
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> conect = connectplot()
    >>> conect.inputs.in_file = bold_file
    >>> conf.inputs.sc217_timeseries = sc217_timeseries
    >>> conf.inputs.sc417_timeseries = sc417_timeseries
    >>> conf.inputs.gd333_timeseries = gd333_timeseries
    >>> conf.inputs.gs360_timeseries = gs360_timeseries
    >>> conf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()

    """
    input_spec = _connectplotInputSpec
    output_spec = _connectplotOutputSpec

    def _run_interface(self, runtime):

        if self.inputs.in_file.endswith('dtseries.nii'):
            sc217 = np.corrcoef(nb.load(self.inputs.sc217_timeseries).get_fdata().T)
            sc417 = np.corrcoef(nb.load(self.inputs.sc417_timeseries).get_fdata().T)
            gd333 = np.corrcoef(nb.load(self.inputs.gd333_timeseries).get_fdata().T)
            gs360 = np.corrcoef(nb.load(self.inputs.gs360_timeseries).get_fdata().T)
           
        else:
            sc217 = np.corrcoef(np.loadtxt(self.inputs.sc217_timeseries,delimiter=',').T)
            sc417 = np.corrcoef(np.loadtxt(self.inputs.sc417_timeseries,delimiter=',').T)
            gd333 = np.corrcoef(np.loadtxt(self.inputs.gd333_timeseries,delimiter=',').T)
            gs360 = np.corrcoef(np.loadtxt(self.inputs.gs360_timeseries,delimiter=',').T)
    

        fig, ax1 = plt.subplots(2,2)
        fig.set_size_inches(20, 20)
        font = {'weight': 'normal','size': 20}
        plot_matrix(mat=sc217, colorbar=False,vmax=1, vmin=-1, axes=ax1[0,0])
        ax1[0,0].set_title('schaefer 200  17 networks', fontdict=font)
        plot_matrix(mat=sc417, colorbar=False,vmax=1, vmin=-1, axes=ax1[0,1])
        ax1[0,1].set_title('schaefer 400  17 networks', fontdict=font)
        plot_matrix(mat=gd333, colorbar=False,vmax=1, vmin=-1, axes=ax1[1,0])
        ax1[1,0].set_title('Gordon 333', fontdict=font)
        plot_matrix(mat=gs360, colorbar=False,vmax=1, vmin=-1, axes=ax1[1,1])
        ax1[1,1].set_title('Glasser 360', fontdict=font)

        self._results['connectplot'] = fname_presuffix('connectivityplot', suffix='_matrixplot.svg',
                                                   newpath=runtime.cwd, use_ext=False)

        fig.savefig( self._results['connectplot'],
                          bbox_inches="tight", pad_inches=None)

        return runtime
