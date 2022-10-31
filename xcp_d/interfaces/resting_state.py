# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces for working with resting-state fMRI data.

.. testsetup::

"""
import os
import shutil

from nilearn.plotting import view_img
from nipype import logging
from nipype.interfaces.afni.preprocess import AFNICommandOutputSpec, DespikeInputSpec
from nipype.interfaces.afni.utils import ReHoInputSpec, ReHoOutputSpec
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
    traits_extension,
)

from xcp_d.utils.fcon import compute_2d_reho, compute_alff, mesh_adjacency
from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.utils import zscore_nifti
from xcp_d.utils.write_save import read_gii, read_ndata, write_gii, write_ndata

LOGGER = logging.getLogger('nipype.interface')


# compute 2D reho
class _SurfaceReHoInputSpec(BaseInterfaceInputSpec):
    surf_bold = File(exists=True,
                     mandatory=True,
                     desc="left or right hemisphere gii ")
    surf_hemi = traits.Str(exists=True, mandatory=True, desc="L or R ")


class _SurfaceReHoOutputSpec(TraitedSpec):
    surf_gii = File(exists=True, mandatory=True, desc=" lh hemisphere reho")


class SurfaceReHo(SimpleInterface):
    """Calculate regional homogeneity (ReHo) on a surface file.

    Examples
    --------
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    >>> surfacereho_wf = SurfaceReHo()
    >>> surfacereho_wf.inputs.surf_bold = 'rhhemi.func.gii'
    >>> surfacereho_wf.inputs.surf_hemi = 'R'
    >>> surfacereho_wf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()
    """

    input_spec = _SurfaceReHoInputSpec
    output_spec = _SurfaceReHoOutputSpec

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


class _ComputeALFFInputSpec(BaseInterfaceInputSpec):
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
    outputdir = File(exists=False,
                     mandatory=False,
                     desc="BIDS output directory")


class _ComputeALFFOutputSpec(TraitedSpec):
    alff_out = File(exists=True, mandatory=True, desc=" alff")


class ComputeALFF(SimpleInterface):
    """Compute ALFF.

    Examples
    --------
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    computealffwf = ComputeALFF()
    computealffwf.inputs.in_file = datafile
    computealffwf.inputs.lowpass = 0.1
    computealffwf.inputs.highpass = 0.01
    computealffwf.inputs.TR = TR
    computealffwf.inputs.mask_file = mask
    computealffwf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()
    """

    input_spec = _ComputeALFFInputSpec
    output_spec = _ComputeALFFOutputSpec

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


class _BrainPlotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="alff or reho")
    mask_file = File(exists=True, mandatory=True, desc="mask file ")


class _BrainPlotOutputSpec(TraitedSpec):
    nifti_html = File(exists=True, mandatory=True, desc="zscore html")


class BrainPlot(SimpleInterface):
    """Create a brainsprite figure from a NIFTI file.

    The image will first be normalized (z-scored) before the figure is generated.
    """

    input_spec = _BrainPlotInputSpec
    output_spec = _BrainPlotOutputSpec

    def _run_interface(self, runtime):

        # create a file name
        z_score_nifti = os.path.split(os.path.abspath(
            self.inputs.in_file))[0] + '/zscore.nii.gz'

        # create a nifti with z-scores
        z_score_nifti = zscore_nifti(img=self.inputs.in_file,
                                     mask=self.inputs.mask_file,
                                     outputname=z_score_nifti)

        html_view = view_img(
            stat_map_img=z_score_nifti,
            threshold=0,
            opacity=0.5,
            cut_coords=[0, 0, 0],
            title="zscore",
            bg_img=None,
        )

        # write the html out
        self._results['nifti_html'] = fname_presuffix(
            'zscore_nifti_',
            suffix='stat.html',
            newpath=runtime.cwd,
            use_ext=False,
        )

        html_view.save_as_html(self._results['nifti_html'])
        return runtime


class ReHoNamePatch(SimpleInterface):
    """Compute ReHo for a given neighbourhood, based on a local neighborhood of that voxel.

    For complete details, see the `3dReHo Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dReHo.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> reho = afni.ReHo()
    >>> reho.inputs.in_file = 'functional.nii'
    >>> reho.inputs.out_file = 'reho.nii.gz'
    >>> reho.inputs.neighborhood = 'vertices'
    >>> reho.cmdline
    '3dReHo -prefix reho.nii.gz -inset functional.nii -nneigh 27'
    >>> res = reho.run()  # doctest: +SKIP
    """

    _cmd = "3dReHo"
    input_spec = ReHoInputSpec
    output_spec = ReHoOutputSpec

    def _run_interface(self, runtime):
        out_file = os.path.join(runtime.cwd, "reho.nii.gz")

        in_file = os.path.join(runtime.cwd, "inset.nii.gz")
        shutil.copyfile(self.inputs.in_file, in_file)

        if traits_extension.isdefined(self.inputs.mask_file):
            mask_file = os.path.join(runtime.cwd, "mask.nii.gz")
            shutil.copyfile(self.inputs.mask_file, mask_file)
            mask_cmd = f"-mask {mask_file}"
        else:
            mask_cmd = ""

        os.system(f"3dReHo -inset {in_file} {mask_cmd} -nneigh 27 -prefix {out_file}")
        self._results["out_file"] = out_file


class DespikePatch(SimpleInterface):
    """Remove 'spikes' from the 3D+time input dataset.

    For complete details, see the `3dDespike Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDespike.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> despike = afni.Despike()
    >>> despike.inputs.in_file = 'functional.nii'
    >>> despike.cmdline
    '3dDespike -prefix functional_despike functional.nii'
    >>> res = despike.run()  # doctest: +SKIP
    """

    _cmd = "3dDespike"
    input_spec = DespikeInputSpec
    output_spec = AFNICommandOutputSpec

    def _run_interface(self, runtime):
        outfile = runtime.cwd + "/3despike.nii.gz"
        shutil.copyfile(self.inputs.in_file, runtime.cwd + "/inset.nii.gz")
        os.system("3dDespike -NEW -prefix  3despike.nii.gz inset.nii.gz")
        self._results['out_file'] = outfile
