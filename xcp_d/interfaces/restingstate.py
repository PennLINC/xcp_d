# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces for working with resting-state fMRI data.

.. testsetup::

"""

import os
import shutil

import pandas as pd
from nipype import logging
from nipype.interfaces.afni.preprocess import Despike, DespikeInputSpec
from nipype.interfaces.afni.utils import ReHoInputSpec, ReHoOutputSpec
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    Undefined,
    traits,
    traits_extension,
)

from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.restingstate import compute_2d_reho, mesh_adjacency
from xcp_d.utils.write_save import read_gii, read_ndata, write_gii, write_ndata

LOGGER = logging.getLogger('nipype.interface')


# compute 2D reho
class _SurfaceReHoInputSpec(BaseInterfaceInputSpec):
    surf_bold = File(exists=True, mandatory=True, desc='left or right hemisphere gii ')
    # TODO: Change to Enum
    surf_hemi = traits.Str(mandatory=True, desc='L or R ')


class _SurfaceReHoOutputSpec(TraitedSpec):
    surf_gii = File(exists=True, mandatory=True, desc=' lh hemisphere reho')


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
        reho_surf = compute_2d_reho(datat=data_matrix, adjacency_matrix=mesh_matrix)

        # Write the output out
        self._results['surf_gii'] = fname_presuffix(
            self.inputs.surf_bold, suffix='.shape.gii', newpath=runtime.cwd, use_ext=False
        )
        write_gii(
            datat=reho_surf,
            template=self.inputs.surf_bold,
            filename=self._results['surf_gii'],
            hemi=self.inputs.surf_hemi,
        )

        return runtime


class _ComputeALFFInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='nifti, cifti or gifti')
    TR = traits.Float(mandatory=True, desc='repetition time')
    low_pass = traits.Float(
        mandatory=True,
        desc='low_pass filter in Hz',
    )
    high_pass = traits.Float(
        mandatory=True,
        desc='high_pass filter in Hz',
    )
    mask = File(
        exists=True,
        mandatory=False,
        desc=' brain mask for nifti file',
    )
    temporal_mask = traits.Either(
        File(exists=True),
        Undefined,
        mandatory=False,
        desc='Temporal mask.',
    )
    n_threads = traits.Int(
        1,
        usedefault=True,
        desc='number of threads to use',
        nohash=True,
    )


class _ComputeALFFOutputSpec(TraitedSpec):
    alff = File(exists=True, mandatory=True, desc=' alff')


class ComputeALFF(SimpleInterface):
    """Compute amplitude of low-frequency fluctuation (ALFF).

    Notes
    -----
    The ALFF implementation is based on :footcite:t:`yu2007altered`,
    although the ALFF values are not scaled by the mean ALFF value across the brain.

    If censoring is applied (i.e., ``fd_thresh > 0``), then the power spectrum will be estimated
    using a Lomb-Scargle periodogram
    :footcite:p:`lomb1976least,scargle1982studies,townsend2010fast,taylorlomb`.

    References
    ----------
    .. footbibliography::
    """

    input_spec = _ComputeALFFInputSpec
    output_spec = _ComputeALFFOutputSpec

    def _run_interface(self, runtime):
        import gc
        from multiprocessing import Pool

        import numpy as np

        from xcp_d.utils.restingstate import compute_alff_chunk

        # Get the nifti/cifti into matrix form
        data_matrix = read_ndata(datafile=self.inputs.in_file, maskfile=self.inputs.mask)
        n_voxels, n_volumes = data_matrix.shape

        sample_mask = None
        temporal_mask = self.inputs.temporal_mask
        if isinstance(temporal_mask, str) and os.path.isfile(temporal_mask):
            censoring_df = pd.read_table(temporal_mask)
            # Invert the temporal mask to make retained volumes 1s and dropped volumes 0s.
            sample_mask = ~censoring_df['framewise_displacement'].values.astype(bool)
            if sample_mask.sum() != n_volumes:
                # Data are not censored
                assert sample_mask.size == n_volumes, f'{sample_mask.size} != {n_volumes}'
                # Censor the data
                data_matrix = data_matrix[:, sample_mask]

            assert sample_mask.sum() == n_volumes, f'{sample_mask.sum()} != {n_volumes}'

        # Split the data_matrix into n_threads chunks of voxels
        voxel_indices = np.array_split(np.arange(n_voxels), self.inputs.n_threads)
        split_arrays = np.array_split(data_matrix, self.inputs.n_threads, axis=0)

        del data_matrix
        gc.collect()

        alff_mat = np.zeros(n_voxels)
        with Pool(processes=self.inputs.n_threads) as pool:
            args = [
                (
                    split_arrays[i_thread],
                    self.inputs.low_pass,
                    self.inputs.high_pass,
                    self.inputs.TR,
                    sample_mask,
                )
                for i_thread in range(self.inputs.n_threads)
            ]
            results = pool.map(compute_alff_chunk, args)

        for i_thread, result in enumerate(results):
            alff_mat[voxel_indices[i_thread]] = result

        # Add extra dimension to the matrix
        alff_mat = alff_mat[:, None]

        # Write out the data
        if self.inputs.in_file.endswith('.dtseries.nii'):
            suffix = '_alff.dscalar.nii'
        elif self.inputs.in_file.endswith('.nii.gz'):
            suffix = '_alff.nii.gz'

        self._results['alff'] = fname_presuffix(
            self.inputs.in_file,
            suffix=suffix,
            newpath=runtime.cwd,
            use_ext=False,
        )
        write_ndata(
            data_matrix=alff_mat,
            template=self.inputs.in_file,
            filename=self._results['alff'],
            mask=self.inputs.mask,
        )
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

    _cmd = '3dReHo'
    input_spec = ReHoInputSpec
    output_spec = ReHoOutputSpec

    def _run_interface(self, runtime):
        out_file = os.path.join(runtime.cwd, 'reho.nii.gz')

        in_file = os.path.join(runtime.cwd, 'inset.nii.gz')
        shutil.copyfile(self.inputs.in_file, in_file)

        if traits_extension.isdefined(self.inputs.mask_file):
            mask_file = os.path.join(runtime.cwd, 'mask.nii.gz')
            shutil.copyfile(self.inputs.mask_file, mask_file)
            mask_cmd = f'-mask {mask_file}'
        else:
            mask_cmd = ''

        os.system(f'3dReHo -inset {in_file} {mask_cmd} -nneigh 27 -prefix {out_file}')  # noqa: S605
        self._results['out_file'] = out_file


class _DespikePatchInputSpec(DespikeInputSpec):
    out_file = File(
        mandatory=False,
        genfile=True,
        desc='output image file name',
        argstr='-prefix %s',
    )


class DespikePatch(Despike):
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

    input_spec = _DespikePatchInputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            return 'inset.nii.gz'
        else:
            return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_filename('out_file'))
        return outputs
