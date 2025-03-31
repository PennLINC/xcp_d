"""Interfaces for Nilearn code."""

import os

import pandas as pd
from nilearn import masking
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.nilearn import NilearnBaseInterface

from xcp_d.utils.utils import denoise_with_nilearn
from xcp_d.utils.write_save import read_ndata, write_ndata


class _IndexImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='A 4D image to index.',
    )
    index = traits.Int(
        0,
        usedefault=True,
        desc='Volume index to select from in_file.',
    )
    out_file = File(
        'img_3d.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the indexed file.',
    )


class _IndexImageOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Concatenated output file.',
    )


class IndexImage(NilearnBaseInterface, SimpleInterface):
    """Select a specific volume from a 4D image."""

    input_spec = _IndexImageInputSpec
    output_spec = _IndexImageOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import index_img

        img_3d = index_img(self.inputs.in_file, self.inputs.index)
        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_3d.to_filename(self._results['out_file'])

        return runtime


class _MergeInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(
        File(exists=True),
        mandatory=True,
        desc='A list of images to concatenate.',
    )
    out_file = File(
        'concat_4d.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the concatenated file to write out. concat_4d.nii.gz by default.',
    )


class _MergeOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Concatenated output file.',
    )


class Merge(NilearnBaseInterface, SimpleInterface):
    """Merge images."""

    input_spec = _MergeInputSpec
    output_spec = _MergeOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import concat_imgs

        img_4d = concat_imgs(self.inputs.in_files)
        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_4d.to_filename(self._results['out_file'])

        return runtime


class _SmoothInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='An image to smooth.',
    )
    fwhm = traits.Either(
        traits.Float(),
        traits.List(
            traits.Float(),
            minlen=3,
            maxlen=3,
        ),
        desc=(
            'Full width at half maximum. '
            'Smoothing strength, as a full-width at half maximum, in millimeters.'
        ),
    )
    out_file = File(
        'smooth_img.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the smoothed file to write out. smooth_img.nii.gz by default.',
    )


class _SmoothOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Smoothed output file.',
    )


class Smooth(NilearnBaseInterface, SimpleInterface):
    """Smooth image."""

    input_spec = _SmoothInputSpec
    output_spec = _SmoothOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import smooth_img

        img_smoothed = smooth_img(self.inputs.in_file, fwhm=self.inputs.fwhm)
        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_smoothed.to_filename(self._results['out_file'])

        return runtime


class _BinaryMathInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='An image to do math on.',
    )
    expression = traits.String(
        mandatory=True,
        desc="A mathematical expression to apply to the image. Must have 'img' in it.",
    )
    out_file = File(
        'out_img.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the mathified file to write out. out_img.nii.gz by default.',
    )


class _BinaryMathOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Mathified output file.',
    )


class BinaryMath(NilearnBaseInterface, SimpleInterface):
    """Do math on an image."""

    input_spec = _BinaryMathInputSpec
    output_spec = _BinaryMathOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import math_img

        img_mathed = math_img(self.inputs.expression, img=self.inputs.in_file)
        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_mathed.to_filename(self._results['out_file'])

        return runtime


class _ApplyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='An image to do math on.',
    )
    mask = File(
        exists=True,
        mandatory=True,
        desc='A mask image.',
    )
    out_file = File(
        'out_img.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the mathified file to write out. out_img.nii.gz by default.',
    )


class _ApplyMaskOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Masked output file.',
    )


class ApplyMask(NilearnBaseInterface, SimpleInterface):
    """Apply a mask to an image."""

    input_spec = _ApplyMaskInputSpec
    output_spec = _ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nb

        img = nb.load(self.inputs.in_file)
        mask_img = nb.load(self.inputs.mask)
        mask_data = mask_img.get_fdata()
        img_data = img.get_fdata()
        if img.ndim == 3:
            img_data = img_data * mask_data
        else:
            img_data = img_data * mask_data[:, :, :, None]

        img_masked = nb.Nifti1Image(img_data, img.affine, img.header)
        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        img_masked.to_filename(self._results['out_file'])

        return runtime


class _ResampleToImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc='An image to average over time.',
    )
    target_file = File(
        exists=True,
        mandatory=True,
        desc='',
    )
    out_file = File(
        'out_img.nii.gz',
        usedefault=True,
        exists=False,
        desc='The name of the resampled file to write out. out_img.nii.gz by default.',
    )


class _ResampleToImageOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='Resampled output file.',
    )


class ResampleToImage(NilearnBaseInterface, SimpleInterface):
    """Resample a source image on a target image.

    No registration is performed: the image should already be aligned.
    """

    input_spec = _ResampleToImageInputSpec
    output_spec = _ResampleToImageOutputSpec

    def _run_interface(self, runtime):
        from nilearn.image import resample_to_img

        resampled_img = resample_to_img(
            source_img=self.inputs.in_file,
            target_img=self.inputs.target_file,
            interpolation='continuous',
        )
        self._results['out_file'] = os.path.join(runtime.cwd, self.inputs.out_file)
        resampled_img.to_filename(self._results['out_file'])


class _DenoiseImageInputSpec(BaseInterfaceInputSpec):
    """Used directly by the CIFTI interface, and modified slightly for the NIFTI one."""

    preprocessed_bold = File(
        exists=True,
        mandatory=True,
        desc=(
            'Preprocessed BOLD data, after dummy volume removal, '
            'but without any additional censoring.'
        ),
    )
    confounds_tsv = traits.Either(
        File(exists=True),
        None,
        desc='A tab-delimited file containing the confounds to remove from the BOLD data.',
    )
    confounds_images = traits.List(
        File(exists=True),
        desc='A list of 4D images containing voxelwise confounds.',
    )
    temporal_mask = File(
        exists=True,
        mandatory=True,
        desc='The tab-delimited high-motion outliers file.',
    )
    TR = traits.Float(mandatory=True, desc='Repetition time')
    bandpass_filter = traits.Bool(mandatory=True, desc='To apply bandpass or not')
    low_pass = traits.Float(mandatory=True, desc='Lowpass filter in Hz')
    high_pass = traits.Float(mandatory=True, desc='Highpass filter in Hz')
    filter_order = traits.Int(mandatory=True, desc='Filter order')
    num_threads = traits.Int(1, usedefault=True, desc='denoise on this many cpus')


class _DenoiseImageOutputSpec(TraitedSpec):
    """Used by both the CIFTI and NIFTI interfaces."""

    denoised_interpolated_bold = File(
        exists=True,
        desc=(
            'The result of denoising the censored preprocessed BOLD data, '
            'followed by cubic spline interpolation and band-pass filtering.'
        ),
    )


class DenoiseCifti(NilearnBaseInterface, SimpleInterface):
    """Denoise a CIFTI BOLD file with Nilearn.

    For more information about the exact steps,
    please see :py:func:`~xcp_d.utils.utils.denoise_with_nilearn`.
    """

    input_spec = _DenoiseImageInputSpec
    output_spec = _DenoiseImageOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.bandpass_filter:
            low_pass, high_pass = None, None
        else:
            low_pass, high_pass = self.inputs.low_pass, self.inputs.high_pass

        preprocessed_bold_arr = read_ndata(self.inputs.preprocessed_bold)

        # Transpose from SxT (xcpd order) to TxS (nilearn order)
        preprocessed_bold_arr = preprocessed_bold_arr.T
        n_volumes = preprocessed_bold_arr.shape[0]

        censoring_df = pd.read_table(self.inputs.temporal_mask)
        if censoring_df.shape[0] != n_volumes:
            raise ValueError(
                f'Temporal mask file has {censoring_df.shape[0]} rows, '
                f'but BOLD data has {n_volumes} volumes.'
            )

        # Invert temporal mask, so low-motion volumes are True and high-motion volumes are False.
        sample_mask = ~censoring_df['framewise_displacement'].to_numpy().astype(bool)

        confounds_df = None
        if self.inputs.confounds_tsv:
            confounds_df = pd.read_table(self.inputs.confounds_tsv)
            if confounds_df.shape[0] != n_volumes:
                raise ValueError(
                    f'Confounds file has {confounds_df.shape[0]} rows, '
                    f'but BOLD data has {n_volumes} volumes.'
                )

            # Drop all-NaN columns representing voxel-wise confounds
            confounds_df = confounds_df.dropna(axis=1, how='all')

        voxelwise_confounds = None
        if self.inputs.confounds_images:
            voxelwise_confounds = [read_ndata(f) for f in self.inputs.confounds_images]

        denoised_interpolated_bold = denoise_with_nilearn(
            preprocessed_bold=preprocessed_bold_arr,
            confounds=confounds_df,
            voxelwise_confounds=voxelwise_confounds,
            sample_mask=sample_mask,
            low_pass=low_pass,
            high_pass=high_pass,
            filter_order=self.inputs.filter_order,
            TR=self.inputs.TR,
            num_threads=self.inputs.num_threads,
        )

        # Transpose from TxS (nilearn order) to SxT (xcpd order)
        denoised_interpolated_bold = denoised_interpolated_bold.T
        self._results['denoised_interpolated_bold'] = os.path.join(
            runtime.cwd,
            'filtered_denoised.dtseries.nii',
        )
        write_ndata(
            denoised_interpolated_bold,
            template=self.inputs.preprocessed_bold,
            filename=self._results['denoised_interpolated_bold'],
            TR=self.inputs.TR,
        )

        return runtime


class _DenoiseNiftiInputSpec(_DenoiseImageInputSpec):
    mask = File(
        exists=True,
        mandatory=True,
        desc='A binary brain mask.',
    )


class DenoiseNifti(NilearnBaseInterface, SimpleInterface):
    """Denoise a NIfTI BOLD file with Nilearn.

    For more information about the exact steps,
    please see :py:func:`~xcp_d.utils.utils.denoise_with_nilearn`.
    """

    input_spec = _DenoiseNiftiInputSpec
    output_spec = _DenoiseImageOutputSpec

    def _run_interface(self, runtime):
        if not self.inputs.bandpass_filter:
            low_pass, high_pass = None, None
        else:
            low_pass, high_pass = self.inputs.low_pass, self.inputs.high_pass

        # Use nilearn.masking.apply_mask because it will do less to the data than NiftiMasker.
        preprocessed_bold_arr = masking.apply_mask(
            imgs=self.inputs.preprocessed_bold,
            mask_img=self.inputs.mask,
        )
        n_volumes = preprocessed_bold_arr.shape[0]

        censoring_df = pd.read_table(self.inputs.temporal_mask)
        if censoring_df.shape[0] != n_volumes:
            raise ValueError(
                f'Temporal mask file has {censoring_df.shape[0]} rows, '
                f'but BOLD data has {n_volumes} volumes.'
            )

        # Invert temporal mask, so low-motion volumes are True and high-motion volumes are False.
        sample_mask = ~censoring_df['framewise_displacement'].to_numpy().astype(bool)

        confounds_df = None
        if self.inputs.confounds_tsv:
            confounds_df = pd.read_table(self.inputs.confounds_tsv)
            if confounds_df.shape[0] != n_volumes:
                raise ValueError(
                    f'Confounds file has {confounds_df.shape[0]} rows, '
                    f'but BOLD data has {n_volumes} volumes.'
                )

            # Drop all-NaN columns representing voxel-wise confounds
            confounds_df = confounds_df.dropna(axis=1, how='all')

        voxelwise_confounds = None
        if self.inputs.confounds_images:
            voxelwise_confounds = []
            for f in self.inputs.confounds_images:
                voxelwise_confounds.append(masking.apply_mask(imgs=f, mask_img=self.inputs.mask))

        denoised_interpolated_bold = denoise_with_nilearn(
            preprocessed_bold=preprocessed_bold_arr,
            confounds=confounds_df,
            voxelwise_confounds=voxelwise_confounds,
            sample_mask=sample_mask,
            low_pass=low_pass,
            high_pass=high_pass,
            filter_order=self.inputs.filter_order,
            TR=self.inputs.TR,
            num_threads=self.inputs.num_threads,
        )

        self._results['denoised_interpolated_bold'] = os.path.join(
            runtime.cwd,
            'filtered_denoised.nii.gz',
        )
        filtered_denoised_img = masking.unmask(
            X=denoised_interpolated_bold,
            mask_img=self.inputs.mask,
        )

        # Explicitly set TR in the header
        pixdim = list(filtered_denoised_img.header.get_zooms())
        pixdim[3] = self.inputs.TR
        filtered_denoised_img.header.set_zooms(pixdim)
        filtered_denoised_img.to_filename(self._results['denoised_interpolated_bold'])

        return runtime
