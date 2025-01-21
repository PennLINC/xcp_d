"""ANTS interfaces."""

import logging
import os

from nipype.interfaces.ants.base import ANTSCommand, ANTSCommandInputSpec
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    InputMultiObject,
    InputMultiPath,
    Str,
    TraitedSpec,
    isdefined,
    traits,
)
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms,
    _FixTraitApplyTransformsInputSpec,
)

from xcp_d.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger('nipype.interface')


class _ConvertTransformFileInputSpec(CommandLineInputSpec):
    dimension = traits.Enum(3, 2, usedefault=True, argstr='%d', position=0)
    in_transform = traits.File(exists=True, argstr='%s', mandatory=True, position=1)
    out_transform = traits.File(
        argstr='%s',
        name_source='in_transform',
        name_template='%s.txt',
        keep_extension=False,
        position=2,
        exists=False,
    )


class _ConvertTransformFileOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ConvertTransformFile(CommandLine):
    """Wrapper for the ANTS ConvertTransformFile command.

    Utility to read in a transform file (presumed to be in binary format) and output it in various
    formats.
    Default output is legacy human-readable text format.
    Without any options, the output filename extension must be .txt or .tfm to signify a
    text-formatted transform file.
    """

    _cmd = 'ConvertTransformFile'
    input_spec = _ConvertTransformFileInputSpec
    output_spec = _ConvertTransformFileOutputSpec


class _CompositeTransformUtilInputSpec(ANTSCommandInputSpec):
    """Input specification for CompositeTransformUtil."""

    process = traits.Enum(
        'assemble',
        'disassemble',
        argstr='--%s',
        position=1,
        usedefault=True,
        desc='What to do with the transform inputs (assemble or disassemble)',
    )
    inverse = traits.Bool(
        False,
        usedefault=True,
        desc='Whether to invert the order of the transform components. Not used by the command.',
    )
    out_file = File(
        exists=False,
        argstr='%s',
        position=2,
        desc='Output file path (only used for disassembly).',
    )
    in_file = InputMultiPath(
        File(exists=True),
        mandatory=True,
        argstr='%s...',
        position=3,
        desc='Input transform file(s)',
    )
    output_prefix = Str(
        'transform',
        usedefault=True,
        argstr='%s',
        position=4,
        desc='A prefix that is prepended to all output files (only used for assembly).',
    )


class _CompositeTransformUtilOutputSpec(TraitedSpec):
    """Output specification for CompositeTransformUtil."""

    affine_transform = File(desc='Affine transform component')
    displacement_field = File(desc='Displacement field component')
    out_file = File(desc='Compound transformation file')


class CompositeTransformUtil(ANTSCommand):
    """Wrapper for the ANTS CompositeTransformUtil command.

    ANTs utility which can combine or break apart transform files into their individual
    constituent components.

    Examples
    --------
    >>> from nipype.interfaces.ants import CompositeTransformUtil
    >>> tran = CompositeTransformUtil()
    >>> tran.inputs.process = 'disassemble'
    >>> tran.inputs.in_file = 'output_Composite.h5'
    >>> tran.cmdline
    'CompositeTransformUtil --disassemble output_Composite.h5 transform'
    >>> tran.run()  # doctest: +SKIP
    example for assembling transformation files
    >>> from nipype.interfaces.ants import CompositeTransformUtil
    >>> tran = CompositeTransformUtil(inverse=False)
    >>> tran.inputs.process = 'assemble'
    >>> tran.inputs.out_file = 'my.h5'
    >>> tran.inputs.in_file = ['AffineTransform.mat', 'DisplacementFieldTransform.nii.gz']
    >>> tran.cmdline
    'CompositeTransformUtil --assemble my.h5 AffineTransform.mat
     DisplacementFieldTransform.nii.gz '
    >>> tran.run()  # doctest: +SKIP
    """

    _cmd = 'CompositeTransformUtil'
    input_spec = _CompositeTransformUtilInputSpec
    output_spec = _CompositeTransformUtilOutputSpec

    def _num_threads_update(self):
        """Do not update the number of threads environment variable.

        CompositeTransformUtil ignores environment variables,
        so override environment update from ANTSCommand class.
        """
        pass

    def _format_arg(self, name, spec, value):
        if name == 'output_prefix' and self.inputs.process == 'assemble':
            return ''
        if name == 'out_file' and self.inputs.process == 'disassemble':
            return ''
        return super()._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.inverse:
            if self.inputs.process == 'disassemble':
                outputs['affine_transform'] = os.path.abspath(
                    f'{self.inputs.output_prefix}_01_AffineTransform.mat'
                )
                outputs['displacement_field'] = os.path.abspath(
                    f'{self.inputs.output_prefix}_00_DisplacementFieldTransform.nii.gz'
                )
            elif self.inputs.process == 'assemble':
                outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        else:
            if self.inputs.process == 'disassemble':
                outputs['affine_transform'] = os.path.abspath(
                    f'{self.inputs.output_prefix}_00_AffineTransform.mat'
                )
                outputs['displacement_field'] = os.path.abspath(
                    f'{self.inputs.output_prefix}_01_DisplacementFieldTransform.nii.gz'
                )
            elif self.inputs.process == 'assemble':
                outputs['out_file'] = os.path.abspath(self.inputs.out_file)

        return outputs


class _ApplyTransformsInputSpec(_FixTraitApplyTransformsInputSpec):
    # Nipype's version doesn't have GenericLabel
    interpolation = traits.Enum(
        'Linear',
        'NearestNeighbor',
        'CosineWindowedSinc',
        'WelchWindowedSinc',
        'HammingWindowedSinc',
        'LanczosWindowedSinc',
        'MultiLabel',
        'Gaussian',
        'BSpline',
        'GenericLabel',
        argstr='%s',
        usedefault=True,
    )


class ApplyTransforms(FixHeaderApplyTransforms):
    """A modified version of FixHeaderApplyTransforms from niworkflows.

    The niworkflows version of ApplyTransforms "fixes the resampled image header
    to match the xform of the reference image".
    This modification overrides the allowed interpolation values,
    since FixHeaderApplyTransforms doesn't support GenericLabel,
    which is preferred over MultiLabel.
    """

    input_spec = _ApplyTransformsInputSpec

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.output_image):
            self.inputs.output_image = fname_presuffix(
                self.inputs.input_image,
                suffix='_trans.nii.gz',
                newpath=runtime.cwd,
                use_ext=False,
            )

        runtime = super()._run_interface(runtime)
        return runtime


class _TransformsToDisplacementInputSpec(ANTSCommandInputSpec):
    dimension = traits.Enum(
        2,
        3,
        4,
        argstr='--dimensionality %d',
        desc=(
            'This option forces the image to be treated '
            'as a specified-dimensional image. If not '
            'specified, antsWarp tries to infer the '
            'dimensionality from the input image.'
        ),
    )
    output_image = traits.Str(
        argstr='--output [%s,1]', desc='output file name', genfile=True, hash_files=False
    )
    reference_image = File(
        argstr='--reference-image %s',
        mandatory=True,
        desc='reference image space that you wish to warp INTO',
        exists=True,
    )
    transforms = InputMultiObject(
        traits.Either(File(exists=True), 'identity'),
        argstr='%s',
        mandatory=True,
        desc='transform files: will be applied in reverse order. For '
        'example, the last specified transform will be applied first.',
    )
    float = traits.Bool(
        argstr='--float %d',
        default_value=False,
        usedefault=True,
        desc='Use float instead of double for computations.',
    )


class _TransformsToDisplacementOutputSpec(TraitedSpec):
    output_image = File(exists=True, desc='Warped image')


class TransformsToDisplacement(ANTSCommand):
    """ApplyTransforms, applied to an input image, transforms it according to a
    reference image and a transform (or a set of transforms).
    """

    _cmd = 'antsApplyTransforms'
    input_spec = _TransformsToDisplacementInputSpec
    output_spec = _TransformsToDisplacementOutputSpec

    def _gen_filename(self, name):
        if name == 'output_image':
            output = self.inputs.output_image
            if not isdefined(output):
                output = 'transforms.nii.gz'
            return output
        return None

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = os.path.abspath(self._gen_filename('output_image'))
        return outputs
