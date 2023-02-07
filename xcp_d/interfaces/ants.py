"""ANTS interfaces."""
import logging
import os

from nipype.interfaces.ants.base import ANTSCommand, ANTSCommandInputSpec
from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    InputMultiPath,
    Str,
    TraitedSpec,
    traits,
)
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms,
    _FixTraitApplyTransformsInputSpec,
)

from xcp_d.utils.filemanip import fname_presuffix

LOGGER = logging.getLogger("nipype.interface")


class _ConvertTransformFileInputSpec(CommandLineInputSpec):
    dimension = traits.Enum((3, 2), default=3, usedefault=True, argstr="%d", position=0)
    in_transform = traits.File(exists=True, argstr="%s", mandatory=True, position=1)
    out_transform = traits.File(
        argstr="%s",
        name_source="in_transform",
        name_template="%s.txt",
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

    _cmd = "ConvertTransformFile"
    input_spec = _ConvertTransformFileInputSpec
    output_spec = _ConvertTransformFileOutputSpec


class _CompositeInvTransformUtilInputSpec(ANTSCommandInputSpec):
    """Input specification for CompositeInvTransformUtil."""

    process = traits.Enum(
        "assemble",
        "disassemble",
        argstr="--%s",
        position=1,
        usedefault=True,
        desc="What to do with the transform inputs (assemble or disassemble)",
    )
    out_file = File(
        exists=False,
        argstr="%s",
        position=2,
        desc="Output file path (only used for disassembly).",
    )
    in_file = InputMultiPath(
        File(exists=True),
        mandatory=True,
        argstr="%s...",
        position=3,
        desc="Input transform file(s)",
    )
    output_prefix = Str(
        "transform",
        usedefault=True,
        argstr="%s",
        position=4,
        desc="A prefix that is prepended to all output files (only used for assembly).",
    )


class _CompositeInvTransformUtilOutputSpec(TraitedSpec):
    """Output specification for CompositeInvTransformUtil."""

    affine_transform = File(desc="Affine transform component")
    displacement_field = File(desc="Displacement field component")
    out_file = File(desc="Compound transformation file")


class CompositeInvTransformUtil(ANTSCommand):
    """Wrapper for the ANTS CompositeTransformUtil command.

    ANTs utility which can combine or break apart transform files into their individual
    constituent components.

    Examples
    --------
    >>> from nipype.interfaces.ants import CompositeInvTransformUtil
    >>> tran = CompositeInvTransformUtil()
    >>> tran.inputs.process = 'disassemble'
    >>> tran.inputs.in_file = 'output_Composite.h5'
    >>> tran.cmdline
    'CompositeTransformUtil --disassemble output_Composite.h5 transform'
    >>> tran.run()  # doctest: +SKIP
    example for assembling transformation files
    >>> from nipype.interfaces.ants import CompositeInvTransformUtil
    >>> tran = CompositeInvTransformUtil()
    >>> tran.inputs.process = 'assemble'
    >>> tran.inputs.out_file = 'my.h5'
    >>> tran.inputs.in_file = ['AffineTransform.mat', 'DisplacementFieldTransform.nii.gz']
    >>> tran.cmdline
    'CompositeTransformUtil --assemble my.h5 AffineTransform.mat
     DisplacementFieldTransform.nii.gz '
    >>> tran.run()  # doctest: +SKIP
    """

    _cmd = "CompositeTransformUtil"
    input_spec = _CompositeInvTransformUtilInputSpec
    output_spec = _CompositeInvTransformUtilOutputSpec

    def _num_threads_update(self):
        """Do not update the number of threads environment variable.

        CompositeInvTransformUtil ignores environment variables,
        so override environment update from ANTSCommand class.
        """
        pass

    def _format_arg(self, name, spec, value):
        if name == "output_prefix" and self.inputs.process == "assemble":
            return ""
        if name == "out_file" and self.inputs.process == "disassemble":
            return ""
        return super(CompositeInvTransformUtil, self)._format_arg(name, spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if self.inputs.process == "disassemble":
            outputs["affine_transform"] = os.path.abspath(
                f"01_{self.inputs.output_prefix}_AffineTransform.mat"
            )
            outputs["displacement_field"] = os.path.abspath(
                f"00_{self.inputs.output_prefix}_DisplacementFieldTransform.nii.gz"
            )
        if self.inputs.process == "assemble":
            outputs["out_file"] = os.path.abspath(self.inputs.out_file)
        return outputs


class _ApplyTransformsInputSpec(_FixTraitApplyTransformsInputSpec):
    # Nipype's version doesn't have GenericLabel
    interpolation = traits.Enum(
        "Linear",
        "NearestNeighbor",
        "CosineWindowedSinc",
        "WelchWindowedSinc",
        "HammingWindowedSinc",
        "LanczosWindowedSinc",
        "MultiLabel",
        "Gaussian",
        "BSpline",
        "GenericLabel",
        argstr="%s",
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
        # Run normally
        self.inputs.output_image = fname_presuffix(
            self.inputs.input_image,
            suffix="_trans.nii.gz",
            newpath=runtime.cwd,
            use_ext=False,
        )
        runtime = super(ApplyTransforms, self)._run_interface(runtime)
        return runtime
