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
    )


class _ConvertTransformFileOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ConvertTransformFile(CommandLine):
    _cmd = "ConvertTransformFile"
    input_spec = _ConvertTransformFileInputSpec
    output_spec = _ConvertTransformFileOutputSpec


class CompositeInvTransformUtilInputSpec(ANTSCommandInputSpec):
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


class CompositeInvTransformUtilOutputSpec(TraitedSpec):
    affine_transform = File(desc="Affine transform component")
    displacement_field = File(desc="Displacement field component")
    out_file = File(desc="Compound transformation file")


class CompositeInvTransformUtil(ANTSCommand):
    """
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
    input_spec = CompositeInvTransformUtilInputSpec
    output_spec = CompositeInvTransformUtilOutputSpec

    def _num_threads_update(self):
        """
        CompositeInvTransformUtil ignores environment variables,
        so override environment update from ANTSCommand class
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
