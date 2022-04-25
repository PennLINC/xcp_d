import logging
from nipype.interfaces.base import (TraitedSpec, CommandLineInputSpec, BaseInterfaceInputSpec,
                                    CommandLine, File, traits, isdefined, InputMultiObject,
                                    OutputMultiObject, SimpleInterface)
from nipype.interfaces import ants
LOGGER = logging.getLogger('nipype.interface')


class _ConvertTransformFileInputSpec(CommandLineInputSpec):
    dimension = traits.Enum(
        (3, 2),
        default= 3,
        usedefault=True,
        argstr="%d",
        position=0)
    in_transform = traits.File(
        exists=True,
        argstr="%s",
        mandatory=True,
        position=1)
    out_transform = traits.File(
        argstr="%s",
        name_source='in_transform',
        name_template='%s.txt',
        keep_extension=False,
        position=2)


class _ConvertTransformFileOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ConvertTransformFile(CommandLine):
    _cmd = "ConvertTransformFile"
    input_spec = _ConvertTransformFileInputSpec
    output_spec = _ConvertTransformFileOutputSpec

