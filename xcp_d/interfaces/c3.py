# -*- coding: utf-8 -*-
"""Convert3D is a command-line tool for converting 3D images between common file formats."""
import logging
import os
from glob import glob

from nipype.interfaces.base import (
    CommandLine,
    CommandLineInputSpec,
    File,
    InputMultiPath,
    OutputMultiPath,
    TraitedSpec,
    isdefined,
    traits,
)

from xcp_d.utils.filemanip import split_filename

iflogger = logging.getLogger("interface")


class _C3dInputSpec(CommandLineInputSpec):
    """Input specification for C3d."""

    in_file = InputMultiPath(
        File(exists=True),
        position=1,
        argstr="%s",
        mandatory=True,
        desc="Input file (wildcard and multiple are supported).",
    )
    out_file = File(
        exists=False,
        argstr="-o %s",
        position=-1,
        xor=["out_files"],
        desc="Output file of last image on the stack.",
    )
    out_files = InputMultiPath(
        File(exists=False),
        argstr="-oo %s",
        xor=["out_file"],
        position=-1,
        desc=(
            "Write all images on the convert3d stack as multiple files."
            " Supports both list of output files or a pattern for the output"
            " filenames (using %d substituion)."
        ),
    )
    pix_type = traits.Enum(
        "float",
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "double",
        argstr="-type %s",
        desc=(
            "Specifies the pixel type for the output image. By default,"
            " images are written in floating point (float) format"
        ),
    )
    scale = traits.Either(
        traits.Int(),
        traits.Float(),
        argstr="-scale %s",
        desc=(
            "Multiplies the intensity of each voxel in the last image on the"
            " stack by the given factor."
        ),
    )
    shift = traits.Either(
        traits.Int(),
        traits.Float(),
        argstr="-shift %s",
        desc="Adds the given constant to every voxel.",
    )
    interp = traits.Enum(
        "Linear",
        "NearestNeighbor",
        "Cubic",
        "Sinc",
        "Gaussian",
        argstr="-interpolation %s",
        desc=(
            "Specifies the interpolation used with -resample and other"
            " commands. Default is Linear."
        ),
    )
    resample = traits.Str(
        argstr="-resample %s",
        desc=(
            "Resamples the image, keeping the bounding box the same, but"
            " changing the number of voxels in the image. The dimensions can be"
            " specified as a percentage, for example to double the number of voxels"
            " in each direction. The -interpolation flag affects how sampling is"
            " performed."
        ),
    )
    smooth = traits.Str(
        argstr="-smooth %s",
        desc=(
            "Applies Gaussian smoothing to the image. The parameter vector"
            " specifies the standard deviation of the Gaussian kernel."
        ),
    )
    multicomp_split = traits.Bool(
        False,
        usedefault=True,
        argstr="-mcs",
        position=0,
        desc="Enable reading of multi-component images.",
    )
    is_4d = traits.Bool(
        False,
        usedefault=True,
        desc=("Changes command to support 4D file operations (default is false)."),
    )


class _C3dOutputSpec(TraitedSpec):
    """Output specification for C3d."""

    out_files = OutputMultiPath(File(exists=True))


class C3d(CommandLine):
    """A wrapper for the c3d command.

    Convert3d is a command-line tool for converting 3D (or 4D) images between
    common file formats. The tool also includes a growing list of commands for
    image manipulation, such as thresholding and resampling. The tool can also
    be used to obtain information about image files. More information on
    Convert3d can be found at:
    https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md

    Examples
    --------
    >>> from nipype.interfaces.c3 import C3d
    >>> c3 = C3d()
    >>> c3.inputs.in_file = "T1.nii"
    >>> c3.inputs.pix_type = "short"
    >>> c3.inputs.out_file = "T1.img"
    >>> c3.cmdline
    'c3d T1.nii -type short -o T1.img'
    >>> c3.inputs.is_4d = True
    >>> c3.inputs.in_file = "epi.nii"
    >>> c3.inputs.out_file = "epi.img"
    >>> c3.cmdline
    'c4d epi.nii -type short -o epi.img'
    """

    input_spec = _C3dInputSpec
    output_spec = _C3dOutputSpec

    _cmd = "c3d"

    def __init__(self, **inputs):
        super(C3d, self).__init__(**inputs)
        self.inputs.on_trait_change(self._is_4d, "is_4d")
        if self.inputs.is_4d:
            self._is_4d()

    def _is_4d(self):
        self._cmd = "c4d" if self.inputs.is_4d else "c3d"

    def _run_interface(self, runtime):
        cmd = self._cmd
        if not isdefined(self.inputs.out_file) and not isdefined(self.inputs.out_files):
            # Convert3d does not want to override file, by default
            # so we define a new output file
            self._gen_outfile()
        runtime = super(C3d, self)._run_interface(runtime)
        self._cmd = cmd
        return runtime

    def _gen_outfile(self):
        # if many infiles, raise exception
        if (len(self.inputs.in_file) > 1) or ("*" in self.inputs.in_file[0]):
            raise AttributeError(
                "Multiple in_files found - specify either `out_file` or `out_files`."
            )
        _, fn, ext = split_filename(self.inputs.in_file[0])
        self.inputs.out_file = fn + "_generated" + ext
        # if generated file will overwrite, raise error
        if os.path.exists(os.path.abspath(self.inputs.out_file)):
            raise IOError("File already found - to overwrite, use `out_file`.")
        iflogger.info("Generating `out_file`.")

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs["out_files"] = os.path.abspath(self.inputs.out_file)
        if isdefined(self.inputs.out_files):
            if len(self.inputs.out_files) == 1:
                _out_files = glob(os.path.abspath(self.inputs.out_files[0]))
            else:
                _out_files = [
                    os.path.abspath(f)
                    for f in self.inputs.out_files
                    if os.path.exists(os.path.abspath(f))
                ]
            outputs["out_files"] = _out_files

        return outputs
