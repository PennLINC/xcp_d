"""Custom wb_command interfaces."""

from nipype import logging
from nipype.interfaces.base import (
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.workbench.base import WBCommand

from xcp_d.utils.filemanip import fname_presuffix

iflogger = logging.getLogger("nipype.interface")


class _ConvertWarpfieldInputSpec(CommandLineInputSpec):
    """Input specification for ConvertWarpfield."""

    fromwhat = traits.Str(
        mandatory=True,
        argstr="-from-%s ",
        position=0,
        desc="world, itk, or fnirt",
    )

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="The input file",
    )

    towhat = traits.Str(
        mandatory=True,
        argstr="-to-%s ",
        position=2,
        desc="world, itk, or fnirt",
    )

    out_file = traits.File(
        argstr="%s",
        name_source="in_file",
        name_template="%s_converted.nii.gz",
        keep_extension=False,
        position=3,
    )

    source_volume = File(
        argstr="%s ",
        position=4,
        desc="fnirt source volume",
    )


class _ConvertWarpfieldOutputSpec(TraitedSpec):
    """Output specification for ConvertWarpfield."""

    out_file = File(exists=True, desc="output file")


class ConvertWarpfield(WBCommand):
    """Interface for wb_command's -convert-warpfield command."""

    input_spec = _ConvertWarpfieldInputSpec
    output_spec = _ConvertWarpfieldOutputSpec
    _cmd = "wb_command -convert-warpfield "


class _ConvertAffineInputSpec(CommandLineInputSpec):
    """Input specification for ConvertAffine."""

    fromwhat = traits.Str(
        mandatory=True,
        argstr="-from-%s ",
        position=0,
        desc="world, itk, or flirt",
    )

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="The input file",
    )

    towhat = traits.Str(
        mandatory=True,
        argstr="-to-%s ",
        position=2,
        desc="world, itk, or flirt",
    )
    out_file = traits.File(
        argstr="%s",
        name_source="in_file",
        name_template="%s_world.nii.gz",
        keep_extension=False,
        position=3,
    )


class _ConvertAffineOutputSpec(TraitedSpec):
    """Output specification for ConvertAffine."""

    out_file = File(exists=True, desc="output file")


class ConvertAffine(WBCommand):
    """Interface for wb_command's -convert-affine command."""

    input_spec = _ConvertAffineInputSpec
    output_spec = _ConvertAffineOutputSpec
    _cmd = "wb_command -convert-affine "


class _ApplyAffineInputSpec(CommandLineInputSpec):
    """Input specification for ApplyAffine."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input file",
    )

    affine = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="The affine file",
    )

    out_file = File(
        argstr="%s",
        name_source="in_file",
        name_template="%s-MNIaffine.surf.gii",
        keep_extension=False,
        position=2,
    )


class _ApplyAffineOutputSpec(TraitedSpec):
    """Output specification for ApplyAffine."""

    out_file = File(exists=True, desc="output file")


class ApplyAffine(WBCommand):
    """Interface for wb_command's -surface-apply-affine command.

    wb_command -surface-apply-affine
       <in-surf> - the surface to transform
       <affine> - the affine file
       <out-surf> - output - the output transformed surface

       [-flirt] - MUST be used if affine is a flirt affine
          <source-volume> - the source volume used when generating the affine
          <target-volume> - the target volume used when generating the affine

    For flirt matrices, you must use the -flirt option, because flirt
    matrices are not a complete description of the coordinate transform they
    represent.  If the -flirt option is not present, the affine must be a
    nifti 'world' affine, which can be obtained with the -convert-affine
    command, or aff_conv from the 4dfp suite.
    """

    input_spec = _ApplyAffineInputSpec
    output_spec = _ApplyAffineOutputSpec
    _cmd = "wb_command -surface-apply-affine "


class _ApplyWarpfieldInputSpec(CommandLineInputSpec):
    """Input specification for ApplyWarpfield."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input file",
    )

    warpfield = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="The warpfield file",
    )

    out_file = File(
        argstr="%s",
        name_source="in_file",
        name_template="%s-MNIwarped.surf.gii",
        keep_extension=False,
        position=2,
    )

    forward_warp = File(
        argstr="-fnirt %s ",
        position=3,
        desc="fnirt forward warpfield",
    )


class _ApplyWarpfieldOutputSpec(TraitedSpec):
    """Output specification for ApplyWarpfield."""

    out_file = File(exists=True, desc="output file")


class ApplyWarpfield(WBCommand):
    """Apply warpfield to surface file.

    wb_command -surface-apply-warpfield
        <in-surf> - the surface to transform
        <warpfield> - the INVERSE warpfield
        <out-surf> - output - the output transformed surface

        [-fnirt] - MUST be used if using a fnirt warpfield
            <forward-warp> - the forward warpfield

    Warping a surface requires the INVERSE of the warpfield used to warp the volume it lines up
    with.
    The header of the forward warp is needed by the -fnirt option in order to correctly
    interpret the displacements in the fnirt warpfield.

    If the -fnirt option is not present, the warpfield must be a nifti 'world' warpfield,
    which can be obtained with the -convert-warpfield command.
    """

    input_spec = _ApplyWarpfieldInputSpec
    output_spec = _ApplyWarpfieldOutputSpec
    _cmd = "wb_command -surface-apply-warpfield "


class _SurfaceSphereProjectUnprojectInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceSphereProjectUnproject."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="a sphere with the desired output mesh",
    )

    sphere_project_to = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="a sphere that aligns with sphere-in",
    )

    sphere_unproject_from = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=2,
        desc="deformed to the desired output space",
    )

    out_file = File(
        name_source="in_file",
        name_template="%s_deformed.surf.gii",
        keep_extension=False,
        argstr="%s",
        position=3,
        desc="The sphere output file",
    )


class _SurfaceSphereProjectUnprojectOutputSpec(TraitedSpec):
    """Input specification for SurfaceSphereProjectUnproject."""

    out_file = File(exists=True, desc="output file")


class SurfaceSphereProjectUnproject(WBCommand):
    """Copy registration deformations to different sphere.

    wb_command -surface-sphere-project-unproject
    <sphere-in> - a sphere with the desired output mesh
    <sphere-project-to> - a sphere that aligns with sphere-in
    <sphere-unproject-from> - <sphere-project-to> deformed to the desired output space
    <sphere-out> - output - the output sphere
    """

    input_spec = _SurfaceSphereProjectUnprojectInputSpec
    output_spec = _SurfaceSphereProjectUnprojectOutputSpec
    _cmd = "wb_command -surface-sphere-project-unproject "


class _ChangeXfmTypeInputSpec(CommandLineInputSpec):
    in_transform = traits.File(exists=True, argstr="%s", mandatory=True, position=0)


class _ChangeXfmTypeOutputSpec(TraitedSpec):
    out_transform = traits.File(exists=True)


class ChangeXfmType(SimpleInterface):
    """Change transform type."""

    input_spec = _ChangeXfmTypeInputSpec
    output_spec = _ChangeXfmTypeOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.in_transform) as f:
            lines = f.readlines()
        listcomp = [
            line.replace("AffineTransform", "MatrixOffsetTransformBase")
            for line in lines
        ]
        outfile = fname_presuffix(
            self.inputs.in_transform,
            suffix="_MatrixOffsetTransformBase",
            newpath=runtime.cwd,
        )
        with open(outfile, "w") as write_file:
            write_file.write("".join(listcomp))
        self._results["out_transform"] = outfile
        return runtime


class _SurfaceAverageInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceAverage."""

    surface_in1 = File(
        exists=True,
        mandatory=True,
        argstr="-surf %s ",
        position=1,
        desc="specify a surface to include in the average",
    )

    surface_in2 = File(
        exists=True,
        mandatory=True,
        argstr="-surf %s ",
        position=2,
        desc="specify a surface to include in the average",
    )

    out_file = File(
        name_source="surface_in1",
        keep_extension=False,
        name_template="%s-avg.surf.gii",
        argstr="%s ",
        position=0,
        desc="output - the output averaged surface",
    )


class _SurfaceAverageOutputSpec(TraitedSpec):
    """Output specification for SurfaceAverage."""

    out_file = File(exists=True, desc="output file")


class SurfaceAverage(WBCommand):
    """Average surface files together.

    wb_command -surface-average
    <surface-out> - output - the output averaged surface
    [-stddev] - compute 3D sample standard deviation
       <stddev-metric-out> - output - the output metric for 3D sample
          standard deviation
    [-uncertainty] - compute caret5 'uncertainty'
       <uncert-metric-out> - output - the output metric for uncertainty
    [-surf] - repeatable - specify a surface to include in the average
       <surface> - a surface file to average
       [-weight] - specify a weighted average
          <weight> - the weight to use (default 1)

    The 3D sample standard deviation is computed as
    'sqrt(sum(squaredlength(xyz - mean(xyz)))/(n - 1))'.

    Uncertainty is a legacy measure used in caret5, and is computed as
    'sum(length(xyz - mean(xyz)))/n'.

    When weights are used, the 3D sample standard deviation treats them as
    reliability weights.
    """

    input_spec = _SurfaceAverageInputSpec
    output_spec = _SurfaceAverageOutputSpec
    _cmd = "wb_command -surface-average "


class _SurfaceGenerateInflatedInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceGenerateInflated."""

    anatomical_surface_in = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="the anatomical surface",
    )

    inflated_out_file = File(
        name_source="anatomical_surface_in",
        keep_extension=False,
        name_template="%s-hcpinflated.surf.gii",
        argstr="%s ",
        position=1,
        desc="output - the output inflated surface",
    )

    very_inflated_out_file = File(
        name_source="anatomical_surface_in",
        keep_extension=False,
        name_template="%s-hcpveryinflated.surf.gii",
        argstr="%s ",
        position=2,
        desc="output - the output very inflated surface",
    )

    iterations_scale_value = traits.Float(
        mandatory=False,
        argstr="-iterations-scale %f ",
        position=3,
        desc="iterations-scale value",
    )


class _SurfaceGenerateInflatedOutputSpec(TraitedSpec):
    """Output specification for SurfaceGenerateInflated."""

    inflated_out_file = File(exists=True, desc="inflated output file")
    very_inflated_out_file = File(exists=True, desc="very inflated output file")


class SurfaceGenerateInflated(WBCommand):
    """Generate inflated surface.

    wb_command -surface-generate-inflated
       <anatomical-surface-in> - the anatomical surface
       <inflated-surface-out> - output - the output inflated surface
       <very-inflated-surface-out> - output - the output very inflated surface

       [-iterations-scale] - optional iterations scaling
          <iterations-scale-value> - iterations-scale value

       Generate inflated and very inflated surfaces. The output surfaces are
       'matched' (have same XYZ range) to the anatomical surface. In most cases,
       an iterations-scale of 1.0 (default) is sufficient.  However, if the
       surface contains a large number of vertices (150,000), try an
       iterations-scale of 2.5.
    """

    input_spec = _SurfaceGenerateInflatedInputSpec
    output_spec = _SurfaceGenerateInflatedOutputSpec
    _cmd = "wb_command -surface-generate-inflated "
