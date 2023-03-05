"""Custom wb_command interfaces."""
import os

import nibabel as nb
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    CommandLineInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)
from nipype.interfaces.workbench.base import WBCommand

from xcp_d.utils.filemanip import fname_presuffix, split_filename
from xcp_d.utils.write_save import get_cifti_intents

iflogger = logging.getLogger("nipype.interface")


class _FixCiftiIntentInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="CIFTI file to check.")


class _FixCiftiIntentOutputSpec(TraitedSpec):
    out_file = File(exists=True, mandatory=True, desc="Fixed CIFTI file.")


class FixCiftiIntent(SimpleInterface):
    """This is not technically a Connectome Workbench interface, but it is related.

    CiftiSmooth (-cifti-smooth) overwrites the output file's intent to match a dtseries extension,
    even when it is a dscalar file.
    This interface sets the appropriate intent based on the extension.

    We initially tried using a _post_run_hook in a modified version of the CiftiSmooth interface,
    but felt that the errors being raised were too opaque.

    If in_file has the correct intent code, it will be returned without modification.
    """

    input_spec = _FixCiftiIntentInputSpec
    output_spec = _FixCiftiIntentOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file

        cifti_intents = get_cifti_intents()
        _, _, out_extension = split_filename(in_file)
        target_intent = cifti_intents.get(out_extension, None)

        if target_intent is None:
            raise ValueError(f"Unknown CIFTI extension '{out_extension}'")

        img = nb.load(in_file)
        out_file = in_file
        # modify the intent if necessary, and write out the modified file
        if img.nifti_header.get_intent()[0] != target_intent:
            out_file = fname_presuffix(
                self.inputs.in_file,
                suffix="_modified",
                newpath=runtime.cwd,
                use_ext=True,
            )

            img.nifti_header.set_intent(target_intent)
            img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime


class _ConvertAffineInputSpec(CommandLineInputSpec):
    """Input specification for ConvertAffine."""

    fromwhat = traits.Str(
        mandatory=True,
        argstr="-from-%s",
        position=0,
        desc="world, itk, or flirt",
    )

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="The input file",
    )

    towhat = traits.Str(
        mandatory=True,
        argstr="-to-%s",
        position=2,
        desc="world, itk, or flirt",
    )
    out_file = File(
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
    _cmd = "wb_command -convert-affine"


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
        name_template="MNIAffine_%s.gii",
        keep_extension=True,
        extensions=[".surf.gii", ".shape.gii"],
        position=2,
    )


class _ApplyAffineOutputSpec(TraitedSpec):
    """Output specification for ApplyAffine."""

    out_file = File(exists=True, desc="output file")


class ApplyAffine(WBCommand):
    """Interface for wb_command's -surface-apply-affine command.

    .. code-block::

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
    _cmd = "wb_command -surface-apply-affine"


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
        argstr="%s",
        position=1,
        desc="The warpfield file",
    )

    out_file = File(
        argstr="%s",
        name_source="in_file",
        name_template="MNIwarped_%s.gii",
        extensions=[".surf.gii", ".shape.gii"],
        position=2,
    )

    forward_warp = File(
        argstr="-fnirt %s",
        position=3,
        desc="fnirt forward warpfield",
    )


class _ApplyWarpfieldOutputSpec(TraitedSpec):
    """Output specification for ApplyWarpfield."""

    out_file = File(exists=True, desc="output file")


class ApplyWarpfield(WBCommand):
    """Apply warpfield to surface file.

    .. code-block::

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
    _cmd = "wb_command -surface-apply-warpfield"


class _SurfaceSphereProjectUnprojectInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceSphereProjectUnproject."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="a sphere with the desired output mesh",
    )

    sphere_project_to = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="a sphere that aligns with sphere-in",
    )

    sphere_unproject_from = File(
        exists=True,
        mandatory=True,
        argstr="%s",
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

    .. code-block::

        wb_command -surface-sphere-project-unproject
            <sphere-in> - a sphere with the desired output mesh
            <sphere-project-to> - a sphere that aligns with sphere-in
            <sphere-unproject-from> - <sphere-project-to> deformed to the desired output space
            <sphere-out> - output - the output sphere
    """

    input_spec = _SurfaceSphereProjectUnprojectInputSpec
    output_spec = _SurfaceSphereProjectUnprojectOutputSpec
    _cmd = "wb_command -surface-sphere-project-unproject"


class _ChangeXfmTypeInputSpec(CommandLineInputSpec):
    in_transform = File(exists=True, argstr="%s", mandatory=True, position=0)


class _ChangeXfmTypeOutputSpec(TraitedSpec):
    out_transform = File(exists=True)


class ChangeXfmType(SimpleInterface):
    """Change transform type."""

    input_spec = _ChangeXfmTypeInputSpec
    output_spec = _ChangeXfmTypeOutputSpec

    def _run_interface(self, runtime):
        with open(self.inputs.in_transform) as f:
            lines = f.readlines()
        listcomp = [line.replace("AffineTransform", "MatrixOffsetTransformBase") for line in lines]
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
        argstr="-surf %s",
        position=1,
        desc="specify a surface to include in the average",
    )

    surface_in2 = File(
        exists=True,
        mandatory=True,
        argstr="-surf %s",
        position=2,
        desc="specify a surface to include in the average",
    )

    out_file = File(
        name_source="surface_in1",
        keep_extension=False,
        name_template="%s-avg.surf.gii",
        argstr="%s",
        position=0,
        desc="output - the output averaged surface",
    )


class _SurfaceAverageOutputSpec(TraitedSpec):
    """Output specification for SurfaceAverage."""

    out_file = File(exists=True, desc="output file")


class SurfaceAverage(WBCommand):
    """Average surface files together.

    .. code-block::

        wb_command -surface-average
            <surface-out> - output - the output averaged surface
            [-stddev] - compute 3D sample standard deviation
                <stddev-metric-out> - output - the output metric for 3D sample standard deviation
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
    _cmd = "wb_command -surface-average"


class _SurfaceGenerateInflatedInputSpec(CommandLineInputSpec):
    """Input specification for SurfaceGenerateInflated."""

    anatomical_surface_in = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="the anatomical surface",
    )

    inflated_out_file = File(
        name_source="anatomical_surface_in",
        keep_extension=False,
        name_template="%s-hcpinflated.surf.gii",
        argstr="%s",
        position=1,
        desc="output - the output inflated surface",
    )

    very_inflated_out_file = File(
        name_source="anatomical_surface_in",
        keep_extension=False,
        name_template="%s-hcpveryinflated.surf.gii",
        argstr="%s",
        position=2,
        desc="output - the output very inflated surface",
    )

    iterations_scale_value = traits.Float(
        mandatory=False,
        argstr="-iterations-scale %f",
        position=3,
        desc="iterations-scale value",
    )


class _SurfaceGenerateInflatedOutputSpec(TraitedSpec):
    """Output specification for SurfaceGenerateInflated."""

    inflated_out_file = File(exists=True, desc="inflated output file")
    very_inflated_out_file = File(exists=True, desc="very inflated output file")


class SurfaceGenerateInflated(WBCommand):
    """Generate inflated surface.

    .. code-block::

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
    _cmd = "wb_command -surface-generate-inflated"


class _CiftiParcellateInputSpec(CommandLineInputSpec):
    """Input specification for the CiftiParcellate command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The cifti file to parcellate",
    )
    atlas_label = File(
        mandatory=True,
        argstr="%s",
        position=1,
        desc="A cifti label file to use for the parcellation",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s",
        position=2,
        desc="Which mapping to parcellate (integer, ROW, or COLUMN)",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="parcellated_%s.ptseries.nii",
        keep_extension=False,
        argstr="%s",
        position=3,
        desc="Output cifti file",
    )

    # NOTE: These are not organized well.
    # -spatial-weights should appear before any in this group.
    spatial_weights = traits.Str(
        argstr="-spatial-weights",
        position=4,
        desc="Use voxel volume and either vertex areas or metric files as weights",
    )
    left_area_surf = File(
        exists=True,
        position=5,
        argstr="-left-area-surface %s",
        desc="Specify the left surface to use",
    )
    right_area_surf = File(
        exists=True,
        position=6,
        argstr="-right-area-surface %s",
        desc="Specify the right surface to use",
    )
    cerebellum_area_surf = File(
        exists=True,
        position=7,
        argstr="-cerebellum-area-surf %s",
        desc="specify the cerebellum surface to use",
    )
    left_area_metric = File(
        exists=True,
        position=8,
        argstr="-left-area-metric %s",
        desc="Specify the left surface metric to use",
    )
    right_area_metric = File(
        exists=True,
        position=9,
        argstr="-right-area-metric %s",
        desc="Specify the right surface  metric to use",
    )
    cerebellum_area_metric = File(
        exists=True,
        position=10,
        argstr="-cerebellum-area-metric %s",
        desc="specify the cerebellum surface  metricto use",
    )

    cifti_weights = File(
        exists=True,
        position=11,
        argstr="-cifti-weights %s",
        desc="Use a cifti file containing weights",
    )
    cor_method = traits.Enum(
        "MEAN",
        "MAX",
        "MIN",
        "INDEXMAX",
        "INDEXMIN",
        "SUM",
        "PRODUCT",
        "STDEV",
        "SAMPSTDEV",
        "VARIANCE",
        "TSNR",
        "COV",
        "L2NORM",
        "MEDIAN",
        "MODE",
        "COUNT_NONZERO",
        position=12,
        default="MEAN",
        argstr="-method %s",
        desc="Specify method of parcellation (default MEAN, or MODE if label data)",
    )
    only_numeric = traits.Bool(
        position=13,
        argstr="-only-numeric",
        desc="Exclude non-numeric values",
    )


class _CiftiParcellateOutputSpec(TraitedSpec):
    """Output specification for the CiftiParcellate command."""

    out_file = File(exists=True, desc="output CIFTI file")


class CiftiParcellate(WBCommand):
    """Extract timeseries from CIFTI file.

    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries.

    Examples
    --------
    >>> ciftiparcel = CiftiParcellate()
    >>> ciftiparcel.inputs.in_file = 'sub-01XX_task-rest.dtseries.nii'
    >>> ciftiparcel.inputs.out_file = 'sub_01XX_task-rest.ptseries.nii'
    >>> ciftiparcel.inputs.atlas_label = 'schaefer_space-fsLR_den-32k_desc-400_atlas.dlabel.nii'
    >>> ciftiparcel.inputs.direction = 'COLUMN'
    >>> ciftiparcel.cmdline
    wb_command -cifti-parcellate sub-01XX_task-rest.dtseries.nii \
    schaefer_space-fsLR_den-32k_desc-400_atlas.dlabel.nii COLUMN \
    sub_01XX_task-rest.ptseries.nii
    """

    input_spec = _CiftiParcellateInputSpec
    output_spec = _CiftiParcellateOutputSpec
    _cmd = "wb_command -cifti-parcellate"


class _CiftiSurfaceResampleInputSpec(CommandLineInputSpec):
    """Input specification for the CiftiSurfaceResample command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="the gifti file",
    )

    current_sphere = File(
        exists=True,
        position=1,
        argstr=" %s",
        desc=" the current sphere surface in gifti for in_file",
    )

    new_sphere = File(
        exists=True,
        position=2,
        argstr=" %s",
        desc=" the new sphere surface to be resample the in_file to, eg fsaverag5 or fsl32k",
    )

    metric = traits.Str(
        argstr=" %s ",
        position=3,
        desc=" fixed for anatomic",
        default="  BARYCENTRIC  ",
    )

    out_file = File(
        name_source=["in_file"],
        name_template="resampled_%s.gii",
        keep_extension=True,
        extensions=[".shape.gii", ".surf.gii"],
        argstr="%s",
        position=4,
        desc="The gifti output, either left and right",
    )


class _CiftiSurfaceResampleOutputSpec(TraitedSpec):
    """Output specification for the CiftiSurfaceResample command."""

    out_file = File(exists=True, desc="output gifti file")


class CiftiSurfaceResample(WBCommand):
    """Resample a surface from one sphere to another.

    TODO: Improve documentation.
    """

    input_spec = _CiftiSurfaceResampleInputSpec
    output_spec = _CiftiSurfaceResampleOutputSpec
    _cmd = "wb_command  -surface-resample"


class _CiftiSeparateMetricInputSpec(CommandLineInputSpec):
    """Input specification for the CiftiSeparateMetric command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s ",
        position=0,
        desc="The input dense series",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s ",
        position=1,
        desc="which dimension to smooth along, ROW or COLUMN",
    )
    metric = traits.Str(
        mandatory=True,
        argstr=" -metric %s ",
        position=2,
        desc="which of the structure eg CORTEX_LEFT CORTEX_RIGHT"
        "check https://www.humanconnectome.org/software/workbench-command/-cifti-separate ",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="correlation_matrix_%s.func.gii",
        keep_extension=True,
        argstr=" %s",
        position=3,
        desc="The gifti output, either left and right",
    )


class _CiftiSeparateMetricOutputSpec(TraitedSpec):
    """Output specification for the CiftiSeparateMetric command."""

    out_file = File(exists=True, desc="output CIFTI file")


class CiftiSeparateMetric(WBCommand):
    """Extract left or right hemisphere surfaces from CIFTI file (.dtseries).

    Other structures can also be extracted.
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries,

    Examples
    --------
    >>> ciftiseparate = CiftiSeparateMetric()
    >>> ciftiseparate.inputs.in_file = 'sub-01XX_task-rest.dtseries.nii'
    >>> ciftiseparate.inputs.metric = "CORTEX_LEFT" # extract left hemisphere
    >>> ciftiseparate.inputs.out_file = 'sub_01XX_task-rest_hemi-L.func.gii'
    >>> ciftiseparate.inputs.direction = 'COLUMN'
    >>> ciftiseparate.cmdline
    wb_command  -cifti-separate 'sub-01XX_task-rest.dtseries.nii'  COLUMN \
      -metric CORTEX_LEFT 'sub_01XX_task-rest_hemi-L.func.gii'
    """

    input_spec = _CiftiSeparateMetricInputSpec
    output_spec = _CiftiSeparateMetricOutputSpec
    _cmd = "wb_command  -cifti-separate "


class _CiftiSeparateVolumeAllInputSpec(CommandLineInputSpec):
    """Input specification for the CiftiSeparateVolumeAll command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="The input dense series",
    )
    direction = traits.Enum(
        "ROW",
        "COLUMN",
        mandatory=True,
        argstr="%s",
        position=1,
        desc="which dimension to smooth along, ROW or COLUMN",
    )
    out_file = File(
        name_source=["in_file"],
        name_template="%s_volumetric_data.nii.gz",
        keep_extension=False,
        argstr="-volume-all %s -crop",
        position=2,
        desc="The NIFTI output.",
    )
    label_file = File(
        name_source=["in_file"],
        name_template="%s_labels.nii.gz",
        keep_extension=False,
        argstr="-label %s",
        position=3,
        desc="A discrete segmentation NIFTI output.",
    )


class _CiftiSeparateVolumeAllOutputSpec(TraitedSpec):
    """Output specification for the CiftiSeparateVolumeAll command."""

    label_file = File(exists=True, desc="NIFTI file with labels.")
    out_file = File(exists=True, desc="NIFTI file with volumetric data.")


class CiftiSeparateVolumeAll(WBCommand):
    """Extract volumetric data from CIFTI file (.dtseries).

    Other structures can also be extracted.
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries,

    Examples
    --------
    >>> ciftiseparate = CiftiSeparateVolumeAll()
    >>> ciftiseparate.inputs.in_file = 'sub-01_task-rest.dtseries.nii'
    >>> ciftiseparate.inputs.out_file = 'sub_01_task-rest_volumetric_data.nii.gz'
    >>> ciftiseparate.inputs.label_file = 'sub_01_task-rest_labels.nii.gz'
    >>> ciftiseparate.inputs.direction = 'COLUMN'
    >>> ciftiseparate.cmdline
    wb_command  -cifti-separate 'sub-01XX_task-rest.dtseries.nii' COLUMN \
        -volume-all 'sub_01_task-rest_volumetric_data.nii.gz' \
        -label 'sub_01_task-rest_labels.nii.gz'
    """

    input_spec = _CiftiSeparateVolumeAllInputSpec
    output_spec = _CiftiSeparateVolumeAllOutputSpec
    _cmd = "wb_command  -cifti-separate "


class _CiftiCreateDenseScalarInputSpec(CommandLineInputSpec):
    """Input specification for the CiftiSeparateVolumeAll command."""

    out_file = File(
        name_source=["volume_data"],
        name_template="%s_combined.dscalar.nii",
        keep_extension=False,
        argstr="%s",
        position=0,
        desc="The CIFTI output.",
    )
    left_metric = File(
        exists=True,
        mandatory=True,
        argstr="-left-metric %s",
        position=1,
        desc="The input surface data from the left hemisphere.",
    )
    right_metric = File(
        exists=True,
        mandatory=True,
        argstr="-right-metric %s",
        position=2,
        desc="The input surface data from the right hemisphere.",
    )
    volume_data = File(
        exists=True,
        mandatory=True,
        argstr="-volume %s",
        position=3,
        desc="The input volumetric data.",
    )
    structure_label_volume = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=4,
        desc="A label file indicating the structure of each voxel in volume_data.",
    )


class _CiftiCreateDenseScalarOutputSpec(TraitedSpec):
    """Output specification for the CiftiCreateDenseScalar command."""

    out_file = File(exists=True, desc="output CIFTI file")


class CiftiCreateDenseScalar(WBCommand):
    """Extract volumetric data from CIFTI file (.dtseries).

    Other structures can also be extracted.
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries,

    Examples
    --------
    >>> cifticreatedensescalar = CiftiCreateDenseScalar()
    >>> cifticreatedensescalar.inputs.out_file = 'sub_01_task-rest.dscalar.nii'
    >>> cifticreatedensescalar.inputs.left_metric = 'sub_01_task-rest_hemi-L.func.gii'
    >>> cifticreatedensescalar.inputs.left_metric = 'sub_01_task-rest_hemi-R.func.gii'
    >>> cifticreatedensescalar.inputs.volume_data = 'sub_01_task-rest_subcortical.nii.gz'
    >>> cifticreatedensescalar.inputs.structure_label_volume = 'sub_01_task-rest_labels.nii.gz'
    >>> cifticreatedensescalar.cmdline
    wb_command -cifti-create-dense-scalar 'sub_01_task-rest.dscalar.nii' \
        -left-metric 'sub_01_task-rest_hemi-L.func.gii' \
        -right-metric 'sub_01_task-rest_hemi-R.func.gii' \
        -volume-data 'sub_01_task-rest_subcortical.nii.gz' 'sub_01_task-rest_labels.nii.gz'
    """

    input_spec = _CiftiCreateDenseScalarInputSpec
    output_spec = _CiftiCreateDenseScalarOutputSpec
    _cmd = "wb_command -cifti-create-dense-scalar"


class _ShowSceneInputSpec(CommandLineInputSpec):
    scene_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
    )
    scene_name_or_number = traits.Either(
        traits.Int,
        traits.Str,
        mandatory=True,
        position=1,
        argstr="%s",
        desc="name or number (starting at one) of the scene in the scene file",
    )
    out_file = File(
        exists=False,
        mandatory=False,
        argstr="%s",
        genfile=True,
        position=2,
        desc="output image file name",
    )
    image_width = traits.Int(
        mandatory=True,
        argstr="%s",
        position=3,
        desc="width of output image(s), in pixels",
    )
    image_height = traits.Int(
        mandatory=True,
        argstr="%s",
        position=4,
        desc="height of output image(s), in pixels",
    )


class _ShowSceneOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output image file name")


class ShowScene(WBCommand):
    """Offscreen rendering of scene to an image file.

    Notes
    -----
    wb_command -show-scene
        <scene-file> - scene file
        <scene-name-or-number> - name or number (starting at one) of the scene in
            the scene file
        <image-file-name> - output image file name
        <image-width> - width of output image(s), in pixels
        <image-height> - height of output image(s), in pixels

        [-use-window-size] - Override image size with window size

        [-no-scene-colors] - Do not use background and foreground colors in scene

        [-set-map-yoke] - Override selected map index for a map yoking group.
            <Map Yoking Roman Numeral> - Roman numeral identifying the map yoking
            group (I, II, III, IV, V, VI, VII, VIII, IX, X)
            <Map Index> - Map index for yoking group.  Indices start at 1 (one)

        [-conn-db-login] - Login for scenes with files in Connectome Database
            <Username> - Connectome DB Username
            <Password> - Connectome DB Password

        Render content of browser windows displayed in a scene into image
        file(s).  The image file name should be similar to "capture.png".  If
        there is only one image to render, the image name will not change.  If
        there is more than one image to render, an index will be inserted into
        the image name: "capture_01.png", "capture_02.png" etc.

        If the scene references files in the Connectome Database,
        the "-conn-db-login" option is available for providing the
        username and password.  If this options is not specified,
        the username and password stored in the user's preferences
        is used.

        The image format is determined by the image file extension.
        The available image formats may vary by operating system.
        Image formats available on this system are:
            bmp
            jpeg
            jpg
            png
            ppm
            tif
            tiff

        The result of using the "-use-window-size" option
        is dependent upon the version used to create the scene.
            * Versions 1.2 and newer contain the width and
            height of the graphics region.  The output image
            will be the width and height from the scene and
            the image width and height specified on the command
            line is ignored.
            * If the scene does not contain the width and height
            of the graphics region, the width and height specified
            on the command line is used for the size of the
            output image.
    """

    input_spec = _ShowSceneInputSpec
    output_spec = _ShowSceneOutputSpec
    _cmd = "wb_command -show-scene"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        frame_number = self.inputs.scene_name_or_number
        if isinstance(frame_number, int):
            # Add a bunch of leading zeros for easy sorting
            out_file = f"frame_{frame_number:06g}.png"
        else:
            out_file = f"frame_{frame_number}.png"

        return out_file


class _CiftiConvertInputSpec(CommandLineInputSpec):
    """Input specification for the CiftiConvert command."""

    target = traits.Enum(
        "from",
        "to",
        mandatory=True,
        position=0,
        argstr="-%s-nifti",
        desc="Convert either to or from nifti.",
    )
    in_file = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=1,
        desc="The input file.",
    )
    cifti_template = File(
        exists=True,
        mandatory=False,
        argstr="%s",
        position=2,
        desc="A cifti file with the dimension(s) and mapping(s) that should be used.",
    )
    TR = traits.Float(
        mandatory=False,
        desc="Repetition time in seconds. Used to reset timepoints.",
        position=4,
        argstr="-reset-timepoints %s 0",
    )
    out_file = File(
        exists=True,
        mandatory=False,
        genfile=True,
        argstr="%s",
        position=3,
        desc="The output file.",
    )


class _CiftiConvertOutputSpec(TraitedSpec):
    """Output specification for the CiftiConvert command."""

    out_file = File(
        exists=True,
        desc="The output file.",
    )


class CiftiConvert(WBCommand):
    """Convert between CIFTI and NIFTI file formats.

    Examples
    --------
    >>> cifticonvert = CiftiConvert()
    >>> cifticonvert.inputs.in_file = 'sub-01_task-rest_bold.dscalar.nii'
    >>> cifticonvert.target = "to"
    >>> cifticonvert.cmdline
    wb_command -cifti-convert -to-nifti 'sub-01_task-rest_bold.dscalar.nii' \
        'sub-01_task-rest_bold_converted.nii.gz'
    """

    input_spec = _CiftiConvertInputSpec
    output_spec = _CiftiConvertOutputSpec
    _cmd = "wb_command -cifti-convert"

    def _gen_filename(self, name):
        if name == "out_file":
            _, fname, ext = split_filename(self.inputs.in_file)
            # if we want to support other cifti outputs, we'll need to change this.
            ext = ".dtseries.nii" if self.inputs.target == "from" else ".nii.gz"
            output = fname + "_converted" + ext
            return output
        else:
            return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["out_file"] = os.path.abspath(self._gen_filename("out_file"))
        return outputs


class _CiftiCreateDenseFromTemplateInputSpec(CommandLineInputSpec):
    """Input specification for the CiftiCreateDenseFromTemplate command."""

    template_cifti = File(
        exists=True,
        mandatory=True,
        argstr="%s",
        position=0,
        desc="File to match brainordinates of.",
    )
    cifti_out = File(
        name_source=["label"],
        name_template="resampled_%s.dlabel.nii",
        keep_extension=False,
        argstr="%s",
        position=1,
        desc="The output cifti file.",
    )
    label = File(
        exists=True,
        mandatory=True,
        argstr="-cifti %s",
        position=2,
        desc="Use input data from surface label files. Input label file.",
    )


class _CiftiCreateDenseFromTemplateOutputSpec(TraitedSpec):
    """Output specification for the CiftiCreateDenseFromTemplate command."""

    cifti_out = File(exists=True, desc="output CIFTI file")


class CiftiCreateDenseFromTemplate(WBCommand):
    """Create CIFTI with matching dense map.

    This command helps you make a new dscalar, dtseries, or dlabel cifti file
    that matches the brainordinate space used in another cifti file.  The
    template file must have the desired brainordinate space in the mapping
    along the column direction (for dtseries, dscalar, dlabel, and symmetric
    dconn this is always the case).  All input cifti files must have a brain
    models mapping along column and use the same volume space and/or surface
    vertex count as the template for structures that they contain.  If any
    input files contain label data, then input files with non-label data are
    not allowed, and the -series option may not be used.

    Any structure that isn't covered by an input is filled with zeros or the
    unlabeled key.

    Examples
    --------
    >>> ccdft = CiftiCreateDenseFromTemplate()
    >>> ccdft.inputs.template_cifti = "sub-01_task-rest_bold.dtseries.nii"
    >>> ccdft.inputs.label = "parcellation.dlabel.nii"
    >>> ccdft.cmdline
    wb_command -cifti-create-dense-from-template \
        sub-01_task-rest_bold.dtseries.nii \
        resampled_parcellation.dlabel.nii \
        -label parcellation.dlabel.nii
    """

    input_spec = _CiftiCreateDenseFromTemplateInputSpec
    output_spec = _CiftiCreateDenseFromTemplateOutputSpec
    _cmd = "wb_command -cifti-create-dense-from-template"
