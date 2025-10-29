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
    isdefined,
    traits,
)
from nipype.interfaces.workbench.base import WBCommand as WBCommandBase

from xcp_d.utils.filemanip import fname_presuffix, split_filename
from xcp_d.utils.write_save import get_cifti_intents

iflogger = logging.getLogger('nipype.interface')


class _WBCommandInputSpec(CommandLineInputSpec):
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='set number of threads')


class WBCommand(WBCommandBase):
    """A base interface for wb_command.

    This inherits from Nipype's WBCommand interface, but adds a num_threads input.
    """

    @property
    def num_threads(self):
        """Get number of threads."""
        return self.inputs.num_threads

    @num_threads.setter
    def num_threads(self, value):
        self.inputs.num_threads = value

    def __init__(self, **inputs):
        super().__init__(**inputs)

        if hasattr(self.inputs, 'num_threads'):
            self.inputs.on_trait_change(self._nthreads_update, 'num_threads')

    def _nthreads_update(self):
        """Update environment with new number of threads."""
        self.inputs.environ['OMP_NUM_THREADS'] = str(self.inputs.num_threads)


class _FixCiftiIntentInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='CIFTI file to check.')


class _FixCiftiIntentOutputSpec(TraitedSpec):
    out_file = File(exists=True, mandatory=True, desc='Fixed CIFTI file.')


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
                suffix='_modified',
                newpath=runtime.cwd,
                use_ext=True,
            )

            img.nifti_header.set_intent(target_intent)
            img.to_filename(out_file)

        self._results['out_file'] = out_file
        return runtime


class _SurfaceSphereProjectUnprojectInputSpec(_WBCommandInputSpec):
    """Input specification for SurfaceSphereProjectUnproject."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='a sphere with the desired output mesh',
    )
    sphere_project_to = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='a sphere that aligns with sphere-in',
    )
    sphere_unproject_from = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=2,
        desc='deformed to the desired output space',
    )
    out_file = File(
        name_source='in_file',
        name_template='%s_deformed.surf.gii',
        keep_extension=False,
        argstr='%s',
        position=3,
        desc='The sphere output file',
    )


class _SurfaceSphereProjectUnprojectOutputSpec(TraitedSpec):
    """Input specification for SurfaceSphereProjectUnproject."""

    out_file = File(exists=True, desc='output file')


class SurfaceSphereProjectUnproject(WBCommand):
    """Copy registration deformations to different sphere.

    A surface registration starts with an input sphere,
    and moves its vertices around on the sphere until it matches the template data.
    This means that the registration deformation is actually represented as the difference
    between two separate files - the starting sphere, and the registered sphere.
    Since the starting sphere of the registration may not have vertex correspondence to any
    other sphere (often, it is a native sphere),
    it can be inconvenient to manipulate or compare these deformations across subjects, etc.

    The purpose of this command is to be able to apply these deformations onto a new sphere
    of the user's choice,
    to make it easier to compare or manipulate them.
    Common uses are to concatenate two successive separate registrations
    (e.g. Human to Chimpanzee, and then Chimpanzee to Macaque)
    or inversion (for dedrifting or symmetric registration schemes).

    <sphere-in> must already be considered to be in alignment with one of the two ends of the
    registration (if your registration is Human to Chimpanzee,
    <sphere-in> must be in register with either Human or Chimpanzee).

    The 'project-to' sphere must be the side of the registration that is aligned with <sphere-in>
    (if your registration is Human to Chimpanzee, and <sphere-in> is aligned with Human, then
    'project-to' should be the original Human sphere).

    The 'unproject-from' sphere must be the remaining sphere of the registration
    (original vs deformed/registered).
    The output is as if you had run the same registration with <sphere-in> as the starting sphere,
    in the direction of deforming the 'project-to' sphere to create the 'unproject-from' sphere.

    Note that this command cannot check for you what spheres are aligned with other spheres,
    and using the wrong spheres or in the incorrect order will not necessarily cause an error
    message.
    In some cases, it may be useful to use a new, arbitrary sphere as the input,
    which can be created with the -surface-create-sphere command.

    .. code-block::

        wb_command -surface-sphere-project-unproject
            <sphere-in> - a sphere with the desired output mesh
            <sphere-project-to> - a sphere that aligns with sphere-in
            <sphere-unproject-from> - <sphere-project-to> deformed to the desired output space
            <sphere-out> - output - the output sphere
    """

    input_spec = _SurfaceSphereProjectUnprojectInputSpec
    output_spec = _SurfaceSphereProjectUnprojectOutputSpec
    _cmd = 'wb_command -surface-sphere-project-unproject'


class _SurfaceAverageInputSpec(_WBCommandInputSpec):
    """Input specification for SurfaceAverage."""

    surface_in1 = File(
        exists=True,
        mandatory=True,
        argstr='-surf %s',
        position=1,
        desc='specify a surface to include in the average',
    )
    surface_in2 = File(
        exists=True,
        mandatory=True,
        argstr='-surf %s',
        position=2,
        desc='specify a surface to include in the average',
    )
    out_file = File(
        name_source='surface_in1',
        keep_extension=False,
        name_template='%s-avg.surf.gii',
        argstr='%s',
        position=0,
        desc='output - the output averaged surface',
    )


class _SurfaceAverageOutputSpec(TraitedSpec):
    """Output specification for SurfaceAverage."""

    out_file = File(exists=True, desc='output file')


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
    _cmd = 'wb_command -surface-average'


class _SurfaceGenerateInflatedInputSpec(_WBCommandInputSpec):
    """Input specification for SurfaceGenerateInflated."""

    anatomical_surface_in = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='the anatomical surface',
    )
    inflated_out_file = File(
        name_source='anatomical_surface_in',
        keep_extension=False,
        name_template='%s-hcpinflated.surf.gii',
        argstr='%s',
        position=1,
        desc='output - the output inflated surface',
    )
    very_inflated_out_file = File(
        name_source='anatomical_surface_in',
        keep_extension=False,
        name_template='%s-hcpveryinflated.surf.gii',
        argstr='%s',
        position=2,
        desc='output - the output very inflated surface',
    )
    iterations_scale_value = traits.Float(
        mandatory=False,
        argstr='-iterations-scale %f',
        position=3,
        desc='iterations-scale value',
    )


class _SurfaceGenerateInflatedOutputSpec(TraitedSpec):
    """Output specification for SurfaceGenerateInflated."""

    inflated_out_file = File(exists=True, desc='inflated output file')
    very_inflated_out_file = File(exists=True, desc='very inflated output file')


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
    _cmd = 'wb_command -surface-generate-inflated'


class _CiftiParcellateWorkbenchInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiParcellateWorkbench command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='The cifti file to parcellate',
    )
    atlas_label = File(
        mandatory=True,
        argstr='%s',
        position=1,
        desc='A cifti label file to use for the parcellation',
    )
    direction = traits.Enum(
        'ROW',
        'COLUMN',
        mandatory=True,
        argstr='%s',
        position=2,
        desc='Which mapping to parcellate (integer, ROW, or COLUMN)',
    )
    out_file = File(
        name_source=['in_file'],
        name_template='parcellated_%s.ptseries.nii',
        keep_extension=False,
        argstr='%s',
        position=3,
        desc='Output cifti file',
    )

    # NOTE: These are not organized well.
    # -spatial-weights should appear before any in this group.
    spatial_weights = traits.Str(
        argstr='-spatial-weights',
        position=4,
        desc='Use voxel volume and either vertex areas or metric files as weights',
    )
    left_area_surf = File(
        exists=True,
        position=5,
        argstr='-left-area-surface %s',
        desc='Specify the left surface to use',
    )
    right_area_surf = File(
        exists=True,
        position=6,
        argstr='-right-area-surface %s',
        desc='Specify the right surface to use',
    )
    cerebellum_area_surf = File(
        exists=True,
        position=7,
        argstr='-cerebellum-area-surf %s',
        desc='specify the cerebellum surface to use',
    )
    left_area_metric = File(
        exists=True,
        position=8,
        argstr='-left-area-metric %s',
        desc='Specify the left surface metric to use',
    )
    right_area_metric = File(
        exists=True,
        position=9,
        argstr='-right-area-metric %s',
        desc='Specify the right surface  metric to use',
    )
    cerebellum_area_metric = File(
        exists=True,
        position=10,
        argstr='-cerebellum-area-metric %s',
        desc='specify the cerebellum surface  metricto use',
    )

    cifti_weights = File(
        exists=True,
        position=11,
        argstr='-cifti-weights %s',
        desc='Use a cifti file containing weights',
    )
    cor_method = traits.Enum(
        'MEAN',
        'MAX',
        'MIN',
        'INDEXMAX',
        'INDEXMIN',
        'SUM',
        'PRODUCT',
        'STDEV',
        'SAMPSTDEV',
        'VARIANCE',
        'TSNR',
        'COV',
        'L2NORM',
        'MEDIAN',
        'MODE',
        'COUNT_NONZERO',
        position=12,
        usedefault=True,
        argstr='-method %s',
        desc='Specify method of parcellation (default MEAN, or MODE if label data)',
    )
    only_numeric = traits.Bool(
        position=13,
        argstr='-only-numeric',
        desc='Exclude non-numeric values',
    )


class _CiftiParcellateWorkbenchOutputSpec(TraitedSpec):
    """Output specification for the CiftiParcellateWorkbench command."""

    out_file = File(exists=True, desc='output CIFTI file')


class CiftiParcellateWorkbench(WBCommand):
    """Extract timeseries from CIFTI file.

    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries.
    """

    input_spec = _CiftiParcellateWorkbenchInputSpec
    output_spec = _CiftiParcellateWorkbenchOutputSpec
    _cmd = 'wb_command -cifti-parcellate'


class _CiftiSurfaceResampleInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiSurfaceResample command.

    Resamples a surface file, given two spherical surfaces that are in register.
    If ADAP_BARY_AREA is used, exactly one of -area-surfs or -area-metrics must be specified.
    This method is not generally recommended for surface resampling,
    but is provided for completeness.

    For cut surfaces (including flatmaps), use -surface-cut-resample.

    Instead of resampling a spherical surface,
    the -surface-sphere-project-unproject command is recommended.
    """

    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='the surface file to resample',
    )
    current_sphere = File(
        exists=True,
        position=1,
        argstr='%s',
        desc='a sphere surface with the mesh that the input surface is currently on',
    )
    new_sphere = File(
        exists=True,
        position=2,
        argstr='%s',
        desc=(
            'a sphere surface that is in register with <current-sphere> '
            'and has the desired output mesh'
        ),
    )
    method = traits.Enum(
        'BARYCENTRIC',
        'ADAP_BARY_AREA',
        argstr='%s',
        position=3,
        desc=(
            'the method name. '
            'The BARYCENTRIC method is generally recommended for anatomical surfaces, '
            'in order to minimize smoothing.'
        ),
        usedefault=True,
    )
    out_file = File(
        name_source=['in_file'],
        name_template='resampled_%s.gii',
        keep_extension=True,
        extensions=['.shape.gii', '.surf.gii'],
        argstr='%s',
        position=4,
        desc='The output surface file.',
    )


class _CiftiSurfaceResampleOutputSpec(TraitedSpec):
    """Output specification for the CiftiSurfaceResample command."""

    out_file = File(exists=True, desc='output gifti file')


class CiftiSurfaceResample(WBCommand):
    """Resample a surface to a different mesh.

    TODO: Improve documentation.
    """

    input_spec = _CiftiSurfaceResampleInputSpec
    output_spec = _CiftiSurfaceResampleOutputSpec
    _cmd = 'wb_command -surface-resample'


class _CiftiSeparateMetricInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiSeparateMetric command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s ',
        position=0,
        desc='The input dense series',
    )
    direction = traits.Enum(
        'ROW',
        'COLUMN',
        mandatory=True,
        argstr='%s ',
        position=1,
        desc='which dimension to smooth along, ROW or COLUMN',
    )
    metric = traits.Str(
        mandatory=True,
        argstr=' -metric %s ',
        position=2,
        desc='which of the structure eg CORTEX_LEFT CORTEX_RIGHT'
        'check https://www.humanconnectome.org/software/workbench-command/-cifti-separate ',
    )
    out_file = File(
        name_source=['in_file'],
        name_template='correlation_matrix_%s.func.gii',
        keep_extension=True,
        argstr=' %s',
        position=3,
        desc='The gifti output, either left and right',
    )


class _CiftiSeparateMetricOutputSpec(TraitedSpec):
    """Output specification for the CiftiSeparateMetric command."""

    out_file = File(exists=True, desc='output CIFTI file')


class CiftiSeparateMetric(WBCommand):
    """Extract left or right hemisphere surfaces from CIFTI file (.dtseries).

    Other structures can also be extracted.
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries,
    """

    input_spec = _CiftiSeparateMetricInputSpec
    output_spec = _CiftiSeparateMetricOutputSpec
    _cmd = 'wb_command  -cifti-separate '


class _CiftiSeparateVolumeAllInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiSeparateVolumeAll command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='The input dense series',
    )
    direction = traits.Enum(
        'ROW',
        'COLUMN',
        mandatory=True,
        argstr='%s',
        position=1,
        desc='which dimension to smooth along, ROW or COLUMN',
    )
    out_file = File(
        name_source=['in_file'],
        name_template='%s_volumetric_data.nii.gz',
        keep_extension=False,
        argstr='-volume-all %s -crop',
        position=2,
        desc='The NIFTI output.',
    )
    label_file = File(
        name_source=['in_file'],
        name_template='%s_labels.nii.gz',
        keep_extension=False,
        argstr='-label %s',
        position=3,
        desc='A discrete segmentation NIFTI output.',
    )


class _CiftiSeparateVolumeAllOutputSpec(TraitedSpec):
    """Output specification for the CiftiSeparateVolumeAll command."""

    label_file = File(exists=True, desc='NIFTI file with labels.')
    out_file = File(exists=True, desc='NIFTI file with volumetric data.')


class CiftiSeparateVolumeAll(WBCommand):
    """Extract volumetric data from CIFTI file (.dtseries).

    Other structures can also be extracted.
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries.
    """

    input_spec = _CiftiSeparateVolumeAllInputSpec
    output_spec = _CiftiSeparateVolumeAllOutputSpec
    _cmd = 'wb_command  -cifti-separate '


class _CiftiCreateDenseScalarInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiSeparateVolumeAll command."""

    out_file = File(
        exists=False,
        mandatory=False,
        genfile=True,
        argstr='%s',
        position=0,
        desc='The CIFTI output.',
    )
    left_metric = File(
        exists=True,
        mandatory=False,
        argstr='-left-metric %s',
        position=1,
        desc='The input surface data from the left hemisphere.',
    )
    right_metric = File(
        exists=True,
        mandatory=False,
        argstr='-right-metric %s',
        position=2,
        desc='The input surface data from the right hemisphere.',
    )
    volume_data = File(
        exists=True,
        mandatory=False,
        argstr='-volume %s',
        position=3,
        desc='The input volumetric data.',
    )
    structure_label_volume = File(
        exists=True,
        mandatory=False,
        argstr='%s',
        position=4,
        desc='A label file indicating the structure of each voxel in volume_data.',
    )


class _CiftiCreateDenseScalarOutputSpec(TraitedSpec):
    """Output specification for the CiftiCreateDenseScalar command."""

    out_file = File(exists=True, desc='output CIFTI file')


class CiftiCreateDenseScalar(WBCommand):
    """Extract volumetric data from CIFTI file (.dtseries).

    Other structures can also be extracted.
    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries.
    """

    input_spec = _CiftiCreateDenseScalarInputSpec
    output_spec = _CiftiCreateDenseScalarOutputSpec
    _cmd = 'wb_command -cifti-create-dense-scalar'

    def _gen_filename(self, name):
        if name != 'out_file':
            return None

        if isdefined(self.inputs.out_file):
            return self.inputs.out_file
        elif isdefined(self.inputs.volume_data):
            _, fname, _ = split_filename(self.inputs.volume_data)
        else:
            _, fname, _ = split_filename(self.inputs.left_metric)

        return f'{fname}_converted.dscalar.nii'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_filename('out_file'))
        return outputs


class _ShowSceneInputSpec(_WBCommandInputSpec):
    scene_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
    )
    scene_name_or_number = traits.Either(
        traits.Int,
        traits.Str,
        mandatory=True,
        position=1,
        argstr='%s',
        desc='name or number (starting at one) of the scene in the scene file',
    )
    out_file = File(
        exists=False,
        mandatory=False,
        argstr='%s',
        genfile=True,
        position=2,
        desc='output image file name',
    )
    image_width = traits.Int(
        mandatory=True,
        argstr='%s',
        position=3,
        desc='width of output image(s), in pixels',
    )
    image_height = traits.Int(
        mandatory=True,
        argstr='%s',
        position=4,
        desc='height of output image(s), in pixels',
    )


class _ShowSceneOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output image file name')


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
    _cmd = 'wb_command -show-scene'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        return self._gen_outfilename() if name == 'out_file' else None

    def _gen_outfilename(self):
        frame_number = self.inputs.scene_name_or_number
        return (
            f'frame_{frame_number:06g}.png'
            if isinstance(frame_number, int)
            else f'frame_{frame_number}.png'
        )


class _CiftiConvertInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiConvert command."""

    target = traits.Enum(
        'from',
        'to',
        mandatory=True,
        position=0,
        argstr='-%s-nifti',
        desc='Convert either to or from nifti.',
    )
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=1,
        desc='The input file.',
    )
    cifti_template = File(
        exists=True,
        mandatory=False,
        argstr='%s',
        position=2,
        desc='A cifti file with the dimension(s) and mapping(s) that should be used.',
    )
    TR = traits.Float(
        mandatory=False,
        desc='Repetition time in seconds. Used to reset timepoints.',
        position=4,
        argstr='-reset-timepoints %s 0',
    )
    out_file = File(
        exists=True,
        mandatory=False,
        genfile=True,
        argstr='%s',
        position=3,
        desc='The output file.',
    )


class _CiftiConvertOutputSpec(TraitedSpec):
    """Output specification for the CiftiConvert command."""

    out_file = File(
        exists=True,
        desc='The output file.',
    )


class CiftiConvert(WBCommand):
    """Convert between CIFTI and NIFTI file formats."""

    input_spec = _CiftiConvertInputSpec
    output_spec = _CiftiConvertOutputSpec
    _cmd = 'wb_command -cifti-convert'

    def _gen_filename(self, name):
        if name != 'out_file':
            return None

        _, fname, ext = split_filename(self.inputs.in_file)
        # if we want to support other cifti outputs, we'll need to change this.
        ext = '.dtseries.nii' if self.inputs.target == 'from' else '.nii.gz'
        return f'{fname}_converted{ext}'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_filename('out_file'))
        return outputs


class _CiftiCreateDenseFromTemplateInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiCreateDenseFromTemplate command."""

    template_cifti = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='File to match brainordinates of.',
    )
    out_file = File(
        exists=False,
        mandatory=False,
        genfile=True,
        argstr='%s',
        position=1,
        desc='The output cifti file.',
    )
    volume_all = File(
        exists=True,
        mandatory=False,
        argstr='-volume-all %s',
        position=2,
        desc='Use input data from volume files. Input volume file.',
    )
    from_cropped = traits.Bool(
        False,
        usedefault=True,
        mandatory=False,
        argstr='-from-cropped',
        position=3,
        desc='Use input data from cropped volume files.',
    )
    left_metric = File(
        exists=True,
        mandatory=False,
        argstr='-metric CORTEX_LEFT %s',
        position=4,
        desc='Use input data from surface files. Input surface file.',
    )
    right_metric = File(
        exists=True,
        mandatory=False,
        argstr='-metric CORTEX_RIGHT %s',
        position=5,
        desc='Use input data from surface files. Input surface file.',
    )
    label = File(
        exists=True,
        mandatory=False,
        argstr='-cifti %s',
        position=6,
        desc='Use input data from surface label files. Input label file.',
    )


class _CiftiCreateDenseFromTemplateOutputSpec(TraitedSpec):
    """Output specification for the CiftiCreateDenseFromTemplate command."""

    out_file = File(exists=True, desc='output CIFTI file')


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
    """

    input_spec = _CiftiCreateDenseFromTemplateInputSpec
    output_spec = _CiftiCreateDenseFromTemplateOutputSpec
    _cmd = 'wb_command -cifti-create-dense-from-template'

    def _gen_filename(self, name):
        if name != 'out_file':
            return None

        if isdefined(self.inputs.out_file):
            return self.inputs.out_file
        elif isdefined(self.inputs.label):
            _, fname, _ = split_filename(self.inputs.label)
        else:
            _, fname, _ = split_filename(self.inputs.template_cifti)

        return f'{fname}_converted.dscalar.nii'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath(self._gen_filename('out_file'))
        return outputs


class _CiftiMathInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiMath command."""

    data = File(
        exists=True,
        mandatory=True,
        argstr='-var data %s',
        position=2,
        desc='First data file to use in the math operation',
    )
    mask = File(
        exists=True,
        mandatory=False,
        argstr='-var mask %s -select 1 1',
        position=3,
        desc='Second data file to use in the math operation',
    )
    expression = traits.Str(
        mandatory=True,
        argstr='"%s"',
        position=0,
        desc='Math expression',
    )
    out_file = File(
        name_source=['data'],
        name_template='mathed_%s.nii',
        keep_extension=True,
        argstr='%s',
        position=1,
        desc='Output cifti file',
    )


class _CiftiMathOutputSpec(TraitedSpec):
    """Output specification for the CiftiMath command."""

    out_file = File(exists=True, desc='output CIFTI file')


class CiftiMath(WBCommand):
    """Evaluate expression on CIFTI files.

    I should use a dynamic trait for the variables going into the math expression,
    but I hardcoded data and mask because those are the only ones I'm currently using.
    """

    input_spec = _CiftiMathInputSpec
    output_spec = _CiftiMathOutputSpec
    _cmd = 'wb_command -cifti-math'


class _CiftiCorrelationInputSpec(_WBCommandInputSpec):
    """Input specification for the CiftiCorrelation command."""

    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='Input file to correlate',
    )
    out_file = File(
        name_source=['in_file'],
        name_template='corr_%s.pconn.nii',
        keep_extension=False,
        argstr='%s',
        position=1,
        desc='Output cifti file',
    )


class _CiftiCorrelationOutputSpec(TraitedSpec):
    """Output specification for the CiftiCorrelation command."""

    out_file = File(exists=True, desc='output CIFTI file')


class CiftiCorrelation(WBCommand):
    """Generate correlation of rows in CIFTI file.

    This interface only supports parcellated time series files for now.
    """

    input_spec = _CiftiCorrelationInputSpec
    output_spec = _CiftiCorrelationOutputSpec
    _cmd = 'wb_command -cifti-correlation'


class _CiftiSmoothInputSpec(_WBCommandInputSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        argstr='%s',
        position=0,
        desc='The input CIFTI file',
    )
    sigma_surf = traits.Float(
        mandatory=True,
        argstr='%s',
        position=1,
        desc='the sigma for the gaussian surface smoothing kernel, in mm',
    )
    sigma_vol = traits.Float(
        mandatory=True,
        argstr='%s',
        position=2,
        desc='the sigma for the gaussian volume smoothing kernel, in mm',
    )
    direction = traits.Enum(
        'ROW',
        'COLUMN',
        mandatory=True,
        argstr='%s',
        position=3,
        desc='which dimension to smooth along, ROW or COLUMN',
    )
    out_file = File(
        name_source=['in_file'],
        name_template='smoothed_%s.nii',
        keep_extension=True,
        argstr='%s',
        position=4,
        desc='The output CIFTI',
    )
    left_surf = File(
        exists=True,
        mandatory=True,
        position=5,
        argstr='-left-surface %s',
        desc='Specify the left surface to use',
    )
    left_corrected_areas = File(
        exists=True,
        position=6,
        argstr='-left-corrected-areas %s',
        desc='vertex areas (as a metric) to use instead of computing them from the left surface.',
    )
    right_surf = File(
        exists=True,
        mandatory=True,
        position=7,
        argstr='-right-surface %s',
        desc='Specify the right surface to use',
    )
    right_corrected_areas = File(
        exists=True,
        position=8,
        argstr='-right-corrected-areas %s',
        desc='vertex areas (as a metric) to use instead of computing them from the right surface',
    )
    cerebellum_surf = File(
        exists=True,
        position=9,
        argstr='-cerebellum-surface %s',
        desc='specify the cerebellum surface to use',
    )
    cerebellum_corrected_areas = File(
        exists=True,
        position=10,
        requires=['cerebellum_surf'],
        argstr='cerebellum-corrected-areas %s',
        desc='vertex areas (as a metric) to use instead of computing them from '
        'the cerebellum surface',
    )
    cifti_roi = File(
        exists=True,
        position=11,
        argstr='-cifti-roi %s',
        desc='CIFTI file for ROI smoothing',
    )
    fix_zeros_vol = traits.Bool(
        position=12,
        argstr='-fix-zeros-volume',
        desc='treat values of zero in the volume as missing data',
    )
    fix_zeros_surf = traits.Bool(
        position=13,
        argstr='-fix-zeros-surface',
        desc='treat values of zero on the surface as missing data',
    )
    merged_volume = traits.Bool(
        position=14,
        argstr='-merged-volume',
        desc='smooth across subcortical structure boundaries',
    )


class _CiftiSmoothOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output CIFTI file')


class CiftiSmooth(WBCommand):
    """Smooth a CIFTI file.

    The input cifti file must have a brain models mapping on the chosen
    dimension, columns for .dtseries, and either for .dconn.  By default,
    data in different structures is smoothed independently (i.e., "parcel
    constrained" smoothing), so volume structures that touch do not smooth
    across this boundary.  Specify ``merged_volume`` to ignore these
    boundaries. Surface smoothing uses the ``GEO_GAUSS_AREA`` smoothing method.

    The ``*_corrected_areas`` options are intended for when it is unavoidable
    to smooth on group average surfaces, it is only an approximate correction
    for the reduction of structure in a group average surface.  It is better
    to smooth the data on individuals before averaging, when feasible.

    The ``fix_zeros_*`` options will treat values of zero as lack of data, and
    not use that value when generating the smoothed values, but will fill
    zeros with extrapolated values.  The ROI should have a brain models
    mapping along columns, exactly matching the mapping of the chosen
    direction in the input file.  Data outside the ROI is ignored.
    """

    input_spec = _CiftiSmoothInputSpec
    output_spec = _CiftiSmoothOutputSpec
    _cmd = 'wb_command -cifti-smoothing'
