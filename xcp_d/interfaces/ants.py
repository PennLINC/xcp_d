"""ANTS interfaces."""

import logging
import os

from nipype.interfaces.base import isdefined, traits
from niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms,
    _FixTraitApplyTransformsInputSpec,
)

LOGGER = logging.getLogger('nipype.interface')


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
            self.inputs.output_image = os.path.join(
                runtime.cwd,
                os.path.basename(self.inputs.input_image),
            )

        runtime = super()._run_interface(runtime)
        return runtime
