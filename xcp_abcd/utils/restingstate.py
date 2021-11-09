#!/usr/bin/env python

from nipype.interfaces.afni.utils import ReHoInputSpec, ReHoOutputSpec
from nipype.interfaces.base import SimpleInterface
import os 


class ReHoNamePatch(SimpleInterface):
    """Compute regional homogenity for a given neighbourhood.l,
    based on a local neighborhood of that voxel.
    For complete details, see the `3dReHo Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dReHo.html>`_
    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> reho = afni.ReHo()
    >>> reho.inputs.in_file = 'functional.nii'
    >>> reho.inputs.out_file = 'reho.nii.gz'
    >>> reho.inputs.neighborhood = 'vertices'
    >>> reho.cmdline
    '3dReHo -prefix reho.nii.gz -inset functional.nii -nneigh 27'
    >>> res = reho.run()  # doctest: +SKIP
    """

    _cmd = "3dReHo"
    input_spec = ReHoInputSpec
    output_spec = ReHoOutputSpec


    def _run_interface(self, runtime):

        new_inset ="cp " +  self.inputs.in_file + " " + runtime.cwd + "/inset.nii.gz"
        new_mask ="cp " + self.inputs.mask_file + " " + runtime.cwd + "/mask.nii.gz"
        outfile = runtime.cwd + "/reho.nii.gz"
        os.system(new_inset)
        os.system(new_mask)
        os.system("3dReHo -inset inset.nii.gz -mask mask.nii.gz -nneigh 27 -prefix prefix.nii.gz")
        self.results['out_file'] = outfile

