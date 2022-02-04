#!/usr/bin/env python

from nipype.interfaces.afni.utils import ReHoInputSpec, ReHoOutputSpec
from nipype.interfaces.afni.preprocess import DespikeInputSpec,AFNICommandOutputSpec
from nipype.interfaces.base import SimpleInterface, CommandLineInputSpec,TraitedSpec, File, traits
import os, shutil


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
        outfile = runtime.cwd + "/reho.nii.gz"
        shutil.copyfile(self.inputs.in_file, runtime.cwd+"/inset.nii.gz")
        shutil.copyfile(self.inputs.mask_file, runtime.cwd+"/mask.nii.gz")
        os.system("3dReHo -inset inset.nii.gz -mask mask.nii.gz -nneigh 27 -prefix reho.nii.gz")
        self._results['out_file'] = outfile


class DespikePatch(SimpleInterface):
    """Removes 'spikes' from the 3D+time input dataset

    For complete details, see the `3dDespike Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dDespike.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> despike = afni.Despike()
    >>> despike.inputs.in_file = 'functional.nii'
    >>> despike.cmdline
    '3dDespike -prefix functional_despike functional.nii'
    >>> res = despike.run()  # doctest: +SKIP

    """

    _cmd = "3dDespike"
    input_spec = DespikeInputSpec
    output_spec = AFNICommandOutputSpec

    def _run_interface(self, runtime):
        outfile = runtime.cwd + "/3despike.nii.gz"
        shutil.copyfile(self.inputs.in_file, runtime.cwd+"/inset.nii.gz")
        os.system("3dDespike -NEW -prefix  3despike.nii.gz inset.nii.gz")
        self._results['out_file'] = outfile



class ConstrastEnahncementInput(CommandLineInputSpec):
    in_file = File(
        mandatory=True,
        desc="The input file ",
    )
    out_file = File(
        desc="out file",
    )

class ConstrastEnahncementOutput(TraitedSpec):
    out_file = File(
        desc="out file",
    )


class ContrastEnhancement(SimpleInterface):
    """contrast enhancement with afni
    3dUnifize  -input inputdat   -prefix  t1w_contras.nii.gz
    """

    _cmd = "3dUnifize"
    input_spec = ConstrastEnahncementInput
    output_spec = ConstrastEnahncementOutput


    def _run_interface(self, runtime):
        outfile = runtime.cwd + "/3dunfixed.nii.gz"
        shutil.copyfile(self.inputs.in_file, runtime.cwd+"/inset.nii.gz")
        os.system("3dUnifize  -input inset.nii.gz   -prefix  3dunfixed.nii.gz")
        self._results['out_file'] = outfile
