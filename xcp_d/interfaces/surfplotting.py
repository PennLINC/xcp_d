# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Surface plotting interfaces."""
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.plot import plot_svgx, plotimage

LOGGER = logging.getLogger("nipype.interface")


class _PlotImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="plot image")


class _PlotImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="out image")


class PlotImage(SimpleInterface):
    """Python class to plot x,y, and z of image data."""

    input_spec = _PlotImageInputSpec
    output_spec = _PlotImageOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_file.svg", newpath=runtime.cwd, use_ext=False
        )

        self._results["out_file"] = plotimage(self.inputs.in_file, self._results["out_file"])

        return runtime


class _PlotSVGDataInputSpec(BaseInterfaceInputSpec):
    rawdata = File(exists=True, mandatory=True, desc="Raw data")
    regressed_data = File(exists=True, mandatory=True, desc="Data after regression")
    residual_data = File(exists=True, mandatory=True, desc="Data after filtering")
    filtered_motion = File(
        exists=True,
        mandatory=True,
        desc="TSV file with filtered motion parameters.",
    )
    TR = traits.Float(default_value=1, desc="Repetition time")

    # Optional inputs
    mask = File(exists=True, mandatory=False, desc="Bold mask")
    tmask = File(exists=True, mandatory=False, desc="Temporal mask")
    seg_data = File(exists=True, mandatory=False, desc="Segmentation file")
    dummy_scans = traits.Int(
        0,
        usedefault=True,
        desc="Number of dummy volumes to drop from the beginning of the run.",
    )


class _PlotSVGDataOutputSpec(TraitedSpec):
    before_process = File(exists=True, mandatory=True, desc=".SVG file before processing")
    after_process = File(exists=True, mandatory=True, desc=".SVG file after processing")


class PlotSVGData(SimpleInterface):
    """Plot fd, dvars, and carpet plots of the bold data before and after regression/filtering.

    It takes in the data that's regressed, the data that's filtered and regressed,
    as well as the segmentation files, TR, FD, bold_mask and unprocessed data.

    It outputs the .SVG files before after processing has taken place.
    """

    input_spec = _PlotSVGDataInputSpec
    output_spec = _PlotSVGDataOutputSpec

    def _run_interface(self, runtime):
        before_process_fn = fname_presuffix(
            "carpetplot_before_",
            suffix="file.svg",
            newpath=runtime.cwd,
            use_ext=False,
        )

        after_process_fn = fname_presuffix(
            "carpetplot_after_",
            suffix="file.svg",
            newpath=runtime.cwd,
            use_ext=False,
        )

        mask_file = self.inputs.mask
        mask_file = mask_file if isdefined(mask_file) else None

        segmentation_file = self.inputs.seg_data
        segmentation_file = segmentation_file if isdefined(segmentation_file) else None

        self._results["before_process"], self._results["after_process"] = plot_svgx(
            preprocessed_file=self.inputs.rawdata,
            residuals_file=self.inputs.regressed_data,
            denoised_file=self.inputs.residual_data,
            tmask=self.inputs.tmask,
            dummy_scans=self.inputs.dummy_scans,
            TR=self.inputs.TR,
            mask=mask_file,
            filtered_motion=self.inputs.filtered_motion,
            seg_data=segmentation_file,
            processed_filename=after_process_fn,
            unprocessed_filename=before_process_fn,
        )

        return runtime
