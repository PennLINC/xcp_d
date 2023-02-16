# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Plotting interfaces."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.plotting import plot_anat
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiPath,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.fsl import FSLCommand, FSLCommandInputSpec

from xcp_d.utils.confounds import load_confound, load_motion
from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.modified_data import compute_fd
from xcp_d.utils.plotting import FMRIPlot, plot_fmri_es
from xcp_d.utils.qcmetrics import compute_dvars, compute_registration_qc
from xcp_d.utils.write_save import read_ndata, write_ndata

LOGGER = logging.getLogger("nipype.interface")


class _CensoringPlotInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc="Raw bold file from fMRIPrep. Used only to identify the right confounds file.",
    )
    tmask = File(exists=True, mandatory=True, desc="Temporal mask.")
    dummy_scans = traits.Int(mandatory=True, desc="Dummy time to drop")
    TR = traits.Float(mandatory=True, desc="Repetition Time")
    head_radius = traits.Float(mandatory=True, desc="Head radius for FD calculation")
    motion_filter_type = traits.Either(None, traits.Str, mandatory=True)
    motion_filter_order = traits.Int(mandatory=True)
    band_stop_min = traits.Either(
        None,
        traits.Float,
        mandatory=True,
        desc="Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )
    band_stop_max = traits.Either(
        None,
        traits.Float,
        mandatory=True,
        desc="Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )
    fd_thresh = traits.Float(mandatory=True, desc="Framewise displacement threshold.")


class _CensoringPlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, mandatory=True, desc="Censoring plot.")


class CensoringPlot(SimpleInterface):
    """Generate a censoring figure.

    This is a line plot showing both the raw and filtered framewise displacement time series,
    with vertical lines/bands indicating volumes removed by the post-processing workflow.

    """

    input_spec = _CensoringPlotInputSpec
    output_spec = _CensoringPlotOutputSpec

    def _run_interface(self, runtime):
        palette = sns.color_palette("colorblind", 4)

        # Load confound matrix and load motion with motion filtering
        confound_matrix = load_confound(datafile=self.inputs.bold_file)[0]
        preproc_motion_df = load_motion(
            confound_matrix.copy(),
            TR=self.inputs.TR,
            motion_filter_type=None,
        )
        preproc_fd_timeseries = compute_fd(
            confound=preproc_motion_df,
            head_radius=self.inputs.head_radius,
        )

        fig, ax = plt.subplots(figsize=(16, 8))

        time_array = np.arange(0, self.inputs.TR * preproc_fd_timeseries.size, self.inputs.TR)

        ax.plot(
            time_array,
            preproc_fd_timeseries,
            label="Raw Framewise Displacement",
            color=palette[0],
        )
        ax.axhline(self.inputs.fd_thresh, label="Outlier Threshold", color="gray", alpha=0.5)

        dummy_scans = self.inputs.dummy_scans
        # This check is necessary, because init_qc_report_wf connects dummy_scans from the
        # inputnode, forcing it to be undefined instead of using the default when not set.
        if not isdefined(dummy_scans):
            dummy_scans = 0

        if self.inputs.dummy_scans:
            ax.axvspan(
                0,
                self.inputs.dummy_scans * self.inputs.TR,
                label="Dummy Volumes",
                alpha=0.5,
                color=palette[1],
            )

        # Plot censored volumes as vertical lines
        tmask_df = pd.read_table(self.inputs.tmask)
        tmask_arr = tmask_df["framewise_displacement"].values
        tmask_idx = np.where(tmask_arr)[0]
        for i_idx, idx in enumerate(tmask_idx):
            if i_idx == 0:
                label = "Censored Volumes"
            else:
                label = ""

            idx_after_dummy_scans = idx + dummy_scans
            ax.axvline(
                idx_after_dummy_scans * self.inputs.TR,
                label=label,
                color=palette[3],
                alpha=0.5,
            )

        # Compute filtered framewise displacement to plot censoring
        if self.inputs.motion_filter_type:
            filtered_motion_df = load_motion(
                confound_matrix.copy(),
                TR=self.inputs.TR,
                motion_filter_type=self.inputs.motion_filter_type,
                motion_filter_order=self.inputs.motion_filter_order,
                band_stop_min=self.inputs.band_stop_min,
                band_stop_max=self.inputs.band_stop_max,
            )
            filtered_fd_timeseries = compute_fd(
                confound=filtered_motion_df,
                head_radius=self.inputs.head_radius,
            )

            ax.plot(
                time_array,
                filtered_fd_timeseries,
                label="Filtered Framewise Displacement",
                color=palette[2],
            )
        else:
            filtered_fd_timeseries = preproc_fd_timeseries.copy()

        ax.set_xlim(0, max(time_array))
        y_max = (
            np.max(
                np.hstack(
                    (
                        preproc_fd_timeseries,
                        filtered_fd_timeseries,
                        [self.inputs.fd_thresh],
                    )
                )
            )
            * 1.5
        )
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Time (seconds)", fontsize=20)
        ax.set_ylabel("Movement (millimeters)", fontsize=20)
        ax.legend(fontsize=20)
        fig.tight_layout()

        self._results["out_file"] = fname_presuffix(
            "censoring",
            suffix="_motion.svg",
            newpath=runtime.cwd,
            use_ext=False,
        )

        fig.savefig(self._results["out_file"])
        return runtime


class _QCPlotsInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True, mandatory=True, desc="Raw bold file from fMRIPrep")
    dummy_scans = traits.Int(mandatory=True, desc="Dummy time to drop")
    tmask = File(exists=True, mandatory=True, desc="Temporal mask")
    cleaned_file = File(exists=True, mandatory=True, desc="Processed file")
    TR = traits.Float(mandatory=True, desc="Repetition Time")
    head_radius = traits.Float(mandatory=True, desc="Head radius for FD calculation")
    mask_file = traits.Either(
        None,
        File(exists=True),
        mandatory=True,
        desc="Mask file from nifti. May be None, for CIFTI processing.",
    )

    # Inputs used only for nifti data
    seg_file = File(exists=True, mandatory=False, desc="Seg file for nifti")
    t1w_mask = File(exists=True, mandatory=False, desc="Mask in T1W")
    template_mask = File(exists=True, mandatory=False, desc="Template mask")
    bold2T1w_mask = File(exists=True, mandatory=False, desc="Bold mask in MNI")
    bold2temp_mask = File(exists=True, mandatory=False, desc="Bold mask in T1W")


class _QCPlotsOutputSpec(TraitedSpec):
    qc_file = File(exists=True, mandatory=True, desc="qc file in tsv")
    raw_qcplot = File(exists=True, mandatory=True, desc="qc plot before regression")
    clean_qcplot = File(exists=True, mandatory=True, desc="qc plot after regression")


class QCPlots(SimpleInterface):
    """Generate pre- and post-processing quality control (QC) figures.

    Examples
    --------
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    qcplots = QCPlots()
    qcplots.inputs.cleaned_file = datafile
    qcplots.inputs.bold_file = rawbold
    qcplots.inputs.TR = TR
    qcplots.inputs.tmask = temporalmask
    qcplots.inputs.mask_file = mask
    qcplots.inputs.dummy_scans = dummy_scans
    qcplots.run()
    .. testcleanup::
    >>> tmpdir.cleanup()
    """

    input_spec = _QCPlotsInputSpec
    output_spec = _QCPlotsOutputSpec

    def _run_interface(self, runtime):
        # Load confound matrix and load motion with motion filtering
        confound_matrix = load_confound(datafile=self.inputs.bold_file)[0]
        preproc_motion_df = load_motion(
            confound_matrix.copy(),
            TR=self.inputs.TR,
            motion_filter_type=None,
        )
        preproc_fd_timeseries = compute_fd(
            confound=preproc_motion_df,
            head_radius=self.inputs.head_radius,
        )
        postproc_fd_timeseries = preproc_fd_timeseries.copy()

        # Get rmsd
        rmsd = confound_matrix["rmsd"]

        # Drop volumes from time series
        # NOTE: TS- Why drop dummy volumes in preprocessed plot?
        dummy_scans = self.inputs.dummy_scans
        preproc_fd_timeseries = preproc_fd_timeseries[dummy_scans:]
        postproc_fd_timeseries = postproc_fd_timeseries[dummy_scans:]
        rmsd = rmsd[dummy_scans:]

        if self.inputs.tmask:  # If a tmask is provided, find # vols censored
            tmask_df = pd.read_table(self.inputs.tmask)
            tmask_arr = tmask_df["framewise_displacement"].values
            num_censored_volumes = np.sum(tmask_arr)
        else:
            num_censored_volumes = 0

        # Compute the DVARS for both bold files provided
        dvars_before_processing = compute_dvars(
            read_ndata(
                datafile=self.inputs.bold_file,
                maskfile=self.inputs.mask_file,
            )[:, dummy_scans:],
        )
        dvars_after_processing = compute_dvars(
            read_ndata(
                datafile=self.inputs.cleaned_file,
                maskfile=self.inputs.mask_file,
            ),
        )

        # get QC plot names
        self._results["raw_qcplot"] = fname_presuffix(
            "preprocess",
            suffix="_raw_qcplot.svg",
            newpath=runtime.cwd,
            use_ext=False,
        )
        self._results["clean_qcplot"] = fname_presuffix(
            "postprocess",
            suffix="_clean_qcplot.svg",
            newpath=runtime.cwd,
            use_ext=False,
        )
        raw_data_removed_TR = read_ndata(
            datafile=self.inputs.bold_file,
            maskfile=self.inputs.mask_file,
        )[:, dummy_scans:]

        # Get file names to write out & write data out
        dropped_bold_file = fname_presuffix(
            self.inputs.bold_file,
            newpath=runtime.cwd,
            suffix="_dropped",
            use_ext=True,
        )

        write_ndata(
            data_matrix=raw_data_removed_TR,
            template=self.inputs.bold_file,
            mask=self.inputs.mask_file,
            filename=dropped_bold_file,
            TR=self.inputs.TR,
        )

        preproc_confounds = pd.DataFrame(
            {
                "FD": preproc_fd_timeseries,
                "DVARS": dvars_before_processing,
            }
        )

        preproc_fig = FMRIPlot(
            func_file=dropped_bold_file,
            seg_file=self.inputs.seg_file,
            data=preproc_confounds,
            mask_file=self.inputs.mask_file,
        ).plot(labelsize=8)

        preproc_fig.savefig(
            self._results["raw_qcplot"],
            bbox_inches="tight",
        )

        # If censoring occurs, censor the cleaned BOLD data and FD time series
        # NOTE: TS- Why are we censoring these plots? This ignores/misrepresents the
        # interpolation step.
        if num_censored_volumes > 0:
            # Apply temporal mask to time series
            postproc_fd_timeseries = preproc_fd_timeseries[tmask_arr == 0]
            rmsd = rmsd[tmask_arr == 0]
            # NOTE: TS- Why mask DVARS before processing?
            dvars_before_processing = dvars_before_processing[tmask_arr == 0]
            dvars_after_processing = dvars_after_processing[tmask_arr == 0]

            # Apply temporal mask to data
            raw_data_removed_TR = read_ndata(
                datafile=self.inputs.cleaned_file,
                maskfile=self.inputs.mask_file,
            )
            raw_data_censored = raw_data_removed_TR[:, tmask_arr == 0]

            # Get temporary filename and write data out
            dropped_clean_file = fname_presuffix(
                self.inputs.bold_file,
                newpath=runtime.cwd,
                suffix="_droppedClean",
                use_ext=True,
            )

            write_ndata(
                data_matrix=raw_data_censored,
                template=self.inputs.bold_file,
                mask=self.inputs.mask_file,
                filename=dropped_clean_file,
                TR=self.inputs.TR,
            )

        else:
            dropped_clean_file = self.inputs.cleaned_file

        postproc_confounds = pd.DataFrame(
            {
                "FD": postproc_fd_timeseries,
                "DVARS": dvars_after_processing,
            }
        )

        postproc_fig = FMRIPlot(
            func_file=dropped_clean_file,
            seg_file=self.inputs.seg_file,
            data=postproc_confounds,
            mask_file=self.inputs.mask_file,
        ).plot(labelsize=8)

        postproc_fig.savefig(
            self._results["clean_qcplot"],
            bbox_inches="tight",
        )

        # Calculate QC measures
        mean_fd = np.mean(postproc_fd_timeseries)
        mean_rms = np.mean(rmsd)
        mean_dvars_before_processing = np.mean(dvars_before_processing)
        mean_dvars_after_processing = np.mean(dvars_after_processing)
        # NOTE: TS- If we didn't mask DVARS before postproc, we'd use preproc_fd_timeseries here.
        motionDVCorrInit = np.corrcoef(postproc_fd_timeseries, dvars_before_processing)[0][1]
        motionDVCorrFinal = np.corrcoef(postproc_fd_timeseries, dvars_after_processing)[0][1]
        rmsd_max_value = np.max(rmsd)

        # A summary of all the values
        qc_values = {
            "meanFD": [mean_fd],
            "relMeansRMSMotion": [mean_rms],
            "relMaxRMSMotion": [rmsd_max_value],
            "meanDVInit": [mean_dvars_before_processing],
            "meanDVFinal": [mean_dvars_after_processing],
            "num_censored_volumes": [num_censored_volumes],
            "nVolsRemoved": [dummy_scans],
            "motionDVCorrInit": [motionDVCorrInit],
            "motionDVCorrFinal": [motionDVCorrFinal],
        }

        # Get the different components in the bold file name
        # eg: ['sub-colornest001', 'ses-1'], etc.
        _, bold_file_name = os.path.split(self.inputs.bold_file)
        bold_file_name_components = bold_file_name.split("_")

        # Fill out dictionary with entities from filename
        qc_dictionary = {}
        for entity in bold_file_name_components[:-1]:
            qc_dictionary.update({entity.split("-")[0]: entity.split("-")[1]})

        qc_dictionary.update(qc_values)
        if self.inputs.bold2T1w_mask:  # If a bold mask in T1w is provided
            # Compute quality of registration
            registration_qc = compute_registration_qc(
                bold2t1w_mask=self.inputs.bold2T1w_mask,
                t1w_mask=self.inputs.t1w_mask,
                bold2template_mask=self.inputs.bold2temp_mask,
                template_mask=self.inputs.template_mask,
            )
            qc_dictionary.update(registration_qc)  # Add values to dictionary

        # Convert dictionary to df and write out the qc file
        df = pd.DataFrame(qc_dictionary)
        self._results["qc_file"] = fname_presuffix(
            self.inputs.cleaned_file,
            suffix="qc_bold.csv",
            newpath=runtime.cwd,
            use_ext=False,
        )
        df.to_csv(self._results["qc_file"], index=False, header=True)

        return runtime


class _QCPlotsESInputSpec(BaseInterfaceInputSpec):
    rawdata = File(exists=True, mandatory=True, desc="Raw data")
    regressed_data = File(
        exists=True,
        mandatory=True,
        desc="Data after regression and interpolation, but not filtering.",
    )
    residual_data = File(exists=True, mandatory=True, desc="Data after filtering")
    filtered_motion = File(
        exists=True,
        mandatory=True,
        desc="TSV file with filtered motion parameters.",
    )
    TR = traits.Float(default_value=1, desc="Repetition time")

    # Optional inputs
    mask = File(exists=True, mandatory=False, desc="Bold mask")
    seg_data = File(exists=True, mandatory=False, desc="Segmentation file")
    dummy_scans = traits.Int(
        0,
        usedefault=True,
        desc="Number of dummy volumes to drop from the beginning of the run.",
    )


class _QCPlotsESOutputSpec(TraitedSpec):
    before_process = File(exists=True, mandatory=True, desc=".SVG file before processing")
    after_process = File(exists=True, mandatory=True, desc=".SVG file after processing")


class QCPlotsES(SimpleInterface):
    """Plot fd, dvars, and carpet plots of the bold data before and after regression/filtering.

    This is essentially equivalent to the QCPlots
    (which are paired pre- and post-processing FMRIPlots), but adapted for the executive summary.

    It takes in the data that's regressed, the data that's filtered and regressed,
    as well as the segmentation files, TR, FD, bold_mask and unprocessed data.

    It outputs the .SVG files before after processing has taken place.
    """

    input_spec = _QCPlotsESInputSpec
    output_spec = _QCPlotsESOutputSpec

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

        self._results["before_process"], self._results["after_process"] = plot_fmri_es(
            preprocessed_file=self.inputs.rawdata,
            residuals_file=self.inputs.regressed_data,
            denoised_file=self.inputs.residual_data,
            dummy_scans=self.inputs.dummy_scans,
            TR=self.inputs.TR,
            mask=mask_file,
            filtered_motion=self.inputs.filtered_motion,
            seg_data=segmentation_file,
            processed_filename=after_process_fn,
            unprocessed_filename=before_process_fn,
        )

        return runtime


class _AnatomicalPlotInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="plot image")


class _AnatomicalPlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="out image")


class AnatomicalPlot(SimpleInterface):
    """Python class to plot x,y, and z of image data."""

    input_spec = _AnatomicalPlotInputSpec
    output_spec = _AnatomicalPlotOutputSpec

    def _run_interface(self, runtime):
        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file, suffix="_file.svg", newpath=runtime.cwd, use_ext=False
        )

        fig = plt.figure(constrained_layout=False, figsize=(25, 10))
        plot_anat(self.inputs.in_file, draw_cross=False, figure=fig)
        fig.savefig(self._results["out_file"], bbox_inches="tight", pad_inches=None)

        return runtime


class _PNGAppendInputSpec(FSLCommandInputSpec):
    in_files = InputMultiPath(
        exists=True,
        position=0,
        argstr="%s",
        mandatory=True,
        desc="input images to combine",
    )
    out_file = File(
        position=-1,
        genfile=True,
        argstr="%s",
        desc="picture to write",
        hash_files=False,
    )


class _PNGAppendOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="picture to write")


class PNGAppend(FSLCommand):
    """Use FSL's pngappend command."""

    _cmd = "pngappend"
    input_spec = _PNGAppendInputSpec
    output_spec = _PNGAppendOutputSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        out_file = self.inputs.out_file
        if not isdefined(out_file):
            out_file = self._gen_fname(self.inputs.in_files[0], ext=".gif")

        outputs["out_file"] = os.path.abspath(out_file)

        return outputs

    def _gen_filename(self, name):
        if name == "out_file":
            return self._list_outputs()["out_file"]

        return None
