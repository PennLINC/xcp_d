# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Quality control plotting interfaces."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from xcp_d.utils.confounds import load_confound, load_motion
from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.modified_data import compute_fd
from xcp_d.utils.plot import FMRIPlot
from xcp_d.utils.qcmetrics import compute_dvars, compute_registration_qc
from xcp_d.utils.write_save import read_ndata, write_ndata

LOGGER = logging.getLogger("nipype.interface")


class _CensoringPlotInputSpec(BaseInterfaceInputSpec):
    bold_file = File(
        exists=True,
        mandatory=True,
        desc="Raw bold file from fMRIPrep. Used only to identify the right confounds file.",
    )
    tmask = File(exists=False, mandatory=False, desc="Temporal mask. Current unused.")
    dummytime = traits.Float(
        exists=False,
        mandatory=False,
        default_value=0,
        desc="Dummy time to drop",
    )
    TR = traits.Float(exists=True, mandatory=True, desc="Repetition Time")
    head_radius = traits.Float(
        exists=True,
        mandatory=False,
        default_value=50,
        desc="Head radius; recommended value is 40 for babies",
    )
    motion_filter_type = traits.Either(
        None,
        traits.Str,
        exists=False,
        mandatory=True,
    )
    motion_filter_order = traits.Int(exists=False, mandatory=True)
    band_stop_min = traits.Either(
        None,
        traits.Float,
        exists=False,
        mandatory=True,
        desc="Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )
    band_stop_max = traits.Either(
        None,
        traits.Float,
        exists=False,
        mandatory=True,
        desc="Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )
    fd_thresh = traits.Float(
        exists=False,
        mandatory=True,
        desc="Framewise displacement threshold."
    )


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

        if self.inputs.dummytime:
            initial_volumes_to_drop = int(np.ceil(self.inputs.dummytime / self.inputs.TR))
            ax.axvspan(
                0,
                initial_volumes_to_drop,
                label="Dummy Volumes",
                alpha=0.5,
                color=palette[1],
            )
        else:
            initial_volumes_to_drop = 0

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

        # NOTE: TS- Probably should replace with the actual tmask file.
        tmask = filtered_fd_timeseries >= self.inputs.fd_thresh
        tmask[:initial_volumes_to_drop] = False

        # Only plot censored volumes if any were flagged
        if sum(tmask) > 0:
            tmask_idx = np.where(tmask)[0]
            for i_idx, idx in enumerate(tmask_idx):
                if i_idx == 0:
                    label = "Censored Volumes"
                else:
                    label = ""

                ax.axvline(idx * self.inputs.TR, label=label, color=palette[3], alpha=0.5)

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


class _QCPlotInputSpec(BaseInterfaceInputSpec):
    bold_file = File(exists=True, mandatory=True, desc="Raw bold file from fMRIPrep")
    mask_file = File(exists=False, mandatory=False, desc="Mask file from nifti")
    seg_file = File(exists=False, mandatory=False, desc="Seg file for nifti")
    cleaned_file = File(exists=True, mandatory=True, desc="Processed file")
    tmask = File(exists=False, mandatory=False, desc="Temporal mask")
    dummytime = traits.Float(
        exists=False,
        mandatory=False,
        default_value=0,
        desc="Dummy time to drop",
    )
    TR = traits.Float(exists=True, mandatory=True, desc="Repetition Time")
    head_radius = traits.Float(
        exists=True,
        mandatory=False,
        default_value=50,
        desc="Head radius; recommended value is 40 for babies",
    )
    bold2T1w_mask = File(exists=False, mandatory=False, desc="Bold mask in MNI")
    bold2temp_mask = File(exists=False, mandatory=False, desc="Bold mask in T1W")
    template_mask = File(exists=False, mandatory=False, desc="Template mask")
    t1w_mask = File(exists=False, mandatory=False, desc="Mask in T1W")


class _QCPlotOutputSpec(TraitedSpec):
    qc_file = File(exists=True, mandatory=True, desc="qc file in tsv")
    raw_qcplot = File(exists=True, mandatory=True, desc="qc plot before regression")
    clean_qcplot = File(exists=True, mandatory=True, desc="qc plot after regression")


class QCPlot(SimpleInterface):
    """Generate a quality control (QC) figure.

    Examples
    --------
    .. testsetup::
    >>> from tempfile import TemporaryDirectory
    >>> tmpdir = TemporaryDirectory()
    >>> os.chdir(tmpdir.name)
    .. doctest::
    computeqcwf = QCPlot()
    computeqcwf.inputs.cleaned_file = datafile
    computeqcwf.inputs.bold_file = rawbold
    computeqcwf.inputs.TR = TR
    computeqcwf.inputs.tmask = temporalmask
    computeqcwf.inputs.mask_file = mask
    computeqcwf.inputs.dummytime = dummytime
    computeqcwf.run()
    .. testcleanup::
    >>> tmpdir.cleanup()
    """

    input_spec = _QCPlotInputSpec
    output_spec = _QCPlotOutputSpec

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

        if self.inputs.dummytime > 0:  # Calculate number of vols to drop if any
            initial_volumes_to_drop = int(np.ceil(self.inputs.dummytime / self.inputs.TR))
        else:
            initial_volumes_to_drop = 0

        # Drop volumes from time series
        # NOTE: TS- Why drop dummy volumes in preprocessed plot?
        preproc_fd_timeseries = preproc_fd_timeseries[initial_volumes_to_drop:]
        postproc_fd_timeseries = postproc_fd_timeseries[initial_volumes_to_drop:]
        rmsd = rmsd[initial_volumes_to_drop:]

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
            )[:, initial_volumes_to_drop:],
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
        )[:, initial_volumes_to_drop:]

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
            bold_file_name_componentsox_inches="tight",
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
            bold_file_name_componentsox_inches="tight",
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
            "nVolsRemoved": [initial_volumes_to_drop],
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
