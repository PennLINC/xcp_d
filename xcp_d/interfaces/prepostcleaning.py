"""Interfaces for the post-processing workflows."""
import numpy as np
import pandas as pd
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    traits,
)

from xcp_d.utils.confounds import load_motion
from xcp_d.utils.filemanip import fname_presuffix
from xcp_d.utils.modified_data import compute_fd, generate_mask


class _CensorScrubInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc=" Partially processed bold or nifti")
    fd_thresh = traits.Float(
        exists=True,
        mandatory=False,
        default_value=0.2,
        desc="Framewise displacement threshold. All values above this will be dropped.",
    )
    initial_volumes_to_drop = traits.Either(
        traits.Int,
        "auto",
        exists=False,
        mandatory=True,
        desc="Number of volumes to remove from the beginning of the BOLD run.",
    )
    fmriprep_confounds_file = File(
        exists=True,
        mandatory=True,
        desc="fMRIPrep confounds tsv.",
    )
    head_radius = traits.Float(
        exists=True,
        mandatory=False,
        default_value=50,
        desc="Head radius in mm.",
    )
    TR = traits.Float(mandatory=True, desc="Repetition time in seconds")
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
        exists=True,
        mandatory=True,
        desc="Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )
    band_stop_max = traits.Either(
        None,
        traits.Float,
        exists=True,
        mandatory=True,
        desc="Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).",
    )


class _CensorScrubOutputSpec(TraitedSpec):
    tmask = traits.Either(
        None,
        File,
        exists=True,
        mandatory=True,
        desc="Temporal mask; all values above fd_thresh set to 1",
    )
    filtered_motion = File(
        exists=True,
        mandatory=True,
        desc=(
            "Framewise displacement timeseries. "
            "This is a TSV file with one column: 'framewise_displacement'."
        ),
    )


class CensorScrub(SimpleInterface):
    """Generate a temporal mask based on recalculated FD.

    Takes in confound files, bold file to be censored, and information about filtering-
    including band stop values and motion filter type.
    Then proceeds to create a motion-filtered confounds matrix and recalculates FD from
    filtered motion parameters.
    Finally generates temporal mask with volumes above FD threshold set to 1,
    then dropped from both confounds file and bolds file.
    Outputs temporal mask, framewise displacement timeseries and censored bold files.
    """

    input_spec = _CensorScrubInputSpec
    output_spec = _CensorScrubOutputSpec

    def _run_interface(self, runtime):

        # Read in fmriprep confounds tsv to calculate FD
        fmriprep_confounds_df = pd.read_table(self.inputs.fmriprep_confounds_file)
        initial_volumes_to_drop = self.inputs.initial_volumes_to_drop

        initial_volumes_df = None
        if initial_volumes_to_drop == "auto":
            nss_cols = [
                c for c in fmriprep_confounds_df.columns
                if c.startswith("non_steady_state_outlier")
            ]
            initial_volumes_df = fmriprep_confounds_df[nss_cols]
        else:
            initial_volumes_columns = [f"dummy_volume_{i}" for i in range(initial_volumes_to_drop)]
            initial_volumes_array = np.vstack(
                (
                    np.eye(initial_volumes_to_drop, dtype=int),
                    np.zeros(
                        (
                            fmriprep_confounds_df.shape[0] - initial_volumes_to_drop,
                            initial_volumes_to_drop,
                        ),
                        dtype=int,
                    ),
                ),
            )
            initial_volumes_df = pd.DataFrame(
                data=initial_volumes_array,
                columns=initial_volumes_columns,
            )

        motion_df = load_motion(
            fmriprep_confounds_df.copy(),
            TR=self.inputs.TR,
            motion_filter_type=self.inputs.motion_filter_type,
            motion_filter_order=self.inputs.motion_filter_order,
            band_stop_min=self.inputs.band_stop_min,
            band_stop_max=self.inputs.band_stop_max,
        )

        fd_timeseries = compute_fd(
            confound=motion_df,
            head_radius=self.inputs.head_radius,
        )
        motion_df["framewise_displacement"] = fd_timeseries

        # Generate temporal mask with all timepoints have FD over threshold
        # set to 1 and then dropped.
        tmask = generate_mask(
            fd_res=fd_timeseries,
            fd_thresh=self.inputs.fd_thresh,
        )
        tmask_idx = np.where(tmask)[0]
        if tmask_idx.size:
            one_hot_outliers_columns = [
                f"framewise_displacement_outlier_{i}" for i in range(tmask_idx.size)
            ]
            one_hot_outliers = np.zeros(
                (fmriprep_confounds_df.shape[0], tmask_idx.size),
                dtype=int,
            )
            for i_idx, idx in enumerate(tmask_idx):
                one_hot_outliers[idx, i_idx] = 1

            one_hot_outliers_df = pd.DataFrame(
                data=one_hot_outliers,
                columns=one_hot_outliers_columns,
            )
        else:
            one_hot_outliers_df = pd.DataFrame()

        if one_hot_outliers_df.empty and initial_volumes_df.empty:
            outliers_df = None
        elif one_hot_outliers_df.empty:
            outliers_df = initial_volumes_df
        elif initial_volumes_df.empty:
            outliers_df = one_hot_outliers_df
        else:
            outliers_df = pd.concat((initial_volumes_df, one_hot_outliers_df), axis=1)

        if outliers_df is not None:
            self._results["tmask"] = fname_presuffix(
                self.inputs.in_file,
                suffix="_desc-fd_outliers.tsv",
                newpath=runtime.cwd,
                use_ext=False,
            )
            outliers_df.to_csv(
                self._results["tmask"],
                index=False,
                header=True,
                sep="\t",
            )
        else:
            self._results["tmask"] = None

        self._results["filtered_motion"] = fname_presuffix(
            self.inputs.in_file,
            suffix="_desc-filtered_motion.tsv",
            newpath=runtime.cwd,
            use_ext=False,
        )

        motion_df.to_csv(
            self._results["filtered_motion"],
            index=False,
            header=True,
            sep="\t",
        )

        return runtime
