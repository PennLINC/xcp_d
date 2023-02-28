"""Interfaces for the concatenation workflow."""
import os
import re

from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    Undefined,
    isdefined,
    traits,
)

from xcp_d.utils.concatenation import concatenate_niimgs, concatenate_tsvs

LOGGER = logging.getLogger("nipype.interface")


class _ConcatenateInputsInputSpec(BaseInterfaceInputSpec):
    preprocessed_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Preprocessed BOLD files, after dummy volume removal.",
    )
    confounds_file = traits.List(
        File(exists=True),
        mandatory=True,
        desc="TSV files with selected confounds for individual BOLD runs.",
    )
    filtered_motion = traits.List(
        File(exists=True),
        mandatory=True,
        desc="TSV files with filtered motion parameters, used for FD calculation.",
    )
    temporal_mask = traits.List(
        File(exists=True),
        mandatory=True,
        desc="TSV files with high-motion outliers indexed.",
    )
    uncensored_denoised_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Denoised BOLD data.",
    )
    filtered_denoised_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Denoised BOLD data.",
    )
    smoothed_denoised_bold = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        desc="Smoothed, denoised BOLD data. Optional.",
    )
    cifti = traits.Bool(
        mandatory=True,
        desc="Whether the data are CIFTIs (True) or NIFTIs (False).",
    )


class _ConcatenateInputsOutputSpec(TraitedSpec):
    preprocessed_bold = File(
        exists=True,
        desc="Concatenated preprocessed BOLD file.",
    )
    confounds_file = File(
        exists=True,
        desc="Concatenated TSV file with selected confounds.",
    )
    filtered_motion = File(
        exists=True,
        desc="Concatenated TSV file with filtered motion parameters, used for FD calculation.",
    )
    temporal_mask = File(
        exists=True,
        desc="Concatenated TSV file with high-motion outliers indexed.",
    )
    uncensored_denoised_bold = File(
        exists=True,
        desc="Concatenated denoised BOLD data.",
    )
    filtered_denoised_bold = File(
        exists=True,
        desc="Concatenated denoised BOLD data.",
    )
    smoothed_denoised_bold = traits.Either(
        File(exists=True),
        Undefined,
        desc="Concatenated, smoothed, denoised BOLD data. Optional.",
    )


class ConcatenateInputs(SimpleInterface):
    """Concatenate inputs."""

    input_spec = _ConcatenateInputsInputSpec
    output_spec = _ConcatenateInputsOutputSpec

    def _run_interface(self, runtime):
        niimg_inputs = {
            "preprocessed_bold": self.inputs.preprocessed_bold,
            "uncensored_denoised_bold": self.inputs.uncensored_denoised_bold,
            "filtered_denoised_bold": self.inputs.filtered_denoised_bold,
            "smoothed_denoised_bold": self.inputs.smoothed_denoised_bold,
        }
        tsv_inputs = {
            "confounds_file": self.inputs.confounds_file,
            "filtered_motion": self.inputs.filtered_motion,
            "temporal_mask": self.inputs.temporal_mask,
        }

        for niimg_name, run_files in niimg_inputs.items():
            LOGGER.warning(f"Concatenating {niimg_name}")
            if len(run_files) == 0 or not isdefined(run_files):
                LOGGER.warning(f"No {niimg_name} files found")
                self._results[niimg_name] = Undefined
                continue

            # TODO: We may need to support ptseries inputs.
            if self.inputs.cifti:
                out_file = os.path.join(runtime.cwd, f"{niimg_name}.dtseries.nii")
            else:
                out_file = os.path.join(runtime.cwd, f"{niimg_name}.nii.gz")

            concatenate_niimgs(run_files, out_file=out_file)
            assert os.path.isfile(out_file), f"Output file {out_file} not created."
            self._results[niimg_name] = out_file

        for tsv_name, run_files in tsv_inputs.items():
            LOGGER.info(f"Concatenating {tsv_name}")
            out_file = os.path.join(runtime.cwd, f"{tsv_name}.tsv")
            concatenate_tsvs(run_files, out_file=out_file)
            self._results[tsv_name] = out_file

        return runtime


class _FilterOutFailedRunsInputSpec(BaseInterfaceInputSpec):
    preprocessed_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Preprocessed BOLD files, after dummy volume removal.",
    )
    confounds_file = traits.List(
        File(exists=True),
        mandatory=True,
        desc="TSV files with selected confounds for individual BOLD runs.",
    )
    filtered_motion = traits.List(
        File(exists=True),
        mandatory=True,
        desc="TSV files with filtered motion parameters, used for FD calculation.",
    )
    temporal_mask = traits.List(
        File(exists=True),
        mandatory=True,
        desc="TSV files with high-motion outliers indexed.",
    )
    uncensored_denoised_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Denoised BOLD data.",
    )
    filtered_denoised_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Denoised BOLD data.",
    )
    smoothed_denoised_bold = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="Smoothed, denoised BOLD data.",
    )
    bold_mask = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="Smoothed, denoised BOLD data.",
    )
    boldref = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="Smoothed, denoised BOLD data.",
    )
    t1w_to_native_xform = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="Smoothed, denoised BOLD data.",
    )


class _FilterOutFailedRunsOutputSpec(TraitedSpec):
    preprocessed_bold = traits.List(
        File(exists=True),
        desc="Preprocessed BOLD files, after dummy volume removal.",
    )
    confounds_file = traits.List(
        File(exists=True),
        desc="TSV files with selected confounds for individual BOLD runs.",
    )
    filtered_motion = traits.List(
        File(exists=True),
        desc="TSV files with filtered motion parameters, used for FD calculation.",
    )
    temporal_mask = traits.List(
        File(exists=True),
        desc="TSV files with high-motion outliers indexed.",
    )
    uncensored_denoised_bold = traits.List(
        File(exists=True),
        desc="Denoised BOLD data.",
    )
    filtered_denoised_bold = traits.List(
        File(exists=True),
        desc="Denoised BOLD data.",
    )
    smoothed_denoised_bold = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        Undefined,
        desc="Smoothed, denoised BOLD data.",
    )
    bold_mask = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        desc="Smoothed, denoised BOLD data.",
    )
    boldref = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        Undefined,
        desc="Smoothed, denoised BOLD data.",
    )
    t1w_to_native_xform = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        Undefined,
        desc="Smoothed, denoised BOLD data.",
    )


class FilterOutFailedRuns(SimpleInterface):
    """Reduce several input lists based on whether entries in one list are defined or not."""

    input_spec = _FilterOutFailedRunsInputSpec
    output_spec = _FilterOutFailedRunsOutputSpec

    def _run_interface(self, runtime):
        filtered_denoised_bold = self.inputs.filtered_denoised_bold
        inputs_to_filter = {
            "preprocessed_bold": self.inputs.preprocessed_bold,
            "confounds_file": self.inputs.confounds_file,
            "filtered_motion": self.inputs.filtered_motion,
            "temporal_mask": self.inputs.temporal_mask,
            "uncensored_denoised_bold": self.inputs.uncensored_denoised_bold,
            "smoothed_denoised_bold": self.inputs.smoothed_denoised_bold,
            "bold_mask": self.inputs.bold_mask,
            "boldref": self.inputs.boldref,
            "t1w_to_native_xform": self.inputs.t1w_to_native_xform,
        }

        n_runs = len(filtered_denoised_bold)
        successful_runs = [i for i, f in enumerate(filtered_denoised_bold) if isdefined(f)]

        if len(successful_runs) < n_runs:
            LOGGER.warning(
                f"Of {n_runs} runs, only runs {', '.join(successful_runs)} were successful."
            )

        self._results["filtered_denoised_bold"] = [
            filtered_denoised_bold[i] for i in successful_runs
        ]

        for input_name, input_list in inputs_to_filter.items():
            if len(input_list) != n_runs:
                LOGGER.warning(
                    f"{input_name} has {len(input_list)} elements, not {n_runs}. Ignoring."
                )
                input_list = [Undefined for _ in range(n_runs)]

            self._results[input_name] = [input_list[i] for i in successful_runs]

        return runtime


class _CleanNameSourceInputSpec(BaseInterfaceInputSpec):
    name_source = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Name source files.",
    )


class _CleanNameSourceOutputSpec(TraitedSpec):
    name_source = traits.Str(
        desc="Name source",
    )


class CleanNameSource(SimpleInterface):
    """Remove run entity from the name source."""

    input_spec = _CleanNameSourceInputSpec
    output_spec = _CleanNameSourceOutputSpec

    def _run_interface(self, runtime):
        # Grab the first file and use that.
        name_source = self.inputs.name_source[0]
        # Remove the run entitty.
        cleaned_name_source = re.sub("_run-[0-9]+_", "_", name_source)
        self._results["name_source"] = cleaned_name_source
        return runtime
