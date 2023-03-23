"""Interfaces for the concatenation workflow."""
import itertools
import os
import re

import pandas as pd
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


class _FilterOutFailedRunsInputSpec(BaseInterfaceInputSpec):
    censored_denoised_bold = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        mandatory=True,
        desc="Denoised BOLD data. This is used to index successful runs.",
    )
    preprocessed_bold = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        mandatory=True,
        desc="Preprocessed BOLD files, after dummy volume removal.",
    )
    fmriprep_confounds_file = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        mandatory=True,
        desc="TSV files with fMRIPrep confounds for individual BOLD runs.",
    )
    filtered_motion = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        mandatory=True,
        desc="TSV files with filtered motion parameters, used for FD calculation.",
    )
    temporal_mask = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        mandatory=True,
        desc="TSV files with high-motion outliers indexed.",
    )
    uncensored_denoised_bold = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        mandatory=True,
        desc="Denoised BOLD data.",
    )
    interpolated_filtered_bold = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        mandatory=True,
        desc="Denoised BOLD data.",
    )
    smoothed_denoised_bold = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="Smoothed, denoised BOLD data. Only set if smoothing was done in postprocessing",
    )
    bold_mask = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="BOLD-based brain mask file. Only used for NIFTI processing.",
    )
    boldref = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="BOLD reference files. Only used for NIFTI processing.",
    )
    anat_to_native_xfm = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc="T1w-to-native space transform files. Only used for NIFTI processing.",
    )
    atlas_names = traits.List(
        traits.List(traits.Str),
        mandatory=True,
        desc="List of lists of atlas names, corresponding to timeseries and timeseries_ciftis.",
    )
    timeseries = traits.List(
        traits.List(File(exists=True)),
        mandatory=True,
        desc="List of lists of parcellated time series TSV files.",
    )
    timeseries_ciftis = traits.Either(
        traits.List(traits.List(File(exists=True))),
        Undefined,
        desc=(
            "List of lists of parcellated time series CIFTI files. "
            "Only defined for CIFTI processing."
        ),
    )


class _FilterOutFailedRunsOutputSpec(TraitedSpec):
    censored_denoised_bold = traits.List(
        File(exists=True),
        desc="Denoised BOLD data.",
    )
    preprocessed_bold = traits.List(
        File(exists=True),
        desc="Preprocessed BOLD files, after dummy volume removal.",
    )
    fmriprep_confounds_file = traits.List(
        File(exists=True),
        desc="fMRIPrep confounds files, after dummy volume removal.",
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
    interpolated_filtered_bold = traits.List(
        File(exists=True),
        desc="Denoised BOLD data.",
    )
    smoothed_denoised_bold = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
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
        desc="Smoothed, denoised BOLD data.",
    )
    anat_to_native_xfm = traits.List(
        traits.Either(
            File(exists=True),
            Undefined,
        ),
        desc="Smoothed, denoised BOLD data.",
    )
    atlas_names = traits.List(
        traits.List(traits.Str),
        desc="List of lists of atlas names, corresponding to timeseries and timeseries_ciftis.",
    )
    timeseries = traits.List(
        traits.List(File(exists=True)),
        desc="List of lists of parcellated time series TSV files.",
    )
    timeseries_ciftis = traits.List(
        traits.Either(
            traits.List(File(exists=True)),
            Undefined,
        ),
        desc=(
            "List of lists of parcellated time series CIFTI files. "
            "Only defined for CIFTI processing."
        ),
    )


class FilterOutFailedRuns(SimpleInterface):
    """Reduce several input lists based on whether entries in one list are defined or not."""

    input_spec = _FilterOutFailedRunsInputSpec
    output_spec = _FilterOutFailedRunsOutputSpec

    def _run_interface(self, runtime):
        censored_denoised_bold = self.inputs.censored_denoised_bold
        inputs_to_filter = {
            "preprocessed_bold": self.inputs.preprocessed_bold,
            "fmriprep_confounds_file": self.inputs.fmriprep_confounds_file,
            "filtered_motion": self.inputs.filtered_motion,
            "temporal_mask": self.inputs.temporal_mask,
            "uncensored_denoised_bold": self.inputs.uncensored_denoised_bold,
            "interpolated_filtered_bold": self.inputs.interpolated_filtered_bold,
            "smoothed_denoised_bold": self.inputs.smoothed_denoised_bold,
            "bold_mask": self.inputs.bold_mask,
            "boldref": self.inputs.boldref,
            "anat_to_native_xfm": self.inputs.anat_to_native_xfm,
            "atlas_names": self.inputs.atlas_names,
            "timeseries": self.inputs.timeseries,
            "timeseries_ciftis": self.inputs.timeseries_ciftis,
        }

        n_runs = len(censored_denoised_bold)
        successful_runs = [i for i, f in enumerate(censored_denoised_bold) if isdefined(f)]

        if len(successful_runs) < n_runs:
            LOGGER.warning(f"Of {n_runs} runs, only runs {successful_runs} were successful.")

        self._results["censored_denoised_bold"] = [
            censored_denoised_bold[i] for i in successful_runs
        ]

        for input_name, input_list in inputs_to_filter.items():
            if len(input_list) != n_runs:
                LOGGER.warning(
                    f"{input_name} has {len(input_list)} elements, not {n_runs}. Ignoring."
                )
                input_list = [Undefined for _ in range(n_runs)]

            self._results[input_name] = [input_list[i] for i in successful_runs]

        return runtime


class _ConcatenateInputsInputSpec(BaseInterfaceInputSpec):
    censored_denoised_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Denoised BOLD data.",
    )
    preprocessed_bold = traits.List(
        File(exists=True),
        mandatory=True,
        desc="Preprocessed BOLD files, after dummy volume removal.",
    )
    fmriprep_confounds_file = traits.List(
        File(exists=True),
        mandatory=True,
        desc="TSV files with fMRIPrep confounds for individual BOLD runs.",
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
    interpolated_filtered_bold = traits.List(
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
    timeseries = traits.List(
        traits.List(File(exists=True)),
        desc="List of lists of parcellated time series TSV files.",
    )
    timeseries_ciftis = traits.List(
        traits.Either(
            traits.List(File(exists=True)),
            Undefined,
        ),
        desc=(
            "List of lists of parcellated time series CIFTI files. "
            "Only defined for CIFTI processing."
        ),
    )


class _ConcatenateInputsOutputSpec(TraitedSpec):
    censored_denoised_bold = File(
        exists=True,
        desc="Concatenated denoised BOLD data.",
    )
    preprocessed_bold = File(
        exists=True,
        desc="Concatenated preprocessed BOLD file.",
    )
    fmriprep_confounds_file = File(
        exists=True,
        desc="Concatenated TSV file with fMRIPrep confounds.",
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
    interpolated_filtered_bold = File(
        exists=True,
        desc="Concatenated denoised BOLD data.",
    )
    smoothed_denoised_bold = traits.Either(
        File(exists=True),
        Undefined,
        desc="Concatenated, smoothed, denoised BOLD data. Optional.",
    )
    timeseries = traits.List(
        File(exists=True),
        desc="Concatenated list of parcellated time series TSV files.",
    )
    timeseries_ciftis = traits.Either(
        traits.List(File(exists=True)),
        Undefined,
        desc=(
            "Concatenated list of parcellated time series CIFTI files. "
            "Only defined for CIFTI processing."
        ),
    )
    run_index = traits.List(
        traits.Int(),
        desc="Index of join points between the *uncensored* runs.",
    )


class ConcatenateInputs(SimpleInterface):
    """Concatenate inputs."""

    input_spec = _ConcatenateInputsInputSpec
    output_spec = _ConcatenateInputsOutputSpec

    def _run_interface(self, runtime):
        merge_inputs = {
            "censored_denoised_bold": self.inputs.censored_denoised_bold,
            "preprocessed_bold": self.inputs.preprocessed_bold,
            "uncensored_denoised_bold": self.inputs.uncensored_denoised_bold,
            "interpolated_filtered_bold": self.inputs.interpolated_filtered_bold,
            "smoothed_denoised_bold": self.inputs.smoothed_denoised_bold,
            "timeseries_ciftis": self.inputs.timeseries_ciftis,
            "fmriprep_confounds_file": self.inputs.fmriprep_confounds_file,
            "filtered_motion": self.inputs.filtered_motion,
            "temporal_mask": self.inputs.temporal_mask,
            "timeseries": self.inputs.timeseries,
        }

        run_index, n_volumes = [], 0
        for run_tmask in self.inputs.temporal_mask[:-1]:
            n_volumes = n_volumes + pd.read_table(run_tmask).shape[0]
            run_index.append(n_volumes)

        self._results["run_index"] = run_index

        for name, run_files in merge_inputs.items():
            LOGGER.info(f"Concatenating {name}")
            if len(run_files) == 0 or any(not isdefined(f) for f in run_files):
                LOGGER.warning(f"No {name} files found")
                self._results[name] = Undefined
                continue

            elif isinstance(run_files[0], list) and not isdefined(run_files[0][0]):
                LOGGER.warning(f"No {name} files found")
                self._results[name] = Undefined
                continue

            if isinstance(run_files[0], list):
                # Files are organized in a list of lists, like parcellated time series.
                transposed_run_files = list(
                    map(list, itertools.zip_longest(*run_files, fillvalue=None))
                )
                out_files = []
                for i_atlas, parc_files in enumerate(transposed_run_files):
                    extension = ".".join(os.path.basename(parc_files[0]).split(".")[1:])
                    out_file = os.path.join(runtime.cwd, f"{name}_{i_atlas}.{extension}")
                    if out_file.endswith(".tsv"):
                        concatenate_tsvs(parc_files, out_file=out_file)
                    else:
                        concatenate_niimgs(parc_files, out_file=out_file)

                    assert os.path.isfile(out_file), f"Output file {out_file} not created."
                    out_files.append(out_file)

                self._results[name] = out_files
            else:
                # Files are a single list of paths.
                extension = ".".join(os.path.basename(run_files[0]).split(".")[1:])
                out_file = os.path.join(runtime.cwd, f"{name}.{extension}")
                if out_file.endswith(".tsv"):
                    concatenate_tsvs(run_files, out_file=out_file)
                else:
                    concatenate_niimgs(run_files, out_file=out_file)

                assert os.path.isfile(out_file), f"Output file {out_file} not created."
                self._results[name] = out_file

        return runtime
