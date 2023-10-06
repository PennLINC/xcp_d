#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The xcp_d preprocessing worklow."""
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

from packaging.version import Version

from xcp_d import config
from xcp_d.cli.parser_utils import (
    _float_or_auto,
    _int_or_auto,
    _restricted_float,
    json_file,
)
from xcp_d.cli.version import check_latest, is_flagged


def get_parser():
    """Build parser object."""
    verstr = f"XCP-D v{config.environment.version}"
    currentv = Version(config.environment.version)

    parser = ArgumentParser(
        description="xcp_d postprocessing workflow of fMRI data",
        epilog="see https://xcp-d.readthedocs.io/en/latest/workflows.html",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # important parameters required
    parser.add_argument(
        "fmri_dir",
        action="store",
        type=Path,
        help=(
            "The root folder of fMRI preprocessing derivatives. "
            "For example, '/path/to/dset/derivatives/fmriprep'."
        ),
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help=(
            "The output path for xcp_d. "
            "This should not include the 'xcp_d' folder. "
            "For example, '/path/to/dset/derivatives'."
        ),
    )
    parser.add_argument(
        "analysis_level",
        action="store",
        choices=["participant"],
        help="The analysis level for xcp_d. Must be specified as 'participant'.",
    )

    # optional arguments
    parser.add_argument("--version", action="version", version=verstr)

    g_bids = parser.add_argument_group("Options for filtering BIDS queries")
    g_bids.add_argument(
        "--participant_label",
        "--participant-label",
        action="store",
        nargs="+",
        help=(
            "A space-delimited list of participant identifiers, or a single identifier. "
            "The 'sub-' prefix can be removed."
        ),
    )
    g_bids.add_argument(
        "-t",
        "--task-id",
        "--task_id",
        action="store",
        help=(
            "The name of a specific task to postprocess. "
            "By default, all tasks will be postprocessed. "
            "If you want to select more than one task to postprocess (but not all of them), "
            "you can either run XCP-D with the --task-id parameter, separately for each task, "
            "or you can use the --bids-filter-file to specify the tasks to postprocess."
        ),
    )
    g_bids.add_argument(
        "--bids-filter-file",
        dest="bids_filters",
        action="store",
        type=json_file,
        default=None,
        metavar="FILE",
        help="A JSON file defining BIDS input filters using PyBIDS.",
    )
    g_bids.add_argument(
        "-m",
        "--combineruns",
        action="store_true",
        default=False,
        help="After denoising, concatenate each derivative from each task across runs.",
    )

    g_surfx = parser.add_argument_group("Options for cifti processing")
    g_surfx.add_argument(
        "-s",
        "--cifti",
        action="store_true",
        default=False,
        help=(
            "Postprocess CIFTI inputs instead of NIfTIs. "
            "A preprocessing pipeline with CIFTI derivatives is required for this flag to work. "
            "This flag is enabled by default for the 'hcp' and 'dcan' input types."
        ),
    )

    g_perfm = parser.add_argument_group("Options for resource management")
    g_perfm.add_argument(
        "--nthreads",
        action="store",
        type=int,
        default=2,
        help="Maximum number of threads across all processes.",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        "--omp_nthreads",
        action="store",
        type=int,
        default=1,
        help="Maximum number of threads per process.",
    )
    g_perfm.add_argument(
        "--mem_gb",
        "--mem-gb",
        action="store",
        type=int,
        help="Upper bound memory limit for xcp_d processes.",
    )
    g_perfm.add_argument(
        "--use-plugin",
        "--use_plugin",
        action="store",
        default=None,
        help=(
            "Nipype plugin configuration file. "
            "For more information, see https://nipype.readthedocs.io/en/0.11.0/users/plugins.html."
        ),
    )
    g_perfm.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="Increases log verbosity for each occurence. Debug level is '-vvv'.",
    )

    g_outputoption = parser.add_argument_group("Input flags")
    g_outputoption.add_argument(
        "--input-type",
        "--input_type",
        required=False,
        default="fmriprep",
        choices=["fmriprep", "dcan", "hcp", "nibabies"],
        help=(
            "The pipeline used to generate the preprocessed derivatives. "
            "The default pipeline is 'fmriprep'. "
            "The 'dcan', 'hcp', and 'nibabies' pipelines are also supported. "
            "'nibabies' assumes the same structure as 'fmriprep'."
        ),
    )

    g_param = parser.add_argument_group("Postprocessing parameters")
    g_param.add_argument(
        "--smoothing",
        default=6,
        action="store",
        type=float,
        help=(
            "FWHM, in millimeters, of the Gaussian smoothing kernel to apply to the denoised BOLD "
            "data. "
            "This may be set to 0."
        ),
    )
    g_param.add_argument(
        "--despike",
        action="store_true",
        default=False,
        help="Despike the BOLD data before postprocessing.",
    )
    g_param.add_argument(
        "-p",
        "--nuisance-regressors",
        "--nuisance_regressors",
        dest="nuisance_regressors",
        required=False,
        choices=[
            "27P",
            "36P",
            "24P",
            "acompcor",
            "aroma",
            "acompcor_gsr",
            "aroma_gsr",
            "custom",
            "none",
        ],
        default="36P",
        type=str,
        help=(
            "Nuisance parameters to be selected. "
            "Descriptions of each of the options are included in xcp_d's documentation."
        ),
    )
    g_param.add_argument(
        "-c",
        "--custom_confounds",
        "--custom-confounds",
        required=False,
        default=None,
        type=Path,
        help=(
            "Custom confounds to be added to the nuisance regressors. "
            "Must be a folder containing confounds files, "
            "in which the file with the name matching the preprocessing confounds file will be "
            "selected."
        ),
    )
    g_param.add_argument(
        "--min_coverage",
        "--min-coverage",
        required=False,
        default=0.5,
        type=_restricted_float,
        help=(
            "Coverage threshold to apply to parcels in each atlas. "
            "Any parcels with lower coverage than the threshold will be replaced with NaNs. "
            "Must be a value between zero and one, indicating proportion of the parcel. "
            "Default is 0.5."
        ),
    )
    g_param.add_argument(
        "--min_time",
        "--min-time",
        required=False,
        default=100,
        type=float,
        help=(
            "Post-scrubbing threshold to apply to individual runs in the dataset. "
            "This threshold determines the minimum amount of time, in seconds, "
            "needed to post-process a given run, once high-motion outlier volumes are removed. "
            "This will have no impact if scrubbing is disabled "
            "(i.e., if the FD threshold is zero or negative). "
            "This parameter can be disabled by providing a zero or a negative value."
        ),
    )
    g_param.add_argument(
        "--dummy-scans",
        "--dummy_scans",
        dest="dummy_scans",
        default=0,
        type=_int_or_auto,
        metavar="{{auto,INT}}",
        help=(
            "Number of volumes to remove from the beginning of each run. "
            "If set to 'auto', xcp_d will extract non-steady-state volume indices from the "
            "preprocessing derivatives' confounds file."
        ),
    )

    g_param.add_argument(
        "--random-seed",
        "--random_seed",
        dest="random_seed",
        default=None,
        type=int,
        metavar="_RANDOM_SEED",
        help="Initialize the random seed for the workflow.",
    )

    g_filter = parser.add_argument_group("Filtering parameters")

    g_filter.add_argument(
        "--disable-bandpass-filter",
        "--disable_bandpass_filter",
        dest="bandpass_filter",
        action="store_false",
        help=(
            "Disable bandpass filtering. "
            "If bandpass filtering is disabled, then ALFF derivatives will not be calculated."
        ),
    )
    g_filter.add_argument(
        "--lower-bpf",
        "--lower_bpf",
        action="store",
        default=0.01,
        type=float,
        help=(
            "Lower cut-off frequency (Hz) for the Butterworth bandpass filter to be applied to "
            "the denoised BOLD data. Set to 0.0 or negative to disable high-pass filtering. "
            "See Satterthwaite et al. (2013)."
        ),
    )
    g_filter.add_argument(
        "--upper-bpf",
        "--upper_bpf",
        action="store",
        default=0.08,
        type=float,
        help=(
            "Upper cut-off frequency (Hz) for the Butterworth bandpass filter to be applied to "
            "the denoised BOLD data. Set to 0.0 or negative to disable low-pass filtering. "
            "See Satterthwaite et al. (2013)."
        ),
    )
    g_filter.add_argument(
        "--bpf-order",
        "--bpf_order",
        action="store",
        default=2,
        type=int,
        help="Number of filter coefficients for the Butterworth bandpass filter.",
    )
    g_filter.add_argument(
        "--motion-filter-type",
        "--motion_filter_type",
        action="store",
        type=str,
        default=None,
        choices=["lp", "notch"],
        help="""\
Type of filter to use for removing respiratory artifact from motion regressors.
If not set, no filter will be applied.

If the filter type is set to "notch", then both ``band-stop-min`` and ``band-stop-max``
must be defined.
If the filter type is set to "lp", then only ``band-stop-min`` must be defined.
""",
    )
    g_filter.add_argument(
        "--band-stop-min",
        "--band_stop_min",
        default=None,
        type=float,
        metavar="BPM",
        help="""\
Lower frequency for the motion parameter filter, in breaths-per-minute (bpm).
Motion filtering is only performed if ``motion-filter-type`` is not None.
If used with the "lp" ``motion-filter-type``, this parameter essentially corresponds to a
low-pass filter (the maximum allowed frequency in the filtered data).
This parameter is used in conjunction with ``motion-filter-order`` and ``band-stop-max``.

When ``motion-filter-type`` is set to "lp" (low-pass filter), another commonly-used value for
this parameter is 6 BPM (equivalent to 0.1 Hertz), based on Gratton et al. (2020).
""",
    )
    g_filter.add_argument(
        "--band-stop-max",
        "--band_stop_max",
        default=None,
        type=float,
        metavar="BPM",
        help="""\
Upper frequency for the band-stop motion filter, in breaths-per-minute (bpm).
Motion filtering is only performed if ``motion-filter-type`` is not None.
This parameter is only used if ``motion-filter-type`` is set to "notch".
This parameter is used in conjunction with ``motion-filter-order`` and ``band-stop-min``.
""",
    )
    g_filter.add_argument(
        "--motion-filter-order",
        "--motion_filter_order",
        default=4,
        type=int,
        help="Number of filter coeffecients for the motion parameter filter.",
    )

    g_censor = parser.add_argument_group("Censoring and scrubbing options")
    g_censor.add_argument(
        "-r",
        "--head_radius",
        "--head-radius",
        default=50,
        type=_float_or_auto,
        help=(
            "Head radius used to calculate framewise displacement, in mm. "
            "The default value is 50 mm, which is recommended for adults. "
            "For infants, we recommend a value of 35 mm. "
            "A value of 'auto' is also supported, in which case the brain radius is "
            "estimated from the preprocessed brain mask by treating the mask as a sphere."
        ),
    )
    g_censor.add_argument(
        "-f",
        "--fd-thresh",
        "--fd_thresh",
        default=0.3,
        type=float,
        help=(
            "Framewise displacement threshold for censoring. "
            "Any volumes with an FD value greater than the threshold will be removed from the "
            "denoised BOLD data. "
            "A threshold of <=0 will disable censoring completely."
        ),
    )
    g_censor.add_argument(
        "--exact-time",
        "--exact_time",
        required=False,
        default=[],
        nargs="+",
        type=float,
        help=(
            "If used, this parameter will produce correlation matrices limited to each requested "
            "amount of time. "
            "If there is more than the required amount of low-motion data, "
            "then volumes will be randomly selected to produce denoised outputs with the exact "
            "amounts of time requested. "
            "If there is less than the required amount of 'good' data, "
            "then the corresponding correlation matrix will not be produced."
        ),
    )

    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "-w",
        "--work_dir",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("working_dir"),
        help="Path to working directory, where intermediate results should be stored.",
    )
    g_other.add_argument(
        "--clean-workdir",
        "--clean_workdir",
        action="store_true",
        default=False,
        help=(
            "Clears working directory of contents. "
            "Use of this flag is not recommended when running concurrent processes of xcp_d."
        ),
    )
    g_other.add_argument(
        "--resource-monitor",
        "--resource_monitor",
        action="store_true",
        default=False,
        help="Enable Nipype's resource monitoring to keep track of memory and CPU usage.",
    )
    g_other.add_argument(
        "--notrack",
        action="store_true",
        default=False,
        help="Opt out of sending tracking information.",
    )
    g_other.add_argument(
        "--fs-license-file",
        metavar="FILE",
        type=Path,
        help=(
            "Path to FreeSurfer license key file. Get it (for free) by registering "
            "at https://surfer.nmr.mgh.harvard.edu/registration.html."
        ),
    )

    g_experimental = parser.add_argument_group("Experimental options")
    g_experimental.add_argument(
        "--warp-surfaces-native2std",
        "--warp_surfaces_native2std",
        action="store_true",
        dest="process_surfaces",
        default=False,
        help="""\
If used, a workflow will be run to warp native-space (``fsnative``) reconstructed cortical
surfaces (``surf.gii`` files) produced by Freesurfer into standard (``fsLR``) space.
These surface files are primarily used for visual quality assessment.
By default, this workflow is disabled.

**IMPORTANT**: This parameter can only be run if the --cifti flag is also enabled.
""",
    )
    g_experimental.add_argument(
        "--dcan-qc",
        "--dcan_qc",
        action="store_true",
        dest="dcan_qc",
        default=False,
        help="Run DCAN QC.",
    )

    latest = check_latest()
    if latest is not None and currentv < latest:
        print(
            f"""\
You are using aslprep-{currentv}, and a newer version of aslprep is available: {latest}.
Please check out our documentation about how and when to upgrade:
https://aslprep.readthedocs.io/en/latest/faq.html#upgrading""",
            file=sys.stderr,
        )

    _blist = is_flagged()
    if _blist[0]:
        _reason = _blist[1] or "unknown"
        print(
            f"""\
WARNING: Version {config.environment.version} of aslprep (current) has been FLAGGED
(reason: {_reason}).
That means some severe flaw was found in it and we strongly
discourage its usage.""",
            file=sys.stderr,
        )

    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    import logging

    parser = get_parser()
    opts = parser.parse_args(args, namespace)
    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    config.from_dict(vars(opts))

    # Retrieve logging level
    build_log = config.loggers.cli

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        import yaml

        with open(opts.use_plugin) as f:
            plugin_settings = yaml.load(f, Loader=yaml.FullLoader)

        _plugin = plugin_settings.get("plugin")
        if _plugin:
            config.nipype.plugin = _plugin
            config.nipype.plugin_args = plugin_settings.get("plugin_args", {})
            config.nipype.nprocs = opts.nprocs or config.nipype.plugin_args.get(
                "n_procs", config.nipype.nprocs
            )

    # Resource management options
    # Note that we're making strong assumptions about valid plugin args
    # This may need to be revisited if people try to use batch plugins
    if 1 < config.nipype.nprocs < config.nipype.omp_nthreads:
        build_log.warning(
            f"Per-process threads (--omp-nthreads={config.nipype.omp_nthreads}) exceed "
            f"total threads (--nthreads/--n_cpus={config.nipype.nprocs})"
        )

    fmri_dir = config.execution.fmri_dir
    output_dir = config.execution.output_dir
    work_dir = config.execution.work_dir

    # Wipe out existing work_dir
    if opts.clean_workdir and work_dir.exists():
        from niworkflows.utils.misc import clean_directory

        build_log.info(f"Clearing previous aslprep working directory: {work_dir}")
        if not clean_directory(work_dir):
            build_log.warning(f"Could not clear all contents of working directory: {work_dir}")

    # Ensure input and output folders are not the same
    if output_dir == fmri_dir:
        parser.error("The selected output folder is the same as the input preprocessing folder.")

    if fmri_dir in work_dir.parents:
        parser.error(
            "The selected working directory is a subdirectory of the input preprocessing folder. "
            "Please modify the output path."
        )

    # Setup directories
    config.execution.log_dir = output_dir / "xcp_d" / "logs"
    # Check and create output and working directories
    config.execution.log_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Force initialization of the BIDSLayout
    config.execution.init()
    all_subjects = config.execution.layout.get_subjects()
    if config.execution.participant_label is None:
        config.execution.participant_label = all_subjects

    participant_label = set(config.execution.participant_label)
    missing_subjects = participant_label - set(all_subjects)
    if missing_subjects:
        parser.error(
            "One or more participant labels were not found in the BIDS directory: "
            f"{', '.join(missing_subjects)}."
        )

    config.execution.participant_label = sorted(participant_label)
