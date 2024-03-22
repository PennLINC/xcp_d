#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The xcp_d preprocessing worklow.

xcp_d preprocessing workflow
============================
"""
import os
import sys

from xcp_d import config


def _build_parser():
    """Build parser object."""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from functools import partial
    from pathlib import Path

    from packaging.version import Version

    from xcp_d.cli.parser_utils import _float_or_auto, _int_or_auto, _restricted_float
    from xcp_d.cli.version import check_latest, is_flagged
    from xcp_d.utils.atlas import select_atlases

    def _path_exists(path, parser):
        """Ensure a given path exists."""
        if path is None or not Path(path).exists():
            raise parser.error(f"Path does not exist: <{path}>.")
        return Path(path).absolute()

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = _path_exists(path, parser)
        if not path.is_file():
            raise parser.error(f"Path should point to a file (or symlink of file): <{path}>.")
        return path

    def _process_value(value):
        import bids

        if value is None:
            return bids.layout.Query.NONE
        elif value == "*":
            return bids.layout.Query.ANY
        else:
            return value

    def _filter_pybids_none_any(dct):
        d = {}
        for k, v in dct.items():
            if isinstance(v, list):
                d[k] = [_process_value(val) for val in v]
            else:
                d[k] = _process_value(v)
        return d

    def _bids_filter(value, parser):
        from json import JSONDecodeError, loads

        if value:
            if Path(value).exists():
                try:
                    return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
                except JSONDecodeError:
                    raise parser.error(f"JSON syntax error in: <{value}>.")
            else:
                raise parser.error(f"Path does not exist: <{value}>.")

    verstr = f"XCP-D v{config.environment.version}"
    currentv = Version(config.environment.version)
    is_release = not any((currentv.is_devrelease, currentv.is_prerelease, currentv.is_postrelease))

    parser = ArgumentParser(
        description=f"XCP-D: Postprocessing Workflow of fMRI Data v{config.environment.version}",
        epilog="See https://xcp-d.readthedocs.io/en/latest/workflows.html",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)
    BIDSFilter = partial(_bids_filter, parser=parser)

    # important parameters required
    parser.add_argument(
        "fmri_dir",
        action="store",
        type=PathExists,
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
            "The output path for XCP-D derivatives. "
            "For example, '/path/to/dset/derivatives/xcp_d'. "
            "As of version 0.7.0, 'xcp_d' will not be appended to the output directory."
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
        "--participant-label",
        "--participant_label",
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
        "--bids_filter_file",
        dest="bids_filters",
        action="store",
        type=BIDSFilter,
        default=None,
        metavar="FILE",
        help=(
            "A JSON file describing custom BIDS input filters using PyBIDS. "
            "For further details, please check out "
            "https://xcp_d.readthedocs.io/en/"
            f"{currentv.base_version if is_release else 'latest'}/usage.html#"
            "filtering-inputs-with-bids-filter-files"
        ),
    )

    g_surfx = parser.add_argument_group("Options for CIFTI processing")
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
        "--nprocs",
        "--nthreads",
        "--n-cpus",
        "--n_cpus",
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
        "--mem-gb",
        "--mem_gb",
        action="store",
        type=int,
        help="Upper bound memory limit, in gigabytes, for XCP-D processes.",
    )
    g_perfm.add_argument(
        "--low-mem",
        action="store_true",
        help="Attempt to reduce memory usage (will increase disk usage in working directory).",
    )
    g_perfm.add_argument(
        "--use-plugin",
        "--use_plugin",
        "--nipype-plugin-file",
        "--nipype_plugin_file",
        action="store",
        default=None,
        type=IsFile,
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
        choices=["fmriprep", "dcan", "hcp", "nibabies", "ukb"],
        help=(
            "The pipeline used to generate the preprocessed derivatives. "
            "The default pipeline is 'fmriprep'. "
            "The 'dcan', 'hcp', 'nibabies', and 'ukb' pipelines are also supported. "
            "'nibabies' assumes the same structure as 'fmriprep'."
        ),
    )

    g_param = parser.add_argument_group("Postprocessing parameters")
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
        "--despike",
        action="store_true",
        default=False,
        help="Despike the BOLD data before postprocessing.",
    )
    g_param.add_argument(
        "-p",
        "--nuisance-regressors",
        "--nuisance_regressors",
        dest="params",
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
            # GSR-only for UKB
            "gsr_only",
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
        "--custom-confounds",
        "--custom_confounds",
        required=False,
        default=None,
        type=PathExists,
        help=(
            "Custom confounds to be added to the nuisance regressors. "
            "Must be a folder containing confounds files, "
            "in which the file with the name matching the preprocessing confounds file will be "
            "selected."
        ),
    )
    g_param.add_argument(
        "--smoothing",
        default=6,
        action="store",
        type=float,
        help=(
            "FWHM, in millimeters, of the Gaussian smoothing kernel to apply to the denoised BOLD "
            "data. "
            "Set to 0 to disable smoothing."
        ),
    )
    g_param.add_argument(
        "-m",
        "--combineruns",
        action="store_true",
        default=False,
        help="After denoising, concatenate each derivative from each task across runs.",
    )

    g_motion_filter = parser.add_argument_group(
        title="Motion filtering parameters",
        description=(
            "These parameters enable and control a filter that will be applied to motion "
            "parameters. "
            "Motion parameters may be contaminated by non-motion noise, and applying a filter "
            "may reduce the impact of that contamination."
        ),
    )
    g_motion_filter.add_argument(
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
    g_motion_filter.add_argument(
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
    g_motion_filter.add_argument(
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
    g_motion_filter.add_argument(
        "--motion-filter-order",
        "--motion_filter_order",
        default=4,
        type=int,
        help="Number of filter coeffecients for the motion parameter filter.",
    )

    g_censor = parser.add_argument_group("Censoring and scrubbing options")
    g_censor.add_argument(
        "-r",
        "--head-radius",
        "--head_radius",
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
        "--min-time",
        "--min_time",
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

    g_temporal_filter = parser.add_argument_group(
        title="Data filtering parameters",
        description=(
            "These parameters determine whether a bandpass filter will be applied to the BOLD "
            "data, after the censoring, denoising, and interpolation steps of the pipeline, "
            "but before recensoring."
        ),
    )
    g_temporal_filter.add_argument(
        "--disable-bandpass-filter",
        "--disable_bandpass_filter",
        dest="bandpass_filter",
        action="store_false",
        help=(
            "Disable bandpass filtering. "
            "If bandpass filtering is disabled, then ALFF derivatives will not be calculated."
        ),
    )
    g_temporal_filter.add_argument(
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
    g_temporal_filter.add_argument(
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
    g_temporal_filter.add_argument(
        "--bpf-order",
        "--bpf_order",
        action="store",
        default=2,
        type=int,
        help="Number of filter coefficients for the Butterworth bandpass filter.",
    )

    g_parcellation = parser.add_argument_group("Parcellation options")

    g_atlases = g_parcellation.add_mutually_exclusive_group(required=False)
    all_atlases = select_atlases(atlases=None, subset="all")
    g_atlases.add_argument(
        "--atlases",
        action="store",
        nargs="+",
        choices=all_atlases,
        default=all_atlases,
        dest="atlases",
        help="Selection of atlases to apply to the data. All are used by default.",
    )
    g_atlases.add_argument(
        "--skip-parcellation",
        "--skip_parcellation",
        action="store_const",
        const=[],
        dest="atlases",
        help="Skip parcellation and correlation steps.",
    )

    g_parcellation.add_argument(
        "--min-coverage",
        "--min_coverage",
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
    g_parcellation.add_argument(
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
        "--random-seed",
        "--random_seed",
        dest="random_seed",
        default=None,
        type=int,
        metavar="_RANDOM_SEED",
        help="Initialize the random seed for the workflow.",
    )
    g_other.add_argument(
        "-w",
        "--work-dir",
        "--work_dir",
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
        "--config-file",
        "--config_file",
        action="store",
        metavar="FILE",
        help=(
            "Use pre-generated configuration file. "
            "Values in file will be overridden by command-line arguments."
        ),
    )
    g_other.add_argument(
        "--write-graph",
        action="store_true",
        default=False,
        help="Write workflow graph.",
    )
    g_other.add_argument(
        "--stop-on-first-crash",
        action="store_true",
        default=False,
        help="Force stopping on first crash, even if a work directory was specified.",
    )
    g_other.add_argument(
        "--notrack",
        action="store_true",
        default=False,
        help="Opt out of sending tracking information.",
    )
    g_other.add_argument(
        "--debug",
        action="store",
        nargs="+",
        choices=config.DEBUG_MODES + ("all",),
        help="Debug mode(s) to enable. 'all' is alias for all available modes.",
    )
    g_other.add_argument(
        "--fs-license-file",
        metavar="FILE",
        type=PathExists,
        help=(
            "Path to FreeSurfer license key file. Get it (for free) by registering "
            "at https://surfer.nmr.mgh.harvard.edu/registration.html."
        ),
    )
    g_other.add_argument(
        "--md-only-boilerplate",
        action="store_true",
        default=False,
        help="Skip generation of HTML and LaTeX formatted citation with pandoc",
    )
    g_other.add_argument(
        "--boilerplate-only",
        "--boilerplate_only",
        action="store_true",
        default=False,
        help="generate boilerplate only",
    )
    g_other.add_argument(
        "--reports-only",
        action="store_true",
        default=False,
        help=(
            "only generate reports, don't run workflows. This will only rerun report "
            "aggregation, not reportlet generation for specific nodes."
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
        "--skip-dcan-qc",
        "--skip_dcan_qc",
        action="store_false",
        dest="dcan_qc",
        default=True,
        help="Do not run DCAN QC.",
    )

    latest = check_latest()
    if latest is not None and currentv < latest:
        print(
            f"""\
You are using XCP-D v{currentv}, and a newer version of XCP-D is available: v{latest}.
Please check out our documentation about how and when to upgrade:
https://xcp_d.readthedocs.io/en/latest/faq.html#upgrading""",
            file=sys.stderr,
        )

    _blist = is_flagged()
    if _blist[0]:
        _reason = _blist[1] or "unknown"
        print(
            f"""\
WARNING: Version {config.environment.version} of XCP-D (current) has been FLAGGED
(reason: {_reason}).
That means some severe flaw was found in it and we strongly
discourage its usage.""",
            file=sys.stderr,
        )

    return parser


def parse_args(args=None, namespace=None):
    """Parse args and run further checks on the command line."""
    import logging

    parser = _build_parser()
    opts = parser.parse_args(args, namespace)
    if opts.config_file:
        skip = {} if opts.reports_only else {"execution": ("run_uuid",)}
        config.load(opts.config_file, skip=skip, init=False)
        config.loggers.cli.info(f"Loaded previous configuration file {opts.config_file}")

    opts = _validate_parameters(opts=opts, build_log=config.loggers.cli, parser=parser)

    # Wipe out existing work_dir
    if opts.clean_workdir and opts.work_dir.exists():
        from niworkflows.utils.misc import clean_directory

        config.loggers.cli.info(f"Clearing previous XCP-D working directory: {opts.work_dir}")
        if not clean_directory(opts.work_dir):
            config.loggers.cli.warning(
                f"Could not clear all contents of working directory: {opts.work_dir}"
            )

    # First check that fmriprep_dir looks like a BIDS folder
    if opts.input_type in ("dcan", "hcp", "ukb"):
        if opts.input_type == "dcan":
            from xcp_d.ingression.abcdbids import convert_dcan2bids as convert_to_bids
        elif opts.input_type == "hcp":
            from xcp_d.ingression.hcpya import convert_hcp2bids as convert_to_bids
        elif opts.input_type == "ukb":
            from xcp_d.ingression.ukbiobank import convert_ukb2bids as convert_to_bids

        converted_fmri_dir = opts.work_dir / f"dset_bids/derivatives/{opts.input_type}"
        converted_fmri_dir.mkdir(exist_ok=True, parents=True)

        convert_to_bids(
            opts.fmri_dir,
            out_dir=str(converted_fmri_dir),
            participant_ids=opts.participant_label,
        )

        opts.fmri_dir = converted_fmri_dir
        assert converted_fmri_dir.exists(), f"Conversion to BIDS failed: {converted_fmri_dir}"

    if not os.path.isfile(os.path.join(opts.fmri_dir, "dataset_description.json")):
        config.loggers.cli.error(
            "No dataset_description.json file found in input directory. "
            "Make sure to point to the specific pipeline's derivatives folder. "
            "For example, use '/dset/derivatives/fmriprep', not /dset/derivatives'."
        )

    config.execution.log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    config.from_dict(vars(opts), init=["nipype"])
    assert config.execution.fmri_dir.exists(), (
        f"Conversion to BIDS failed: {config.execution.fmri_dir}",
    )

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
    version = config.environment.version

    if config.execution.xcp_d_dir is None:
        config.execution.xcp_d_dir = output_dir

    # Update the config with an empty dict to trigger initialization of all config
    # sections (we used `init=False` above).
    # This must be done after cleaning the work directory, or we could delete an
    # open SQLite database
    config.from_dict({})

    # Ensure input and output folders are not the same
    if output_dir == fmri_dir:
        rec_path = fmri_dir / "derivatives" / f"xcp_d-{version.split('+')[0]}"
        parser.error(
            "The selected output folder is the same as the input BIDS folder. "
            f"Please modify the output path (suggestion: {rec_path})."
        )

    if fmri_dir in work_dir.parents:
        parser.error(
            "The selected working directory is a subdirectory of the input BIDS folder. "
            "Please modify the output path."
        )

    # Setup directories
    config.execution.log_dir = config.execution.xcp_d_dir / "logs"
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


def _validate_parameters(opts, build_log, parser):
    """Validate parameters.

    This function was abstracted out of build_workflow to make testing easier.
    """
    import os
    from pathlib import Path

    opts.fmri_dir = opts.fmri_dir.resolve()
    opts.output_dir = opts.output_dir.resolve()
    opts.work_dir = opts.work_dir.resolve()

    # Set the FreeSurfer license
    if opts.fs_license_file is not None:
        opts.fs_license_file = opts.fs_license_file.resolve()
        if opts.fs_license_file.is_file():
            os.environ["FS_LICENSE"] = str(opts.fs_license_file)

        else:
            parser.error(f"Freesurfer license DNE: {opts.fs_license_file}.")
    else:
        fs_license_file = os.environ.get("FS_LICENSE", "/opt/freesurfer/license.txt")
        if not Path(fs_license_file).is_file():
            parser.error(
                "A valid FreeSurfer license file is required. "
                "Set the FS_LICENSE environment variable or use the '--fs-license-file' flag."
            )

        os.environ["FS_LICENSE"] = str(fs_license_file)

    # Resolve custom confounds folder
    if opts.custom_confounds:
        opts.custom_confounds = str(opts.custom_confounds.resolve())

    # Bandpass filter parameters
    if opts.lower_bpf <= 0 and opts.upper_bpf <= 0:
        opts.bandpass_filter = False

    if (
        opts.bandpass_filter
        and (opts.lower_bpf >= opts.upper_bpf)
        and (opts.lower_bpf > 0 and opts.upper_bpf > 0)
    ):
        parser.error(
            f"'--lower-bpf' ({opts.lower_bpf}) must be lower than "
            f"'--upper-bpf' ({opts.upper_bpf})."
        )
    elif not opts.bandpass_filter:
        build_log.warning("Bandpass filtering is disabled. ALFF outputs will not be generated.")

    # Scrubbing parameters
    if opts.fd_thresh <= 0 and opts.min_time > 0:
        ignored_params = "\n\t".join(["--min-time"])
        build_log.warning(
            "Framewise displacement-based scrubbing is disabled. "
            f"The following parameters will have no effect:\n\t{ignored_params}"
        )
        opts.min_time = 0

    # Motion filtering parameters
    if opts.motion_filter_type == "notch":
        if not (opts.band_stop_min and opts.band_stop_max):
            parser.error(
                "Please set both '--band-stop-min' and '--band-stop-max' if you want to apply "
                "the 'notch' motion filter."
            )
        elif opts.band_stop_min >= opts.band_stop_max:
            parser.error(
                f"'--band-stop-min' ({opts.band_stop_min}) must be lower than "
                f"'--band-stop-max' ({opts.band_stop_max})."
            )
        elif opts.band_stop_min < 1 or opts.band_stop_max < 1:
            build_log.warning(
                f"Either '--band-stop-min' ({opts.band_stop_min}) or "
                f"'--band-stop-max' ({opts.band_stop_max}) is suspiciously low. "
                "Please remember that these values should be in breaths-per-minute."
            )

    elif opts.motion_filter_type == "lp":
        if not opts.band_stop_min:
            parser.error(
                "Please set '--band-stop-min' if you want to apply the 'lp' motion filter."
            )
        elif opts.band_stop_min < 1:
            build_log.warning(
                f"'--band-stop-min' ({opts.band_stop_max}) is suspiciously low. "
                "Please remember that this value should be in breaths-per-minute."
            )

        if opts.band_stop_max:
            build_log.warning("'--band-stop-max' is ignored when '--motion-filter-type' is 'lp'.")

    elif opts.band_stop_min or opts.band_stop_max:
        build_log.warning(
            "'--band-stop-min' and '--band-stop-max' are ignored if '--motion-filter-type' "
            "is not set."
        )

    # Parcellation parameters
    if not opts.atlases and opts.min_coverage != 0.5:
        build_log.warning(
            "When no atlases are selected or parcellation is explicitly skipped "
            "('--skip-parcellation'), '--min-coverage' will have no effect."
        )

    # Some parameters are automatically set depending on the input type.
    if opts.input_type in ("dcan", "hcp"):
        if not opts.cifti:
            build_log.warning(
                f"With input_type {opts.input_type}, cifti processing (--cifti) will be "
                "enabled automatically."
            )
            opts.cifti = True

        if not opts.process_surfaces:
            build_log.warning(
                f"With input_type {opts.input_type}, surface normalization "
                "(--warp-surfaces-native2std) will be enabled automatically."
            )
            opts.process_surfaces = True

    elif opts.input_type == "ukb":
        if opts.cifti:
            build_log.warning(
                f"With input_type {opts.input_type}, cifti processing (--cifti) will be "
                "disabled automatically."
            )
            opts.cifti = False

        if opts.process_surfaces:
            build_log.warning(
                f"With input_type {opts.input_type}, surface normalization "
                "(--warp-surfaces-native2std) will be disabled automatically."
            )
            opts.process_surfaces = False

    # process_surfaces and nifti processing are incompatible.
    if opts.process_surfaces and not opts.cifti:
        parser.error(
            "In order to perform surface normalization (--warp-surfaces-native2std), "
            "you must enable cifti processing (--cifti)."
        )

    return opts
