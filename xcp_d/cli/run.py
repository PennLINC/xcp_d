#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The xcp_d preprocessing worklow.

xcp_d preprocessing workflow
============================
"""
import gc
import logging
import os
import sys
import uuid
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from time import strftime

from niworkflows import NIWORKFLOWS_LOG

from xcp_d.cli.parser_utils import (
    _float_or_auto,
    _int_or_auto,
    _restricted_float,
    check_deps,
    json_file,
)

warnings.filterwarnings("ignore")

logging.addLevelName(25, "IMPORTANT")  # Add a new level between INFO and WARNING
logging.addLevelName(15, "VERBOSE")  # Add a new level between INFO and DEBUG
logger = logging.getLogger("cli")


def get_parser():
    """Build parser object."""
    from xcp_d.__about__ import __version__

    verstr = f"xcp_d v{__version__}"

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
        action="store",
        help=(
            "The name of a specific task to postprocess. "
            "By default, all tasks will be postprocessed."
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
        action="store",
        type=int,
        default=1,
        help="Maximum number of threads per process.",
    )
    g_perfm.add_argument(
        "--mem_gb",
        "--mem_gb",
        action="store",
        type=int,
        help="Upper bound memory limit for xcp_d processes.",
    )
    g_perfm.add_argument(
        "--use-plugin",
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
        action="store",
        default=0.01,
        type=float,
        help=(
            "Lower cut-off frequency (Hz) for the Butterworth bandpass filter to be applied to "
            "the denoised BOLD data. "
            "See Satterthwaite et al. (2013)."
        ),
    )
    g_filter.add_argument(
        "--upper-bpf",
        action="store",
        default=0.08,
        type=float,
        help=(
            "Upper cut-off frequency (Hz) for the Butterworth bandpass filter to be applied to "
            "the denoised BOLD data. "
            "See Satterthwaite et al. (2013)."
        ),
    )
    g_filter.add_argument(
        "--bpf-order",
        action="store",
        default=2,
        type=int,
        help="Number of filter coefficients for the Butterworth bandpass filter.",
    )
    g_filter.add_argument(
        "--motion-filter-type",
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
        default=4,
        type=int,
        help="Number of filter coeffecients for the motion parameter filter.",
    )

    g_censor = parser.add_argument_group("Censoring and scrubbing options")
    g_censor.add_argument(
        "-r",
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
        default=0.2,
        type=float,
        help=(
            "Framewise displacement threshold for censoring. "
            "Any volumes with an FD value greater than the threshold will be removed from the "
            "denoised BOLD data. "
            "A threshold of <=0 will disable censoring completely."
        ),
    )

    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "-w",
        "--work_dir",
        action="store",
        type=Path,
        default=Path("working_dir"),
        help="Path to working directory, where intermediate results should be stored.",
    )
    g_other.add_argument(
        "--clean-workdir",
        action="store_true",
        default=False,
        help=(
            "Clears working directory of contents. "
            "Use of this flag is not recommended when running concurrent processes of xcp_d."
        ),
    )
    g_other.add_argument(
        "--resource-monitor",
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

    g_experimental = parser.add_argument_group("Experimental options")
    g_experimental.add_argument(
        "--warp-surfaces-native2std",
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
        help="Run DCAN QC, including executive summary generation.",
    )

    return parser


def _main(args=None):
    from multiprocessing import set_start_method

    set_start_method("forkserver")

    main(args=args)


def main(args=None):
    """Run the main workflow."""
    from multiprocessing import Manager, Process

    opts = get_parser().parse_args(args)

    exec_env = os.name

    sentry_sdk = None
    if not opts.notrack:
        import sentry_sdk

        from xcp_d.utils.sentry import sentry_setup

        sentry_setup(opts, exec_env)

    # Retrieve and set logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    logger.setLevel(log_level)

    # Call build_workflow(opts, retval)
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(opts, retval))
        p.start()
        p.join()

        retcode = p.exitcode or retval.get("return_code", 0)

        work_dir = Path(retval.get("work_dir"))
        fmri_dir = Path(retval.get("fmri_dir"))
        output_dir = Path(retval.get("output_dir"))
        plugin_settings = retval.get("plugin_settings", None)
        subject_list = retval.get("subject_list", None)
        run_uuid = retval.get("run_uuid", None)
        xcpd_wf = retval.get("workflow", None)

    retcode = retcode or int(xcpd_wf is None)
    if retcode != 0:
        sys.exit(retcode)

    # Check workflow for missing commands
    missing = check_deps(xcpd_wf)
    if missing:
        print("Cannot run xcp_d. Missing dependencies:", file=sys.stderr)
        for iface, cmd in missing:
            print(f"\t{cmd} (Interface: {iface})")
        sys.exit(2)

    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Track start of workflow with sentry
    if not opts.notrack:
        from xcp_d.utils.sentry import start_ping

        start_ping(run_uuid, len(subject_list))

    errno = 1  # Default is error exit unless otherwise set
    try:
        xcpd_wf.run(**plugin_settings)

    except Exception as e:
        if not opts.notrack:
            from xcp_d.utils.sentry import process_crashfile

            crashfolders = [
                output_dir / "xcp_d" / f"sub-{s}" / "log" / run_uuid for s in subject_list
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob("crash*.*"):
                    process_crashfile(crashfile)

        if "Workflow did not execute cleanly" not in str(e):
            sentry_sdk.capture_exception(e)

        logger.critical("xcp_d failed: %s", e)
        raise

    else:
        errno = 0
        logger.log(25, "xcp_d finished without errors")
        if not opts.notrack:
            sentry_sdk.capture_message("xcp_d finished without errors", level="info")

    finally:
        from shutil import copyfile
        from subprocess import CalledProcessError, TimeoutExpired, check_call

        from pkg_resources import resource_filename as pkgrf

        from xcp_d.interfaces.report_core import generate_reports

        citation_files = {
            ext: output_dir / "xcp_d" / "logs" / f"CITATION.{ext}"
            for ext in ("bib", "tex", "md", "html")
        }

        if citation_files["md"].exists():
            # Generate HTML file resolving citations
            cmd = [
                "pandoc",
                "-s",
                "--bibliography",
                pkgrf("xcp_d", "data/boilerplate.bib"),
                "--filter",
                "pandoc-citeproc",
                "--metadata",
                'pagetitle="xcp_d citation boilerplate"',
                str(citation_files["md"]),
                "-o",
                str(citation_files["html"]),
            ]
            logger.info("Generating an HTML version of the citation boilerplate...")
            try:
                check_call(cmd, timeout=10)
            except (FileNotFoundError, CalledProcessError, TimeoutExpired):
                logger.warning(f"Could not generate CITATION.html file:\n{' '.join(cmd)}")

            # Generate LaTex file resolving citations
            cmd = [
                "pandoc",
                "-s",
                "--bibliography",
                pkgrf("xcp_d", "data/boilerplate.bib"),
                "--natbib",
                str(citation_files["md"]),
                "-o",
                str(citation_files["tex"]),
            ]
            logger.info("Generating a LaTeX version of the citation boilerplate...")
            try:
                check_call(cmd, timeout=10)
            except (FileNotFoundError, CalledProcessError, TimeoutExpired):
                logger.warning(f"Could not generate CITATION.tex file:\n{' '.join(cmd)}")
            else:
                copyfile(pkgrf("xcp_d", "data/boilerplate.bib"), citation_files["bib"])

        else:
            logger.warning(
                "xcp_d could not find the markdown version of "
                f"the citation boilerplate ({citation_files['md']}). "
                "HTML and LaTeX versions of it will not be available"
            )

        # Generate reports phase
        failed_reports = generate_reports(
            subject_list=subject_list,
            fmri_dir=fmri_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            run_uuid=run_uuid,
            config=pkgrf("xcp_d", "data/reports.yml"),
            packagename="xcp_d",
            dcan_qc=opts.dcan_qc,
        )

        if failed_reports and not opts.notrack:
            sentry_sdk.capture_message(
                f"Report generation failed for {failed_reports} subjects", level="error"
            )
        sys.exit(int((errno + failed_reports) > 0))


def build_workflow(opts, retval):
    """Create the Nipype workflow that supports the whole execution graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows fmriprep to enforce
    a hard-limited memory-scope.
    """
    from bids import BIDSLayout
    from nipype import config as ncfg
    from nipype import logging as nlogging

    from xcp_d.__about__ import __version__
    from xcp_d.utils.bids import collect_participants
    from xcp_d.workflows.base import init_xcpd_wf

    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))

    build_log = nlogging.getLogger("nipype.workflow")
    build_log.setLevel(log_level)
    nlogging.getLogger("nipype.interface").setLevel(log_level)
    nlogging.getLogger("nipype.utils").setLevel(log_level)

    fmri_dir = opts.fmri_dir.resolve()
    output_dir = opts.output_dir.resolve()
    work_dir = opts.work_dir.resolve()

    retval["return_code"] = 0

    # Check the validity of inputs
    if output_dir == fmri_dir:
        rec_path = fmri_dir / "derivatives" / f"xcp_d-{__version__.split('+')[0]}"
        build_log.error(
            "The selected output folder is the same as the input fmri input. "
            "Please modify the output path "
            f"(suggestion: {rec_path})."
        )
        retval["return_code"] = 1

    if opts.analysis_level != "participant":
        build_log.error('Please select analysis level "participant"')
        retval["return_code"] = 1

    # Bandpass filter parameters
    if opts.bandpass_filter and (opts.lower_bpf >= opts.upper_bpf):
        build_log.error(
            f"'--lower-bpf' ({opts.lower_bpf}) must be lower than "
            f"'--upper-bpf' ({opts.upper_bpf})."
        )
        retval["return_code"] = 1
    elif not opts.bandpass_filter:
        build_log.warning("Bandpass filtering is disabled. ALFF outputs will not be generated.")

    # Scrubbing parameters
    if opts.fd_thresh <= 0:
        ignored_params = "\n\t".join(
            [
                "--min-time",
                "--motion-filter-type",
                "--band-stop-min",
                "--band-stop-max",
                "--motion-filter-order",
                "--head_radius",
            ]
        )
        build_log.warning(
            "Framewise displacement-based scrubbing is disabled. "
            f"The following parameters will have no effect:\n\t{ignored_params}"
        )
        opts.min_time = 0
        opts.motion_filter_type = None
        opts.band_stop_min = None
        opts.band_stop_max = None
        opts.motion_filter_order = None

    # Motion filtering parameters
    if opts.motion_filter_type == "notch":
        if not (opts.band_stop_min and opts.band_stop_max):
            build_log.error(
                "Please set both '--band-stop-min' and '--band-stop-max' if you want to apply "
                "the 'notch' motion filter."
            )
            retval["return_code"] = 1
        elif opts.band_stop_min >= opts.band_stop_max:
            build_log.error(
                f"'--band-stop-min' ({opts.band_stop_min}) must be lower than "
                f"'--band-stop-max' ({opts.band_stop_max})."
            )
            retval["return_code"] = 1
        elif opts.band_stop_min < 1 or opts.band_stop_max < 1:
            build_log.warning(
                f"Either '--band-stop-min' ({opts.band_stop_min}) or "
                f"'--band-stop-max' ({opts.band_stop_max}) is suspiciously low. "
                "Please remember that these values should be in breaths-per-minute."
            )

    elif opts.motion_filter_type == "lp":
        if not opts.band_stop_min:
            build_log.error(
                "Please set '--band-stop-min' if you want to apply the 'lp' motion filter."
            )
            retval["return_code"] = 1
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

    # process_surfaces and nifti processing are incompatible.
    if opts.process_surfaces and not opts.cifti:
        build_log.error(
            "In order to perform surface normalization (--warp-surfaces-native2std), "
            "you must enable cifti processing (--cifti)."
        )
        retval["return_code"] = 1

    if retval["return_code"] == 1:
        return retval

    if opts.clean_workdir:
        from niworkflows.utils.misc import clean_directory

        build_log.info(f"Clearing previous xcp_d working directory: {work_dir}")
        if not clean_directory(work_dir):
            build_log.warning(f"Could not clear all contents of working directory: {work_dir}")

    retval["return_code"] = 1
    retval["workflow"] = None
    retval["fmri_dir"] = str(fmri_dir)
    retval["output_dir"] = str(output_dir)
    retval["work_dir"] = str(work_dir)

    # First check that fmriprep_dir looks like a BIDS folder
    if opts.input_type in ("dcan", "hcp"):
        if opts.input_type == "dcan":
            from xcp_d.utils.dcan2fmriprep import convert_dcan2bids as convert_to_bids
        elif opts.input_type == "hcp":
            from xcp_d.utils.hcp2fmriprep import convert_hcp2bids as convert_to_bids

        NIWORKFLOWS_LOG.info(f"Converting {opts.input_type} to fmriprep format")
        print(f"checking the {opts.input_type} files")
        converted_fmri_dir = os.path.join(work_dir, f"dset_bids/derivatives/{opts.input_type}")
        os.makedirs(converted_fmri_dir, exist_ok=True)

        convert_to_bids(
            fmri_dir,
            out_dir=converted_fmri_dir,
            participant_ids=opts.participant_label,
        )

        fmri_dir = converted_fmri_dir

    if not os.path.isfile((os.path.join(fmri_dir, "dataset_description.json"))):
        build_log.error(
            "No dataset_description.json file found in input directory. "
            "Make sure to point to the specific pipeline's derivatives folder. "
            "For example, use '/dset/derivatives/fmriprep', not /dset/derivatives'."
        )
        retval["return_code"] = 1

    # Set up some instrumental utilities
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4()}"
    retval["run_uuid"] = run_uuid

    layout = BIDSLayout(str(fmri_dir), validate=False, derivatives=True)
    subject_list = collect_participants(layout, participant_label=opts.participant_label)
    retval["subject_list"] = subject_list

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml

        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)

        plugin_settings.setdefault("plugin_args", {})

    else:
        # Defaults
        plugin_settings = {
            "plugin": "MultiProc",
            "plugin_args": {
                "raise_insufficient": False,
                "maxtasksperchild": 1,
            },
        }

    # nthreads = plugin_settings['plugin_args'].get('n_procs')
    # Permit overriding plugin config with specific CLI options
    # if nthreads is None or opts.nthreads is not None:
    nthreads = opts.nthreads
    # if nthreads is None or nthreads < 1:
    # nthreads = cpu_count()
    # plugin_settings['plugin_args']['n_procs'] = nthreads

    if opts.mem_gb:
        plugin_settings["plugin_args"]["memory_gb"] = opts.mem_gb

    omp_nthreads = opts.omp_nthreads
    # if omp_nthreads == 0:
    # omp_nthreads = min(nthreads - 1 if nthreads > 1 else cpu_count(), 8)
    if (nthreads == 1) or (omp_nthreads > nthreads):
        omp_nthreads = 1

    plugin_settings["plugin_args"]["n_procs"] = nthreads

    if 1 < nthreads < omp_nthreads:
        build_log.warning(
            f"Per-process threads (--omp-nthreads={omp_nthreads}) exceed total "
            f"threads (--nthreads/--n_cpus={nthreads})"
        )

    retval["plugin_settings"] = plugin_settings

    # Set up directories
    log_dir = output_dir / "xcp_d" / "logs"

    # Check and create output and working directories
    output_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)
    work_dir.mkdir(exist_ok=True, parents=True)

    # Nipype config (logs and execution)
    ncfg.update_config(
        {
            "logging": {
                "log_directory": str(log_dir),
                "log_to_file": True,
                "workflow_level": log_level,
                "interface_level": log_level,
                "utils_level": log_level,
            },
            "execution": {
                "crashdump_dir": str(log_dir),
                "crashfile_format": "txt",
                "get_linked_libs": False,
            },
            "monitoring": {
                "enabled": opts.resource_monitor,
                "sample_frequency": "0.5",
                "summary_append": True,
            },
        }
    )

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    # Build main workflow
    build_log.log(
        25,
        f"""\
Running xcp_d version {__version__}:
    * fMRI directory path: {fmri_dir}.
    * Participant list: {subject_list}.
    * Run identifier: {run_uuid}.

""",
    )

    retval["workflow"] = init_xcpd_wf(
        layout=layout,
        omp_nthreads=omp_nthreads,
        fmri_dir=str(fmri_dir),
        high_pass=opts.lower_bpf,
        low_pass=opts.upper_bpf,
        bpf_order=opts.bpf_order,
        bandpass_filter=opts.bandpass_filter,
        motion_filter_type=opts.motion_filter_type,
        motion_filter_order=opts.motion_filter_order,
        band_stop_min=opts.band_stop_min,
        band_stop_max=opts.band_stop_max,
        subject_list=subject_list,
        work_dir=str(work_dir),
        task_id=opts.task_id,
        bids_filters=opts.bids_filters,
        despike=opts.despike,
        smoothing=opts.smoothing,
        params=opts.nuisance_regressors,
        cifti=opts.cifti,
        analysis_level=opts.analysis_level,
        output_dir=str(output_dir),
        head_radius=opts.head_radius,
        custom_confounds_folder=opts.custom_confounds,
        dummy_scans=opts.dummy_scans,
        fd_thresh=opts.fd_thresh,
        process_surfaces=opts.process_surfaces,
        dcan_qc=opts.dcan_qc,
        input_type=opts.input_type,
        min_coverage=opts.min_coverage,
        min_time=opts.min_time,
        combineruns=opts.combineruns,
        name="xcpd_wf",
    )

    logs_path = Path(output_dir) / "xcp_d" / "logs"
    boilerplate = retval["workflow"].visit_desc()

    if boilerplate:
        citation_files = {
            ext: logs_path / f"CITATION.{ext}" for ext in ("bib", "tex", "md", "html")
        }
        # To please git-annex users and also to guarantee consistency
        # among different renderings of the same file, first remove any
        # existing one
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

        citation_files["md"].write_text(boilerplate)

    build_log.log(
        25,
        (
            "Works derived from this xcp_d execution should "
            f"include the following boilerplate:\n\n{boilerplate}"
        ),
    )

    retval["return_code"] = 0

    return retval


if __name__ == "__main__":
    raise RuntimeError(
        "xcp_d/cli/run.py should not be run directly;\n"
        "Please use the `xcp_d` command-line interface."
    )
