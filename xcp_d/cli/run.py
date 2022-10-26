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
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from time import strftime

from niworkflows import NIWORKFLOWS_LOG

warnings.filterwarnings("ignore")

logging.addLevelName(25, "IMPORTANT")  # Add a new level between INFO and WARNING
logging.addLevelName(15, "VERBOSE")  # Add a new level between INFO and DEBUG
logger = logging.getLogger("cli")


def _warn_redirect(message, category):
    logger.warning("Captured warning (%s): %s", category, message)


def check_deps(workflow):
    """Check the dependencies for the workflow."""
    from nipype.utils.filemanip import which

    return sorted(
        (node.interface.__class__.__name__, node.interface._cmd)
        for node in workflow._get_all_nodes()
        if (
            hasattr(node.interface, "_cmd")
            and which(node.interface._cmd.split()[0]) is None
        )
    )


class DeprecatedStoreAction(Action):
    """A custom argparse "store" action to raise a DeprecationWarning.

    Based off of https://gist.github.com/bsolomon1124/44f77ed2f15062c614ef6e102bc683a5.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: U100
        """Call the argument."""
        NIWORKFLOWS_LOG.warn(
            f"Argument '{option_string}' is deprecated and will be removed in version 0.3.0. "
            "Please use '--nuisance-regressors' or '-p'."
        )
        setattr(namespace, self.dest, values)


def get_parser():
    """Build parser object."""
    from xcp_d.__about__ import __version__

    verstr = f"xcp_d v{__version__}"

    parser = ArgumentParser(
        description="xcp_d postprocessing workflow of fMRI data",
        epilog="see https://xcp-d.readthedocs.io/en/latest/generalworkflow.html",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # important parameters required
    parser.add_argument(
        "fmri_dir",
        action="store",
        type=Path,
        help="the root folder of a preprocessed fMRI output .",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="the output path for xcp_d",
    )
    parser.add_argument(
        "analysis_level",
        action="store",
        type=str,
        help='the analysis level for xcp_d, must be specified as "participant".',
    )

    # optional arguments
    parser.add_argument("--version", action="version", version=verstr)

    g_bidx = parser.add_argument_group("Options for filtering BIDS queries")
    g_bidx.add_argument(
        "--participant_label",
        "--participant-label",
        action="store",
        nargs="+",
        help=(
            "a space delimited list of participant identifiers or a single "
            "identifier (the sub- prefix can be removed)"
        ),
    )
    g_bidx.add_argument(
        "-t",
        "--task-id",
        action="store",
        help="select a specific task to be selected for the postprocessing ",
    )
    g_bidx.add_argument(
        "-m",
        "--combineruns",
        action="store_true",
        default=False,
        help="this option combines all runs into one file",
    )

    g_surfx = parser.add_argument_group("Options for cifti processing")
    g_surfx.add_argument(
        "-s",
        "--cifti",
        action="store_true",
        default=False,
        help="postprocess cifti instead of nifti this is set default for dcan and hcp",
    )

    g_perfm = parser.add_argument_group("Options to for resource management")
    g_perfm.add_argument(
        "--nthreads",
        action="store",
        type=int,
        default=2,
        help="maximum number of threads across all processes",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        action="store",
        type=int,
        default=1,
        help="maximum number of threads per-process",
    )
    g_perfm.add_argument(
        "--mem_gb",
        "--mem_gb",
        action="store",
        type=int,
        help="upper bound memory limit for xcp_d processes",
    )
    g_perfm.add_argument(
        "--use-plugin",
        action="store",
        default=None,
        help=(
            "nipype plugin configuration file. for more information see "
            "https://nipype.readthedocs.io/en/0.11.0/users/plugins.html"
        ),
    )
    g_perfm.add_argument(
        "-v",
        "--verbose",
        dest="verbose_count",
        action="count",
        default=0,
        help="increases log verbosity for each occurence, debug level is -vvv",
    )

    g_outputoption = parser.add_argument_group("Input flags")
    g_outputoption.add_argument(
        "--input-type",
        required=False,
        default='fmriprep',
        choices=['fmriprep', 'dcan', 'hpc', 'nibabies'],
        help=(
            "The pipeline used to generate the preprocessed derivatives. "
            "The default pipeline is 'fmriprep'. "
            "The 'dcan', 'hcp', and 'nibabies' pipelines are also supported. "
            "'nibabies' assumes the same structure as 'fmriprep'."
        ),
    )

    g_param = parser.add_argument_group("Parameters for postprocessing")
    g_param.add_argument(
        "--smoothing",
        default=6,
        action="store",
        type=float,
        help="smoothing the postprocessed output (fwhm)",
    )
    g_param.add_argument(
        "--despike",
        action="store_true",
        default=False,
        help="despike the nifti/cifti before postprocessing",
    )

    nuisance_params = g_param.add_mutually_exclusive_group()
    nuisance_params.add_argument(
        "--nuissance-regressors",
        dest="nuisance_regressors",
        action=DeprecatedStoreAction,
        required=False,
        default="36P",
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
        type=str,
        help=(
            "Nuisance parameters to be selected, other options include 24P and 36P acompcor and "
            "aroma. See Ciric et. al (2007) for more information about regression strategies. "
            "This parameter is deprecated and will be removed in version 0.3.0. "
            "Please use ``-p`` or ``--nuisance-regressors``."
        ),
    )
    nuisance_params.add_argument(
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
        type=str,
        help="Nuisance parameters to be selected. See Ciric et. al (2007).",
    )
    g_param.add_argument(
        "-c",
        "--custom_confounds",
        required=False,
        default=None,
        type=Path,
        help="Custom confound to be added to nuisance regressors.",
    )
    g_param.add_argument(
        "-d",
        "--dummytime",
        default=0,
        type=float,
        help="first volume in seconds to be removed or skipped before postprocessing",
    )

    g_filter = parser.add_argument_group("Filtering parameters and default value")

    bandpass_filter_params = g_filter.add_mutually_exclusive_group()
    bandpass_filter_params.add_argument(
        "--disable-bandpass-filter",
        "--disable_bandpass_filter",
        dest="bandpass_filter",
        action="store_false",
        help="Disable bandpass filtering.",
    )
    bandpass_filter_params.add_argument(
        "--bandpass_filter",
        dest="bandpass_filter",
        action=DeprecatedStoreAction,
        type=bool,
        help=(
            "Whether to Butterworth bandpass filter the data or not. "
            "This parameter is deprecated and will be removed in version 0.3.0. "
            "Bandpass filtering is performed by default, and if you wish to disable it, "
            "please use `--disable-bandpass-filter``."
        ),
    )

    g_filter.add_argument(
        "--lower-bpf",
        action="store",
        default=0.009,
        type=float,
        help="lower cut-off frequency (Hz) for the butterworth bandpass filter",
    )
    g_filter.add_argument(
        "--upper-bpf",
        action="store",
        default=0.08,
        type=float,
        help="upper cut-off frequency (Hz) for the butterworth bandpass filter",
    )
    g_filter.add_argument(
        "--bpf-order",
        action="store",
        default=2,
        type=int,
        help="number of filter coefficients for butterworth bandpass filter",
    )
    g_filter.add_argument(
        "--motion-filter-type",
        action="store",
        type=str,
        default=None,
        choices=["lp", "notch"],
        help="""\
Type of band-stop filter to use for removing respiratory artifact from motion regressors.
If not set, no filter will be applied.

If the filter type is set to "notch", then both ``band-stop-min`` and ``band-stop-max``
must be defined.
If the filter type is set to "lp", then only ``band-stop-min`` must be defined.
"""
    )
    g_filter.add_argument(
        "--band-stop-min",
        default=None,
        type=float,
        metavar="BPM",
        help="""\
Lower frequency for the band-stop motion filter, in breaths-per-minute (bpm).
Motion filtering is only performed if ``motion-filter-type`` is not None.
If used with the "lp" ``motion-filter-type``, this parameter essentially corresponds to a
low-pass filter (the maximum allowed frequency in the filtered data).
This parameter is used in conjunction with ``motion-filter-order`` and ``band-stop-max``.

.. list-table:: Recommended values, based on participant age
    :align: left
    :header-rows: 1
    :stub-columns: 1

    *   - Age Range (years)
        - Recommended Value (bpm)
    *   - < 1
        - 30
    *   - 1 - 2
        - 25
    *   - 2 - 6
        - 20
    *   - 6 - 12
        - 15
    *   - 12 - 18
        - 12
    *   - 19 - 65
        - 12
    *   - 65 - 80
        - 12
    *   - > 80
        - 10

When ``motion-filter-type`` is set to "lp" (low-pass filter), another commonly-used value for
this parameter is 6 BPM (equivalent to 0.1 Hertz), based on Gratton et al. (2020).
"""
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

.. list-table:: Recommended values, based on participant age
    :align: left
    :header-rows: 1
    :stub-columns: 1

    *   - Age Range (years)
        - Recommended Value (bpm)
    *   - < 1
        - 60
    *   - 1 - 2
        - 50
    *   - 2 - 6
        - 35
    *   - 6 - 12
        - 25
    *   - 12 - 18
        - 20
    *   - 19 - 65
        - 18
    *   - 65 - 80
        - 28
    *   - > 80
        - 30
"""
    )
    g_filter.add_argument(
        "--motion-filter-order",
        default=4,
        type=int,
        help="number of filter coeffecients for the band-stop filter",
    )

    g_censor = parser.add_argument_group("Censoring and scrubbing options")
    g_censor.add_argument(
        "-r",
        "--head_radius",
        default=50,
        type=float,
        help=(
            "head radius for computing FD, default is 50mm, "
            "35mm is recommended for baby"
        ),
    )
    g_censor.add_argument(
        "-f",
        "--fd-thresh",
        default=0.2,
        type=float,
        help="framewise displacement threshold for censoring, default is 0.2mm",
    )

    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "-w",
        "--work_dir",
        action="store",
        type=Path,
        default=Path("working_dir"),
        help="path where intermediate results should be stored",
    )
    g_other.add_argument(
        "--clean-workdir",
        action="store_true",
        default=False,
        help=(
            "Clears working directory of contents. Use of this flag is not"
            "recommended when running concurrent processes of xcp_d."
        ),
    )
    g_other.add_argument(
        "--resource-monitor",
        action="store_true",
        default=False,
        help="enable Nipype's resource monitoring to keep track of memory and CPU usage",
    )
    g_other.add_argument(
        "--notrack",
        action="store_true",
        default=False,
        help="Opt-out of sending tracking information",
    )

    g_experimental = parser.add_argument_group('Experimental options')
    g_experimental.add_argument(
        '--warp-surfaces-native2std',
        action='store_true',
        dest="process_surfaces",
        default=False,
        help="""\
If used, a workflow will be run to warp native-space (``fsnative``) reconstructed cortical
surfaces (``surf.gii`` files) produced by Freesurfer into standard (``fsLR``) space.
These surface files are primarily used for visual quality assessment.
By default, this workflow is disabled.

.. list-table:: The surface files that are generated by the workflow
    :align: left
    :header-rows: 1
    :stub-columns: 1

    * - Filename
      - Description
    * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_pial.surf.gii``
      - The gray matter / pial matter border.
    * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_smoothwm.surf.gii``
      - The smoothed gray matter / white matter border for the cortex.
    * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_midthickness.surf.gii``
      - The midpoints between wm and pial surfaces.
        This is derived from the FreeSurfer graymid
        (``mris_expand`` with distance=0.5 applied to the WM surfs).
    * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_inflated.surf.gii``
      - An inflation of the midthickness surface (useful for visualization).
        This file is only created if the input type is "hcp" or "dcan".
    * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_midthickness.surf.gii``
      - The midpoints between wm and pial surfaces.
        This is created by averaging the coordinates from the wm and pial surfaces.
    * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_inflated.surf.gii``
      - An inflation of the midthickness surface (useful for visualization).
        This is derived from the HCP midthickness file.
        This file is only created if the input type is "fmriprep" or "nibabies".
    * - ``<source_entities>_space-fsLR_den-32k_hemi-<L|R>_desc-hcp_vinflated.surf.gii``
      - A very-inflated midthicknesss surface (also for visualization).
        This is derived from the HCP midthickness file.
        This file is only created if the input type is "fmriprep" or "nibabies".
"""
    )

    return parser


def main():
    """Run the main workflow."""
    from multiprocessing import Manager, Process, set_start_method

    from nipype import logging as nlogging

    set_start_method("forkserver")
    warnings.showwarning = _warn_redirect
    opts = get_parser().parse_args()

    exec_env = os.name

    sentry_sdk = None
    if not opts.notrack:
        import sentry_sdk

        from xcp_d.utils.sentry import sentry_setup

        sentry_setup(opts, exec_env)

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    nlogging.getLogger("nipype.workflow").setLevel(log_level)
    nlogging.getLogger("nipype.interface").setLevel(log_level)
    nlogging.getLogger("nipype.utils").setLevel(log_level)

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
                logger.warning(
                    f"Could not generate CITATION.html file:\n{' '.join(cmd)}"
                )

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
                logger.warning(
                    f"Could not generate CITATION.tex file:\n{' '.join(cmd)}"
                )
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
            combineruns=opts.combineruns,
            input_type=opts.input_type,
            cifti=opts.cifti,
            config=pkgrf("xcp_d", "data/reports.yml"),
            packagename="xcp_d",
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
    from xcp_d.workflow.base import init_xcpd_wf

    build_log = nlogging.getLogger("nipype.workflow")

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
        from xcp_d.utils.bids import _add_subject_prefix

        if not opts.cifti:
            build_log.warning(
                f"With input_type {opts.input_type}, cifti processing (--cifti) will be "
                "enabled automatically."
            )
            opts.cifti = True

        if not opts.process_surfaces:
            build_log.warning(
                f"With input_type {opts.input_type}, surface processing "
                "(--warp-surfaces-native2std) will be enabled automatically."
            )
            opts.process_surfaces = True

        if opts.input_type == "dcan":
            from xcp_d.utils.dcan2fmriprep import dcan2fmriprep as convert_to_fmriprep
        elif opts.input_type == "hcp":
            from xcp_d.utils.hcp2fmriprep import hcp2fmriprep as convert_to_fmriprep

        NIWORKFLOWS_LOG.info(f"Converting {opts.input_type} to fmriprep format")
        print(f"checking the {opts.input_type} files")
        converted_fmri_dir = os.path.join(work_dir, "dcanhcp")
        os.makedirs(converted_fmri_dir, exist_ok=True)

        if opts.participant_label is not None:
            for subject_id in opts.participant_label:
                convert_to_fmriprep(
                    fmri_dir,
                    outdir=converted_fmri_dir,
                    sub_id=_add_subject_prefix(str(subject_id)),
                )
        else:
            convert_to_fmriprep(fmri_dir, outdir=converted_fmri_dir)

        fmri_dir = converted_fmri_dir

    if opts.process_surfaces and not opts.cifti:
        build_log.warning(
            "With current settings, structural surfaces will be warped to standard space, "
            "but BOLD postprocessing will be performed on volumetric data. "
            "This is not recommended."
        )

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
            "logging": {"log_directory": str(log_dir), "log_to_file": True},
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
        lower_bpf=opts.lower_bpf,
        upper_bpf=opts.upper_bpf,
        bpf_order=opts.bpf_order,
        bandpass_filter=opts.bandpass_filter,
        motion_filter_type=opts.motion_filter_type,
        motion_filter_order=opts.motion_filter_order,
        band_stop_min=opts.band_stop_min,
        band_stop_max=opts.band_stop_max,
        subject_list=subject_list,
        work_dir=str(work_dir),
        task_id=opts.task_id,
        despike=opts.despike,
        smoothing=opts.smoothing,
        params=opts.nuisance_regressors,
        cifti=opts.cifti,
        analysis_level=opts.analysis_level,
        output_dir=str(output_dir),
        head_radius=opts.head_radius,
        custom_confounds=opts.custom_confounds,
        dummytime=opts.dummytime,
        fd_thresh=opts.fd_thresh,
        process_surfaces=opts.process_surfaces,
        input_type=opts.input_type,
        name="xcpd_wf",
    )

    retval["return_code"] = 0

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
    return retval


if __name__ == "__main__":
    raise RuntimeError(
        "xcp_d/cli/run.py should not be run directly;\n"
        "Please use the `xcp_d` command-line interface."
    )
