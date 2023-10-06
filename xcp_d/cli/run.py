#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The XCP-D preprocessing worklow."""
from xcp_d import config


def _validate_parameters():
    """Validate parameters.

    This function was abstracted out of build_workflow to make testing easier.
    """
    import os

    config.execution.fmri_dir = config.execution.fmri_dir.resolve()
    config.execution.output_dir = config.execution.output_dir.resolve()
    config.execution.work_dir = config.execution.work_dir.resolve()

    return_code = 0

    # Set the FreeSurfer license
    if config.execution.fs_license_file is not None:
        config.execution.fs_license_file = config.execution.fs_license_file.resolve()
        if config.execution.fs_license_file.is_file():
            os.environ["FS_LICENSE"] = str(config.execution.fs_license_file)

        else:
            config.loggers.cli.error(
                f"Freesurfer license DNE: {config.execution.fs_license_file}."
            )
            return_code = 1

    # Check the validity of inputs
    if config.execution.output_dir == config.execution.fmri_dir:
        rec_path = (
            config.execution.fmri_dir / "derivatives" / f"xcp_d-{config.environment.version.split('+')[0]}"
        )
        config.loggers.cli.error(
            "The selected output folder is the same as the input fmri input. "
            "Please modify the output path "
            f"(suggestion: {rec_path})."
        )
        return_code = 1

    if opts.analysis_level != "participant":
        config.loggers.cli.error('Please select analysis level "participant"')
        return_code = 1

    # Bandpass filter parameters
    if opts.lower_bpf <= 0 and opts.upper_bpf <= 0:
        opts.bandpass_filter = False

    if (
        opts.bandpass_filter
        and (opts.lower_bpf >= opts.upper_bpf)
        and (opts.lower_bpf > 0 and opts.upper_bpf > 0)
    ):
        config.loggers.cli.error(
            f"'--lower-bpf' ({opts.lower_bpf}) must be lower than "
            f"'--upper-bpf' ({opts.upper_bpf})."
        )
        return_code = 1
    elif not opts.bandpass_filter:
        config.loggers.cli.warning("Bandpass filtering is disabled. ALFF outputs will not be generated.")

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
        config.loggers.cli.warning(
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
            config.loggers.cli.error(
                "Please set both '--band-stop-min' and '--band-stop-max' if you want to apply "
                "the 'notch' motion filter."
            )
            return_code = 1
        elif opts.band_stop_min >= opts.band_stop_max:
            config.loggers.cli.error(
                f"'--band-stop-min' ({opts.band_stop_min}) must be lower than "
                f"'--band-stop-max' ({opts.band_stop_max})."
            )
            return_code = 1
        elif opts.band_stop_min < 1 or opts.band_stop_max < 1:
            config.loggers.cli.warning(
                f"Either '--band-stop-min' ({opts.band_stop_min}) or "
                f"'--band-stop-max' ({opts.band_stop_max}) is suspiciously low. "
                "Please remember that these values should be in breaths-per-minute."
            )

    elif opts.motion_filter_type == "lp":
        if not opts.band_stop_min:
            config.loggers.cli.error(
                "Please set '--band-stop-min' if you want to apply the 'lp' motion filter."
            )
            return_code = 1
        elif opts.band_stop_min < 1:
            config.loggers.cli.warning(
                f"'--band-stop-min' ({opts.band_stop_max}) is suspiciously low. "
                "Please remember that this value should be in breaths-per-minute."
            )

        if opts.band_stop_max:
            config.loggers.cli.warning("'--band-stop-max' is ignored when '--motion-filter-type' is 'lp'.")

    elif opts.band_stop_min or opts.band_stop_max:
        config.loggers.cli.warning(
            "'--band-stop-min' and '--band-stop-max' are ignored if '--motion-filter-type' "
            "is not set."
        )

    # Some parameters are automatically set depending on the input type.
    if opts.input_type in ("dcan", "hcp"):
        if not opts.cifti:
            config.loggers.cli.warning(
                f"With input_type {opts.input_type}, cifti processing (--cifti) will be "
                "enabled automatically."
            )
            opts.cifti = True

        if not opts.process_surfaces:
            config.loggers.cli.warning(
                f"With input_type {opts.input_type}, surface normalization "
                "(--warp-surfaces-native2std) will be enabled automatically."
            )
            opts.process_surfaces = True

    # process_surfaces and nifti processing are incompatible.
    if opts.process_surfaces and not opts.cifti:
        config.loggers.cli.error(
            "In order to perform surface normalization (--warp-surfaces-native2std), "
            "you must enable cifti processing (--cifti)."
        )
        return_code = 1

    return opts, return_code


def main(args=None, namespace=None):
    """Run the main workflow."""
    import gc
    import sys
    from multiprocessing import Manager, Process
    from os import EX_SOFTWARE
    from pathlib import Path

    from xcp_d.utils.bids import write_derivative_description

    from multiprocessing import Manager, Process

    from xcp_d.cli.parser import parse_args

    parse_args()

    sentry_sdk = None
    if not config.execution.notrack:
        import sentry_sdk

        from xcp_d.utils.sentry import sentry_setup

        sentry_setup()

    # Validate the config before writing it out to a file
    _validate_parameters()

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low.
    # The most straightforward way to communicate with the child process is via the filesystem.
    config_file = config.execution.work_dir / f"config-{config.execution.run_uuid}.toml"
    config.to_filename(config_file)

    # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
    # Because Python on Linux does not ever free virtual memory (VM), running the
    # workflow construction jailed within a process preempts excessive VM buildup.
    with Manager() as mgr:
        from xcp_d.cli.workflow import build_workflow

        retval = mgr.dict()
        p = Process(target=build_workflow, args=(str(config_file), retval))
        p.start()
        p.join()

        retcode = p.exitcode or retval.get("return_code", 0)
        xcpd_wf = retval.get("workflow", None)

    # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
    # function executed constrained in a process may change the config (and thus the global
    # state of XCP-D).
    config.load(config_file)

    if config.execution.reports_only:
        sys.exit(int(retcode > 0))

    if xcpd_wf and config.execution.write_graph:
        xcpd_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

    retcode = retcode or (xcpd_wf is None) * EX_SOFTWARE
    if retcode != 0:
        sys.exit(retcode)

    # Generate boilerplate
    with Manager() as mgr:
        from xcp_d.cli.workflow import build_boilerplate

        p = Process(target=build_boilerplate, args=(str(config_file), xcpd_wf))
        p.start()
        p.join()

    if config.execution.boilerplate_only:
        sys.exit(int(retcode > 0))

    # Clean up master process before running workflow, which may create forks
    gc.collect()

    # Sentry tracking
    if sentry_sdk is not None:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("run_uuid", config.execution.run_uuid)
            scope.set_tag("npart", len(config.execution.participant_label))
        sentry_sdk.add_breadcrumb(message="XCP-D started", level="info")
        sentry_sdk.capture_message("XCP-D started", level="info")

    config.loggers.workflow.log(
        15,
        "\n".join(["XCP-D config:"] + [f"\t\t{s}" for s in config.dumps().splitlines()]),
    )
    config.loggers.workflow.log(25, "XCP-D started!")
    errno = 1  # Default is error exit unless otherwise set
    try:
        xcpd_wf.run(**config.nipype.get_plugin())
    except Exception as e:
        if not config.execution.notrack:
            from xcp_d.utils.sentry import process_crashfile

            crashfolders = [
                config.execution.output_dir
                / "xcp_d"
                / f"sub-{s}"
                / "log"
                / config.execution.run_uuid
                for s in config.execution.participant_label
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob("crash*.*"):
                    process_crashfile(crashfile)

            if "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)
        config.loggers.workflow.critical("XCP-D failed: %s", e)
        raise
    else:
        config.loggers.workflow.log(25, "XCP-D finished successfully!")
        if not config.execution.notrack:
            success_message = "XCP-D finished without errors"
            sentry_sdk.add_breadcrumb(message=success_message, level="info")
            sentry_sdk.capture_message(success_message, level="info")

        # Bother users with the boilerplate only iff the workflow went okay.
        boiler_file = config.execution.output_dir / "xcp_d" / "logs" / "CITATION.md"
        if boiler_file.exists():
            if config.environment.exec_env in (
                "singularity",
                "docker",
                "xcp_d-docker",
            ):
                boiler_file = Path("<OUTPUT_PATH>") / boiler_file.relative_to(
                    config.execution.output_dir
                )
            config.loggers.workflow.log(
                25,
                "Works derived from this XCP-D execution should include the "
                f"boilerplate text found in {boiler_file}.",
            )
        errno = 0
    finally:
        from niworkflows.reports.core import generate_reports
        from pkg_resources import resource_filename as pkgrf

        # Generate reports phase
        failed_reports = generate_reports(
            config.execution.participant_label,
            config.execution.output_dir,
            config.execution.run_uuid,
            config=pkgrf("xcp_d", "data/reports-spec.yml"),
            packagename="xcp_d",
        )
        write_derivative_description(
            config.execution.bids_dir, config.execution.output_dir / "xcp_d"
        )

        if failed_reports and not config.execution.notrack:
            sentry_sdk.capture_message(
                f"Report generation failed for {failed_reports} subjects",
                level="error",
            )
        sys.exit(int((errno + failed_reports) > 0))


def _validate_parameters(opts, config.loggers.cli):
    """Validate parameters.

    This function was abstracted out of build_workflow to make testing easier.
    """
    opts.fmri_dir = opts.fmri_dir.resolve()
    opts.output_dir = opts.output_dir.resolve()
    opts.work_dir = opts.work_dir.resolve()

    return_code = 0

    # Set the FreeSurfer license
    if opts.fs_license_file is not None:
        opts.fs_license_file = opts.fs_license_file.resolve()
        if opts.fs_license_file.is_file():
            os.environ["FS_LICENSE"] = str(opts.fs_license_file)

        else:
            config.loggers.cli.error(f"Freesurfer license DNE: {opts.fs_license_file}.")
            return_code = 1

    # Check the validity of inputs
    if opts.output_dir == opts.fmri_dir:
        rec_path = (
            opts.fmri_dir / "derivatives" / f"xcp_d-{config.environment.version.split('+')[0]}"
        )
        config.loggers.cli.error(
            "The selected output folder is the same as the input fmri input. "
            "Please modify the output path "
            f"(suggestion: {rec_path})."
        )
        return_code = 1

    if opts.analysis_level != "participant":
        config.loggers.cli.error('Please select analysis level "participant"')
        return_code = 1

    # Bandpass filter parameters
    if opts.lower_bpf <= 0 and opts.upper_bpf <= 0:
        opts.bandpass_filter = False

    if (
        opts.bandpass_filter
        and (opts.lower_bpf >= opts.upper_bpf)
        and (opts.lower_bpf > 0 and opts.upper_bpf > 0)
    ):
        config.loggers.cli.error(
            f"'--lower-bpf' ({opts.lower_bpf}) must be lower than "
            f"'--upper-bpf' ({opts.upper_bpf})."
        )
        return_code = 1
    elif not opts.bandpass_filter:
        config.loggers.cli.warning("Bandpass filtering is disabled. ALFF outputs will not be generated.")

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
        config.loggers.cli.warning(
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
            config.loggers.cli.error(
                "Please set both '--band-stop-min' and '--band-stop-max' if you want to apply "
                "the 'notch' motion filter."
            )
            return_code = 1
        elif opts.band_stop_min >= opts.band_stop_max:
            config.loggers.cli.error(
                f"'--band-stop-min' ({opts.band_stop_min}) must be lower than "
                f"'--band-stop-max' ({opts.band_stop_max})."
            )
            return_code = 1
        elif opts.band_stop_min < 1 or opts.band_stop_max < 1:
            config.loggers.cli.warning(
                f"Either '--band-stop-min' ({opts.band_stop_min}) or "
                f"'--band-stop-max' ({opts.band_stop_max}) is suspiciously low. "
                "Please remember that these values should be in breaths-per-minute."
            )

    elif opts.motion_filter_type == "lp":
        if not opts.band_stop_min:
            config.loggers.cli.error(
                "Please set '--band-stop-min' if you want to apply the 'lp' motion filter."
            )
            return_code = 1
        elif opts.band_stop_min < 1:
            config.loggers.cli.warning(
                f"'--band-stop-min' ({opts.band_stop_max}) is suspiciously low. "
                "Please remember that this value should be in breaths-per-minute."
            )

        if opts.band_stop_max:
            config.loggers.cli.warning("'--band-stop-max' is ignored when '--motion-filter-type' is 'lp'.")

    elif opts.band_stop_min or opts.band_stop_max:
        config.loggers.cli.warning(
            "'--band-stop-min' and '--band-stop-max' are ignored if '--motion-filter-type' "
            "is not set."
        )

    # Some parameters are automatically set depending on the input type.
    if opts.input_type in ("dcan", "hcp"):
        if not opts.cifti:
            config.loggers.cli.warning(
                f"With input_type {opts.input_type}, cifti processing (--cifti) will be "
                "enabled automatically."
            )
            opts.cifti = True

        if not opts.process_surfaces:
            config.loggers.cli.warning(
                f"With input_type {opts.input_type}, surface normalization "
                "(--warp-surfaces-native2std) will be enabled automatically."
            )
            opts.process_surfaces = True

    # process_surfaces and nifti processing are incompatible.
    if opts.process_surfaces and not opts.cifti:
        config.loggers.cli.error(
            "In order to perform surface normalization (--warp-surfaces-native2std), "
            "you must enable cifti processing (--cifti)."
        )
        return_code = 1

    return opts, return_code


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

    from xcp_d.utils.bids import collect_participants
    from xcp_d.workflows.base import init_xcpd_wf

    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))

    config.loggers.cli = nlogging.getLogger("nipype.workflow")
    config.loggers.cli.setLevel(log_level)
    nlogging.getLogger("nipype.interface").setLevel(log_level)
    nlogging.getLogger("nipype.utils").setLevel(log_level)

    opts, retval["return_code"] = _validate_parameters(opts, config.loggers.cli)

    if retval["return_code"] == 1:
        return retval

    if opts.clean_workdir:
        from niworkflows.utils.misc import clean_directory

        config.loggers.cli.info(f"Clearing previous xcp_d working directory: {opts.work_dir}")
        if not clean_directory(opts.work_dir):
            config.loggers.cli.warning(
                f"Could not clear all contents of working directory: {opts.work_dir}"
            )

    retval["return_code"] = 1
    retval["workflow"] = None
    retval["fmri_dir"] = str(opts.fmri_dir)
    retval["output_dir"] = str(opts.output_dir)
    retval["work_dir"] = str(opts.work_dir)

    # First check that fmriprep_dir looks like a BIDS folder
    if opts.input_type in ("dcan", "hcp"):
        if opts.input_type == "dcan":
            from xcp_d.utils.dcan2fmriprep import convert_dcan2bids as convert_to_bids
        elif opts.input_type == "hcp":
            from xcp_d.utils.hcp2fmriprep import convert_hcp2bids as convert_to_bids

        NIWORKFLOWS_LOG.info(f"Converting {opts.input_type} to fmriprep format")
        converted_fmri_dir = os.path.join(
            opts.work_dir,
            f"dset_bids/derivatives/{opts.input_type}",
        )
        os.makedirs(converted_fmri_dir, exist_ok=True)

        convert_to_bids(
            opts.fmri_dir,
            out_dir=converted_fmri_dir,
            participant_ids=opts.participant_label,
        )

        opts.fmri_dir = Path(converted_fmri_dir)

    if not os.path.isfile((os.path.join(opts.fmri_dir, "dataset_description.json"))):
        config.loggers.cli.error(
            "No dataset_description.json file found in input directory. "
            "Make sure to point to the specific pipeline's derivatives folder. "
            "For example, use '/dset/derivatives/fmriprep', not /dset/derivatives'."
        )
        retval["return_code"] = 1

    # Set up some instrumental utilities
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4()}"
    retval["run_uuid"] = run_uuid

    layout = BIDSLayout(str(opts.fmri_dir), validate=False, derivatives=True)
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

    # Permit overriding plugin config with specific CLI options
    nthreads = opts.nthreads
    omp_nthreads = opts.omp_nthreads

    if (nthreads == 1) or (omp_nthreads > nthreads):
        omp_nthreads = 1

    plugin_settings["plugin_args"]["n_procs"] = nthreads

    if 1 < nthreads < omp_nthreads:
        config.loggers.cli.warning(
            f"Per-process threads (--omp-nthreads={omp_nthreads}) exceed total "
            f"threads (--nthreads/--n_cpus={nthreads})"
        )

    if opts.mem_gb:
        plugin_settings["plugin_args"]["memory_gb"] = opts.mem_gb

    retval["plugin_settings"] = plugin_settings

    # Set up directories
    log_dir = opts.output_dir / "xcp_d" / "logs"

    # Check and create output and working directories
    opts.output_dir.mkdir(exist_ok=True, parents=True)
    opts.work_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

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
    config.loggers.cli.log(
        25,
        f"""\
Running xcp_d version {config.environment.version}:
    * fMRI directory path: {opts.fmri_dir}.
    * Participant list: {subject_list}.
    * Run identifier: {run_uuid}.

""",
    )

    retval["workflow"] = init_xcpd_wf(
        subject_list=subject_list,
        name="xcpd_wf",
    )

    boilerplate = retval["workflow"].visit_desc()

    if boilerplate:
        citation_files = {ext: log_dir / f"CITATION.{ext}" for ext in ("bib", "tex", "md", "html")}
        # To please git-annex users and also to guarantee consistency among different renderings
        # of the same file, first remove any existing ones
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

        citation_files["md"].write_text(boilerplate)

    config.loggers.cli.log(
        25,
        (
            "Works derived from this xcp_d execution should include the following boilerplate:\n\n"
            f"{boilerplate}"
        ),
    )

    retval["return_code"] = 0

    return retval


if __name__ == "__main__":
    raise RuntimeError(
        "xcp_d/cli/run.py should not be run directly;\n"
        "Please use the `xcp_d` command-line interface."
    )
