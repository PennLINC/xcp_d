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
            config.execution.fmri_dir / "derivatives" /
            f"xcp_d-{config.environment.version.split('+')[0]}"
        )
        config.loggers.cli.error(
            "The selected output folder is the same as the input fmri input. "
            "Please modify the output path "
            f"(suggestion: {rec_path})."
        )
        return_code = 1

    if config.workflow.analysis_level != "participant":
        config.loggers.cli.error('Please select analysis level "participant"')
        return_code = 1

    # Bandpass filter parameters
    if config.workflow.lower_bpf <= 0 and config.workflow.upper_bpf <= 0:
        config.workflow.bandpass_filter = False

    if (
        config.workflow.bandpass_filter
        and (config.workflow.lower_bpf >= config.workflow.upper_bpf)
        and (config.workflow.lower_bpf > 0 and config.workflow.upper_bpf > 0)
    ):
        config.loggers.cli.error(
            f"'--lower-bpf' ({config.workflow.lower_bpf}) must be lower than "
            f"'--upper-bpf' ({config.workflow.upper_bpf})."
        )
        return_code = 1
    elif not config.workflow.bandpass_filter:
        config.loggers.cli.warning(
            "Bandpass filtering is disabled. ALFF outputs will not be generated."
        )

    # Scrubbing parameters
    if config.workflow.fd_thresh <= 0:
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
        config.workflow.min_time = 0
        config.workflow.motion_filter_type = None
        config.workflow.band_stop_min = None
        config.workflow.band_stop_max = None
        config.workflow.motion_filter_order = None

    # Motion filtering parameters
    if config.workflow.motion_filter_type == "notch":
        if not (config.workflow.band_stop_min and config.workflow.band_stop_max):
            config.loggers.cli.error(
                "Please set both '--band-stop-min' and '--band-stop-max' if you want to apply "
                "the 'notch' motion filter."
            )
            return_code = 1
        elif config.workflow.band_stop_min >= config.workflow.band_stop_max:
            config.loggers.cli.error(
                f"'--band-stop-min' ({config.workflow.band_stop_min}) must be lower than "
                f"'--band-stop-max' ({config.workflow.band_stop_max})."
            )
            return_code = 1
        elif config.workflow.band_stop_min < 1 or config.workflow.band_stop_max < 1:
            config.loggers.cli.warning(
                f"Either '--band-stop-min' ({config.workflow.band_stop_min}) or "
                f"'--band-stop-max' ({config.workflow.band_stop_max}) is suspiciously low. "
                "Please remember that these values should be in breaths-per-minute."
            )

    elif config.workflow.motion_filter_type == "lp":
        if not config.workflow.band_stop_min:
            config.loggers.cli.error(
                "Please set '--band-stop-min' if you want to apply the 'lp' motion filter."
            )
            return_code = 1
        elif config.workflow.band_stop_min < 1:
            config.loggers.cli.warning(
                f"'--band-stop-min' ({config.workflow.band_stop_max}) is suspiciously low. "
                "Please remember that this value should be in breaths-per-minute."
            )

        if config.workflow.band_stop_max:
            config.loggers.cli.warning(
                "'--band-stop-max' is ignored when '--motion-filter-type' is 'lp'."
            )

    elif config.workflow.band_stop_min or config.workflow.band_stop_max:
        config.loggers.cli.warning(
            "'--band-stop-min' and '--band-stop-max' are ignored if '--motion-filter-type' "
            "is not set."
        )

    # Some parameters are automatically set depending on the input type.
    if config.workflow.input_type in ("dcan", "hcp"):
        if not config.workflow.cifti:
            config.loggers.cli.warning(
                f"With input_type {config.workflow.input_type}, "
                "cifti processing (--cifti) will be enabled automatically."
            )
            config.workflow.cifti = True

        if not config.workflow.process_surfaces:
            config.loggers.cli.warning(
                f"With input_type {config.workflow.input_type}, surface normalization "
                "(--warp-surfaces-native2std) will be enabled automatically."
            )
            config.workflow.process_surfaces = True

    # process_surfaces and nifti processing are incompatible.
    if config.workflow.process_surfaces and not config.workflow.cifti:
        config.loggers.cli.error(
            "In order to perform surface normalization (--warp-surfaces-native2std), "
            "you must enable cifti processing (--cifti)."
        )
        return_code = 1

    return return_code


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


if __name__ == "__main__":
    raise RuntimeError(
        "xcp_d/cli/run.py should not be run directly;\n"
        "Please use the `xcp_d` command-line interface."
    )
