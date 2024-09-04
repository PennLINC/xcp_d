#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The XCP-D postprocessing worklow.

XCP-D postprocessing workflow
=============================
"""

from xcp_d import config


def main():
    """Entry point."""
    import gc
    import sys
    from multiprocessing import Manager, Process
    from os import EX_SOFTWARE
    from pathlib import Path

    from xcp_d.cli.parser import parse_args
    from xcp_d.cli.workflow import build_workflow
    from xcp_d.utils.bids import (
        write_atlas_dataset_description,
        write_dataset_description,
    )

    parse_args(args=sys.argv[1:])

    if "pdb" in config.execution.debug:
        from xcp_d.utils.debug import setup_exceptionhook

        setup_exceptionhook()
        config.nipype.plugin = "Linear"

    sentry_sdk = None
    if not config.execution.notrack and not config.execution.debug:
        import sentry_sdk

        from xcp_d.utils.sentry import sentry_setup

        sentry_setup()

    # CRITICAL Save the config to a file. This is necessary because the execution graph
    # is built as a separate process to keep the memory footprint low. The most
    # straightforward way to communicate with the child process is via the filesystem.
    config_file = config.execution.work_dir / config.execution.run_uuid / "config.toml"
    config_file.parent.mkdir(exist_ok=True, parents=True)
    config.to_filename(config_file)

    # CRITICAL Call build_workflow(config_file, retval) in a subprocess.
    # Because Python on Linux does not ever free virtual memory (VM), running the
    # workflow construction jailed within a process preempts excessive VM buildup.
    if "pdb" not in config.execution.debug:
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=build_workflow, args=(str(config_file), retval))
            p.start()
            p.join()
            retval = dict(retval.items())  # Convert to base dictionary

            if p.exitcode:
                retval["return_code"] = p.exitcode

    else:
        retval = build_workflow(str(config_file), {})

    exitcode = retval.get("return_code", 0)
    xcpd_wf = retval.get("workflow", None)

    # CRITICAL Load the config from the file. This is necessary because the ``build_workflow``
    # function executed constrained in a process may change the config (and thus the global
    # state of XCP-D).
    config.load(config_file)

    if config.execution.reports_only:
        sys.exit(int(exitcode > 0))

    if xcpd_wf and config.execution.write_graph:
        xcpd_wf.write_graph(graph2use="colored", format="svg", simple_form=True)

    exitcode = exitcode or (xcpd_wf is None) * EX_SOFTWARE
    if exitcode != 0:
        sys.exit(exitcode)

    # Generate boilerplate
    with Manager() as mgr:
        from xcp_d.cli.workflow import build_boilerplate

        p = Process(target=build_boilerplate, args=(str(config_file), xcpd_wf))
        p.start()
        p.join()

    if config.execution.boilerplate_only:
        sys.exit(int(exitcode > 0))

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
                config.execution.xcp_d_dir / f"sub-{s}" / "log" / config.execution.run_uuid
                for s in config.execution.participant_label
            ]
            for crashfolder in crashfolders:
                for crashfile in crashfolder.glob("crash*.*"):
                    process_crashfile(crashfile)

            if sentry_sdk is not None and "Workflow did not execute cleanly" not in str(e):
                sentry_sdk.capture_exception(e)

        config.loggers.workflow.critical("XCP-D failed: %s", e)
        raise

    else:
        config.loggers.workflow.log(25, "XCP-D finished successfully!")
        if sentry_sdk is not None:
            success_message = "XCP-D finished without errors"
            sentry_sdk.add_breadcrumb(message=success_message, level="info")
            sentry_sdk.capture_message(success_message, level="info")

        # Bother users with the boilerplate only iff the workflow went okay.
        boiler_file = config.execution.xcp_d_dir / "logs" / "CITATION.md"
        if boiler_file.exists():
            if config.environment.exec_env in (
                "apptainer",
                "docker",
            ):
                boiler_file = Path("<OUTPUT_PATH>") / boiler_file.relative_to(
                    config.execution.xcp_d_dir
                )
            config.loggers.workflow.log(
                25,
                "Works derived from this XCP-D execution should include the "
                f"boilerplate text found in {boiler_file}.",
            )

        errno = 0

    finally:
        from xcp_d.reports.core import generate_reports

        # Write dataset description before generating reports
        write_dataset_description(config.execution.fmri_dir, config.execution.xcp_d_dir)

        if config.execution.atlases:
            write_atlas_dataset_description(config.execution.xcp_d_dir / "atlases")

        # Generate reports phase
        session_list = (
            config.execution.get().get("bids_filters", {}).get("bold", {}).get("session")
        )

        # Generate reports phase
        failed_reports = generate_reports(
            subject_list=config.execution.participant_label,
            output_dir=config.execution.xcp_d_dir,
            abcc_qc=config.workflow.abcc_qc,
            run_uuid=config.execution.run_uuid,
            session_list=session_list,
        )

        if failed_reports:
            msg = (
                "Report generation was not successful for the following participants "
                f': {", ".join(failed_reports)}.'
            )
            config.loggers.cli.error(msg)
            if sentry_sdk is not None:
                sentry_sdk.capture_message(msg, level="error")

        sys.exit(int((errno + len(failed_reports)) > 0))


if __name__ == "__main__":
    raise RuntimeError(
        "xcp_d/cli/run.py should not be run directly;\n"
        "Please use the `xcp_d` command-line interface."
    )
