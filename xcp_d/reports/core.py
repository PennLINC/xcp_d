# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tools for generating Reports.

This is adapted from fMRIPost-AROMA.
"""

from pathlib import Path

from nireports.assembler.report import Report

from xcp_d import config, data
from xcp_d.interfaces.execsummary import ExecutiveSummary


def run_reports(
    output_dir,
    subject_label,
    run_uuid,
    bootstrap_file=None,
    out_filename="report.html",
    reportlets_dir=None,
    errorname="report.err",
    **entities,
):
    """Run the reports."""
    robj = Report(
        output_dir,
        run_uuid,
        bootstrap_file=bootstrap_file,
        out_filename=out_filename,
        reportlets_dir=reportlets_dir,
        plugins=None,
        plugin_meta=None,
        metadata=None,
        **entities,
    )

    # Count nbr of subject for which report generation failed
    try:
        robj.generate_report()
    except:  # noqa: E722
        import sys
        import traceback

        # Store the list of subjects for which report generation failed
        traceback.print_exception(*sys.exc_info(), file=str(Path(output_dir) / "logs" / errorname))
        return subject_label

    return None


def generate_reports(
    subject_list,
    output_dir,
    abcc_qc,
    run_uuid,
    session_list=None,
    bootstrap_file=None,
    work_dir=None,
):
    """Generate reports for a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"

    if isinstance(subject_list, str):
        subject_list = [subject_list]

    errors = []
    for subject_label in subject_list:
        # The number of sessions is intentionally not based on session_list but
        # on the total number of sessions, because I want the final derivatives
        # folder to be the same whether sessions were run one at a time or all-together.
        n_ses = len(config.execution.layout.get_sessions(subject=subject_label))

        if bootstrap_file is not None:
            # If a config file is precised, we do not override it
            html_report = "report.html"
        elif n_ses <= config.execution.aggr_ses_reports:
            # If there are only a few session for this subject,
            # we aggregate them in a single visual report.
            bootstrap_file = data.load("reports-spec.yml")
            html_report = "report.html"
        else:
            # Beyond a threshold, we separate the anatomical report from the functional.
            bootstrap_file = data.load("reports-spec-anat.yml")
            html_report = f'sub-{subject_label.lstrip("sub-")}_anat.html'

        report_error = run_reports(
            output_dir,
            subject_label,
            run_uuid,
            bootstrap_file=bootstrap_file,
            out_filename=html_report,
            reportlets_dir=reportlets_dir,
            errorname=f"report-{run_uuid}-{subject_label}.err",
            subject=subject_label,
        )
        # If the report generation failed, append the subject label for which it failed
        if report_error is not None:
            errors.append(report_error)

        if n_ses > config.execution.aggr_ses_reports:
            # Beyond a certain number of sessions per subject,
            # we separate the functional reports per session
            if session_list is None:
                all_filters = config.execution.bids_filters or {}
                filters = all_filters.get("bold", {})
                session_list = config.execution.layout.get_sessions(
                    subject=subject_label, **filters
                )

            # Drop ses- prefixes
            session_list = [ses[4:] if ses.startswith("ses-") else ses for ses in session_list]

            for session_label in session_list:
                bootstrap_file = data.load("reports-spec-func.yml")
                html_report = f'sub-{subject_label.lstrip("sub-")}_ses-{session_label}_func.html'

                report_error = run_reports(
                    output_dir,
                    subject_label,
                    run_uuid,
                    bootstrap_file=bootstrap_file,
                    out_filename=html_report,
                    reportlets_dir=reportlets_dir,
                    errorname=f"report-{run_uuid}-{subject_label}-func.err",
                    subject=subject_label,
                    session=session_label,
                )
                # If the report generation failed, append the subject label for which it failed
                if report_error is not None:
                    errors.append(report_error)

    if errors:
        error_list = ", ".join(
            f"{subid} ({err})" for subid, err in zip(subject_list, errors) if err
        )
        config.loggers.cli.error(
            "Processing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    elif abcc_qc:
        config.loggers.cli.info("Generating executive summary.")

        if session_list is None:
            all_filters = config.execution.bids_filters or {}
            filters = all_filters.get("bold", {})
            session_list = config.execution.layout.get_sessions(subject=subject_label, **filters)

        # Drop ses- prefixes
        session_list = [ses[4:] if ses.startswith("ses-") else ses for ses in session_list]
        if not session_list:
            session_list = [None]

        for session_label in session_list:
            exsumm = ExecutiveSummary(
                xcpd_path=output_dir,
                subject_id=subject_label,
                session_id=session_label,
            )
            exsumm.collect_inputs()
            exsumm.generate_report()

        config.loggers.cli.info("Reports generated successfully")

    return errors
