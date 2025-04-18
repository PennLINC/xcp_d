# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tools for generating Reports.

This is adapted from fMRIPost-AROMA.
"""

from pathlib import Path

from bids.layout import Query
from nireports.assembler.report import Report

from xcp_d import config, data
from xcp_d.interfaces.execsummary import ExecutiveSummary


def run_reports(
    out_dir,
    subject_label,
    run_uuid,
    bootstrap_file=None,
    out_filename='report.html',
    dataset_dir=None,
    errorname='report.err',
    metadata=None,
    **entities,
):
    """Run the reports."""
    robj = Report(
        out_dir,
        run_uuid,
        bootstrap_file=bootstrap_file,
        out_filename=out_filename,
        reportlets_dir=dataset_dir,
        plugins=None,
        plugin_meta=None,
        metadata=metadata,
        **entities,
    )

    # Count nbr of subject for which report generation failed
    try:
        robj.generate_report()
    except:  # noqa: E722
        import sys
        import traceback

        # Store the list of subjects for which report generation failed
        traceback.print_exception(
            *sys.exc_info(),
            file=str(Path(out_dir) / 'logs' / errorname),
        )
        return subject_label

    return None


def generate_reports(
    output_level,
    dataset_dir,
    abcc_qc,
    run_uuid,
    bootstrap_file=None,
    work_dir=None,
):
    """Generate reports for a list of subjects.

    Parameters
    ----------
    output_level : {'root', 'subject', 'session'}
    """
    errors = []
    bootstrap_file = data.load('reports-spec.yml') if bootstrap_file is None else bootstrap_file

    bids_filters = config.execution.bids_filters or {}
    subject_list = config.execution.participant_label
    subject_list = [sub[4:] if sub.startswith('sub-') else sub for sub in subject_list]
    for subject_label in subject_list:
        filters = bids_filters.get('bold', {})
        filters['session'] = config.execution.session_id or Query.OPTIONAL
        # Extract session IDs from the processed DWIs
        sessions = config.execution.layout.get_sessions(
            subject=subject_label,
            suffix='bold',
            **filters,
        )
        if output_level == 'session' and not sessions:
            report_dir = dataset_dir
            output_level = 'subject'
            config.loggers.workflow.warning(
                'Session-level reports were requested, '
                'but data was found without a session level. '
                'Writing out reports to subject level.'
            )

        if not sessions:
            sessions = [Query.NONE]

        if output_level != 'session' and len(sessions) <= config.execution.aggr_ses_reports:
            html_report = f'sub-{subject_label}.html'

            if output_level == 'root':
                report_dir = dataset_dir
            elif output_level == 'subject':
                report_dir = Path(dataset_dir) / f'sub-{subject_label}'

            report_error = run_reports(
                out_dir=report_dir,
                subject_label=subject_label,
                run_uuid=run_uuid,
                bootstrap_file=bootstrap_file,
                out_filename=html_report,
                reportlets_dir=dataset_dir,
                errorname=f'report-{run_uuid}-{subject_label}.err',
                subject=subject_label,
                session=None,
            )
            # If the report generation failed, append the subject label for which it failed
            if report_error is not None:
                errors.append(report_error)

            if abcc_qc:
                for session_label in sessions:
                    if session_label == Query.NONE:
                        session_label = None

                    exsumm = ExecutiveSummary(
                        xcpd_path=dataset_dir,
                        output_dir=report_dir,
                        subject_id=subject_label,
                        session_id=session_label,
                    )
                    exsumm.collect_inputs()
                    exsumm.generate_report()
        else:
            for session_label in sessions:
                if session_label == Query.NONE:
                    html_report = f'sub-{subject_label}.html'
                    session_label = None
                else:
                    html_report = f'sub-{subject_label}_ses-{session_label}.html'

                if output_level == 'root':
                    report_dir = dataset_dir
                elif output_level == 'subject':
                    report_dir = Path(dataset_dir) / f'sub-{subject_label}'
                elif output_level == 'session':
                    report_dir = (
                        Path(dataset_dir) / f'sub-{subject_label}' / f'ses-{session_label}'
                    )

                report_error = run_reports(
                    out_dir=report_dir,
                    subject_label=subject_label,
                    run_uuid=run_uuid,
                    bootstrap_file=bootstrap_file,
                    out_filename=html_report,
                    reportlets_dir=dataset_dir,
                    errorname=f'report-{run_uuid}-{subject_label}.err',
                    metadata={
                        'session_str': f", session '{session_label}'" if session_label else '',
                    },
                    subject=subject_label,
                    session=session_label,
                )
                # If the report generation failed, append the subject label for which it failed
                if report_error is not None:
                    errors.append(report_error)

                if abcc_qc:
                    exsumm = ExecutiveSummary(
                        xcpd_path=dataset_dir,
                        output_dir=report_dir,
                        subject_id=subject_label,
                        session_id=session_label,
                    )
                    exsumm.collect_inputs()
                    exsumm.generate_report()

            # Someday, when we have anatomical reports, add a section here that
            # finds sessions and makes the reports.

    if errors:
        error_list = ', '.join(
            f'{subid} ({err})' for subid, err in zip(subject_list, errors, strict=False) if err
        )
        config.loggers.cli.error(
            'Processing did not finish successfully. Errors occurred while processing '
            'data from participants: %s. Check the HTML reports for details.',
            error_list,
        )

    config.loggers.cli.info('Reports generated successfully')

    return errors
