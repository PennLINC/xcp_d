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
    processing_list,
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
    # The number of sessions is intentionally not based on session_list but
    # on the total number of sessions, because I want the final derivatives
    # folder to be the same whether sessions were run one at a time or all-together.
    all_func_sessions = config.execution.layout.get_sessions(suffix='bold')
    n_ses = len(all_func_sessions)

    errors = []
    bids_filters = config.execution.bids_filters or {}
    for subject_label, _, sessions in processing_list:
        subject_label = subject_label[4:] if subject_label.startswith('sub-') else subject_label
        # Drop ses- prefixes
        sessions = [ses or Query.NONE for ses in sessions]  # replace "" with Query.NONE
        sessions = [ses for ses in sessions if isinstance(ses, str)]  # drop Queries
        sessions = [ses[4:] if ses.startswith('ses-') else ses for ses in sessions]

        if output_level == 'session' and not sessions:
            report_dir = dataset_dir
            output_level = 'subject'
            config.loggers.workflow.warning(
                'Session-level reports were requested, '
                'but data was found without a session level. '
                'Writing out reports to subject level.'
            )

        if output_level != 'session' and n_ses <= config.execution.aggr_ses_reports:
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
                dataset_dir=dataset_dir,
                errorname=f'report-{run_uuid}-{subject_label}.err',
                subject=subject_label,
                session=None,
            )
            # If the report generation failed, append the subject label for which it failed
            if report_error is not None:
                errors.append(report_error)

            if abcc_qc:
                # Executive summaries should always be split by session, when sessions exist.
                if not sessions:
                    sessions = [None]

                for session_label in sessions:
                    exsumm = ExecutiveSummary(
                        xcpd_path=dataset_dir,
                        output_dir=report_dir,
                        subject_id=subject_label,
                        session_id=session_label,
                    )
                    exsumm.collect_inputs()
                    exsumm.generate_report()
        else:
            if not sessions:
                sessions = [None]

            for session_label in sessions:
                if session_label is None:
                    html_report = f'sub-{subject_label}.html'
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
                    dataset_dir=dataset_dir,
                    errorname=f'report-{run_uuid}-{subject_label}-{session_label}.err',
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

    if errors:
        # Suboptimal to just report the subject IDs (with potential duplicates)
        subject_list = [sl[0] for sl in processing_list]
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
