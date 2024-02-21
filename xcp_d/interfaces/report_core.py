# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tools for generating Reports.

This is from niworkflows, a patch will be submitted.
"""
import glob
import logging
import os
from pathlib import Path

from niworkflows.reports.core import Report as _Report

from xcp_d.interfaces.execsummary import ExecutiveSummary
from xcp_d.utils.bids import get_entity
from xcp_d.utils.doc import fill_doc

LOGGER = logging.getLogger("cli")


class Report(_Report):
    """A modified form of niworkflows' core Report object."""

    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = self.packagename or settings.get("package", None)

        # Removed from here: Appending self.packagename to self.root and self.out_dir
        # In this version, pass reportlets_dir and out_dir with fmriprep in the path.

        if self.subject_id is not None:
            self.root = self.root / f"sub-{self.subject_id}"

        if "template_path" in settings:
            self.template_path = config.parent / settings["template_path"]

        self.index(settings["sections"])


#
# The following are the interface used directly by fMRIPrep
#


def run_reports(
    out_dir,
    subject_label,
    run_uuid,
    config=None,
    reportlets_dir=None,
    packagename=None,
):
    """Run the reports.

    Parameters
    ----------
    out_dir : :obj:`str`
        The output directory.
    subject_label : :obj:`str`
        The subject ID.
    run_uuid : :obj:`str`
        The UUID of the run for which the report will be generated.
    config : None or :obj:`str`, optional
        Configuration file.
    reportlets_dir : None or :obj:`str`, optional
        Path to the reportlets directory.
    packagename : None or :obj:`str`, optional
        The name of the package.

    Returns
    -------
    str
        An HTML file generated from a Report object.
    """
    return Report(
        out_dir,
        run_uuid,
        config=config,
        subject_id=subject_label,
        packagename=packagename,
        reportlets_dir=reportlets_dir,
    ).generate_report()


@fill_doc
def generate_reports(
    subject_list,
    output_dir,
    run_uuid,
    config=None,
    packagename=None,
):
    """Execute run_reports on a list of subjects.

    subject_list : :obj:`list` of :obj:`str`
        List of subject IDs.
    fmri_dir : :obj:`str`
        The path to the fMRI directory.
    work_dir : :obj:`str`
        The path to the working directory.
    output_dir : :obj:`str`
        The path to the output directory.
    run_uuid : :obj:`str`
        The UUID of the run for which the report will be generated.
    config : None or :obj:`str`, optional
        Configuration file.
    packagename : None or :obj:`str`, optional
        The name of the package.
    """
    # reportlets_dir = None
    report_errors = [
        run_reports(
            Path(output_dir),
            subject_label,
            run_uuid,
            config=config,
            packagename=packagename,
            reportlets_dir=Path(output_dir),
        )
        for subject_label in subject_list
    ]

    errno = sum(report_errors)

    if errno:
        error_list = ", ".join(
            f"{subid} ({err})" for subid, err in zip(subject_list, report_errors) if err
        )
        LOGGER.error(
            "Processing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    else:
        LOGGER.info("Generating executive summary.")
        for subject_label in subject_list:
            brainplotfiles = glob.glob(
                os.path.join(
                    output_dir,
                    f"sub-{subject_label}",
                    "figures/*_bold.svg",
                ),
            )
            if not brainplotfiles:
                LOGGER.warning(
                    "No postprocessing BOLD figures found for subject %s.",
                    subject_label,
                )
                session_id = None
            else:
                brainplotfile = brainplotfiles[0]
                session_id = get_entity(brainplotfile, "ses")

            exsumm = ExecutiveSummary(
                xcpd_path=output_dir,
                subject_id=subject_label,
                session_id=session_id,
            )
            exsumm.collect_inputs()
            exsumm.generate_report()

        print("Reports generated successfully")
    return errno
