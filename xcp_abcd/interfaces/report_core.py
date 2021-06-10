# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from pathlib import Path
from niworkflows.reports.core import Report as _Report

# this is from niworklfows, a patched will be submitted


class Report(_Report):
    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = self.packagename or settings.get("package", None)

        # Removed from here: Appending self.packagename to self.root and self.out_dir
        # In this version, pass reportlets_dir and out_dir with fmriprep in the path.

        if self.subject_id is not None:
            self.root = self.root / "sub-{}".format(self.subject_id)

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
    """
    Run the reports.

    .. testsetup::

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    >>> from pkg_resources import resource_filename
    >>> from shutil import copytree
    >>> test_data_path = resource_filename('fmriprep', 'data/tests/work')
    >>> testdir = Path(tmpdir)
    >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
    >>> (testdir / 'fmriprep').mkdir(parents=True, exist_ok=True)

    .. doctest::

    >>> run_reports(testdir / 'out', '01', 'madeoutuuid', packagename='fmriprep',
    ...             reportlets_dir=testdir / 'work' / 'reportlets' / 'fmriprep')
    0

    .. testcleanup::

    >>> os.chdir(cwd)

    """
    return Report(
        out_dir,
        run_uuid,
        config=config,
        subject_id=subject_label,
        packagename=packagename,
        reportlets_dir=reportlets_dir,
    ).generate_report()


def generate_reports(
    subject_list, fmriprep_dir, work_dir,output_dir, run_uuid, config=None, packagename=None
):
    """Execute run_reports on a list of subjects."""
    #reportlets_dir = None
    if work_dir is not None:
         print('Path(work_dir) / "reportlets"')
    report_errors = [
        run_reports(
            Path(output_dir)/'xcp_abcd',
            subject_label,
            run_uuid,
            config=config,
            packagename=packagename,
            reportlets_dir=Path(output_dir)/'xcp_abcd',
        )
        for subject_label in subject_list
    ]
    print('fmriprep_dir :' + str(fmriprep_dir))
    errno = sum(report_errors)
    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            "%s (%d)" % (subid, err)
            for subid, err in zip(subject_list, report_errors)
            if err
        )
        logger.error(
            "Processsing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    return errno