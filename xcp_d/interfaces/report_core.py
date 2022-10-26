# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Tools for generating Reports.

This is from niworkflows, a patch will be submitted.
"""
import glob
import os
from pathlib import Path

from niworkflows.reports.core import Report as _Report

from xcp_d.interfaces.layout_builder import LayoutBuilder
from xcp_d.utils.bids import _getsesid


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
    out_dir : str
        The output directory.
    subject_label : str
        The subject ID.
    run_uuid : str
        The UUID of the run for which the report will be generated.
    config : None or str, optional
        Configuration file.
    reportlets_dir : None or str, optional
        Path to the reportlets directory.
    packagename : None or str, optional
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


def generate_reports(subject_list,
                     fmri_dir,
                     work_dir,
                     output_dir,
                     run_uuid,
                     cifti=False,
                     config=None,
                     packagename=None,
                     combineruns=False,
                     input_type='fmriprep'):
    """Execute run_reports on a list of subjects.

    subject_list : list of str
        List of subject IDs.
    fmri_dir : str
        The path to the fMRI directory.
    work_dir : str
        The path to the working directory.
    output_dir : str
        The path to the output directory.
    run_uuid : str
        The UUID of the run for which the report will be generated.
    config : None or str, optional
        Configuration file.
    packagename : None or str, optional
        The name of the package.
    combineruns : bool, optional
        Whether to concatenate runs or not. Default is False.
    input_type : {'fmriprep', 'dcan', 'hcp'}, optional
        Default is 'fmriprep'.
    """
    # reportlets_dir = None
    if work_dir is not None:
        work_dir = work_dir
    report_errors = [
        run_reports(
            Path(output_dir) / 'xcp_d',
            subject_label,
            run_uuid,
            config=config,
            packagename=packagename,
            reportlets_dir=Path(output_dir) / 'xcp_d',
        ) for subject_label in subject_list
    ]

    fmri_dir = fmri_dir
    errno = sum(report_errors)

    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            f"{subid} ({err})"
            for subid, err in zip(subject_list, report_errors) if err)
        logger.error(
            "Processsing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    else:
        # concate cifi and nifti here for multiple runs
        if combineruns:
            from xcp_d.utils.concatenation import concatenate_derivatives

            if input_type == 'dcan':
                fmri_dir = str(work_dir) + '/dcanhcp'
            elif input_type == 'hcp':
                fmri_dir = str(work_dir) + '/hcp/hcp'
            print('Concatenating bold files ...')
            concatenate_derivatives(
                subjects=subject_list,
                fmridir=str(fmri_dir),
                outputdir=str(Path(str(output_dir)) / 'xcp_d/'),
                work_dir=work_dir,
                cifti=cifti,
            )
            print('Concatenation complete!')

        for subject_label in subject_list:
            brainplotfile = glob.glob(
                os.path.join(
                    output_dir,
                    f'xcp_d/sub-{subject_label}',
                    'figures/*_bold.svg',
                ),
            )[0]
            LayoutBuilder(html_path=str(Path(output_dir)) + '/xcp_d/',
                          subject_id=subject_label,
                          session_id=_getsesid(brainplotfile))

        print('Reports generated successfully')
    return errno
