"""
The workflow builder factory method.

All the checks and the construction of the workflow are done
inside this function that has pickleable inputs and output
dictionary (``retval``) to allow isolation using a
``multiprocessing.Process`` that allows XCP-D to enforce
a hard-limited memory-scope.

"""


def build_workflow(config_file, retval):
    """Create the Nipype Workflow that supports the whole execution graph."""
    from niworkflows.utils.bids import collect_participants
    from niworkflows.utils.misc import check_valid_fs_license

    from xcp_d import config, data
    from xcp_d.interfaces.report_core import generate_reports
    from xcp_d.utils.bids import check_pipeline_version
    from xcp_d.utils.utils import check_deps
    from xcp_d.workflows.base import init_xcpd_wf

    config.load(config_file)
    build_log = config.loggers.workflow

    version = config.environment.version

    retval["return_code"] = 1
    retval["workflow"] = None

    banner = [f"Running XCP-D version {version}"]
    notice_path = data.load.readable("NOTICE")
    if notice_path.exists():
        banner[0] += "\n"
        banner += [f"License NOTICE {'#' * 50}"]
        banner += [f"XCP-D {version}"]
        banner += notice_path.read_text().splitlines(keepends=False)[1:]
        banner += ["#" * len(banner[1])]
    build_log.log(25, f"\n{' ' * 9}".join(banner))

    # warn if older results exist: check for dataset_description.json in output folder
    msg = check_pipeline_version(
        "XCP-D",
        version,
        config.execution.xcp_d_dir / "dataset_description.json",
    )
    if msg is not None:
        build_log.warning(msg)

    # Please note this is the input folder's dataset_description.json
    dset_desc_path = config.execution.fmri_dir / "dataset_description.json"
    if dset_desc_path.exists():
        from hashlib import sha256

        desc_content = dset_desc_path.read_bytes()
        config.execution.bids_description_hash = sha256(desc_content).hexdigest()

    # First check that fmri_dir looks like a BIDS folder
    subject_list = collect_participants(
        config.execution.fmri_dir, participant_label=config.execution.participant_label
    )

    # Called with reports only
    if config.execution.reports_only:
        from xcp_d.data import load as load_data

        build_log.log(25, "Running --reports-only on participants %s", ", ".join(subject_list))
        retval["return_code"] = generate_reports(
            subject_list,
            config.execution.xcp_d_dir,
            config.execution.run_uuid,
            config=load_data("reports-spec.yml"),
            packagename="xcp_d",
        )
        return retval

    # Build main workflow
    init_msg = [
        "Building XCP-D's workflow:",
        f"Preprocessing derivatives path: {config.execution.fmri_dir}.",
        f"Participant list: {subject_list}.",
        f"Run identifier: {config.execution.run_uuid}.",
    ]

    if config.execution.derivatives:
        init_msg += [f"Searching for derivatives: {config.execution.derivatives}."]

    if config.execution.fs_subjects_dir:
        init_msg += [f"Pre-run FreeSurfer's SUBJECTS_DIR: {config.execution.fs_subjects_dir}."]

    build_log.log(25, f"\n{' ' * 11}* ".join(init_msg))

    retval["workflow"] = init_xcpd_wf()

    # Check for FS license after building the workflow
    if not check_valid_fs_license():
        build_log.critical(
            """\
ERROR: a valid license file is required for FreeSurfer to run. XCP-D looked for an existing \
license file at several paths, in this order: 1) command line argument ``--fs-license-file``; \
2) ``$FS_LICENSE`` environment variable; and 3) the ``$FREESURFER_HOME/license.txt`` path. Get it \
(for free) by registering at https://surfer.nmr.mgh.harvard.edu/registration.html"""
        )
        retval["return_code"] = 126  # 126 == Command invoked cannot execute.
        return retval

    # Check workflow for missing commands
    missing = check_deps(retval["workflow"])
    if missing:
        build_log.critical(
            "Cannot run XCP-D. Missing dependencies:%s",
            "\n\t* ".join([""] + [f"{cmd} (Interface: {iface})" for iface, cmd in missing]),
        )
        retval["return_code"] = 127  # 127 == command not found.
        return retval

    config.to_filename(config_file)
    build_log.info(
        "XCP-D workflow graph with %d nodes built successfully.",
        len(retval["workflow"]._get_all_nodes()),
    )
    retval["return_code"] = 0
    return retval


def build_boilerplate(config_file, workflow):
    """Write boilerplate in an isolated process."""
    from xcp_d import config

    config.load(config_file)
    logs_path = config.execution.xcp_d_dir / "logs"
    boilerplate = workflow.visit_desc()
    citation_files = {ext: logs_path / f"CITATION.{ext}" for ext in ("bib", "tex", "md", "html")}

    if boilerplate:
        # To please git-annex users and also to guarantee consistency
        # among different renderings of the same file, first remove any
        # existing one
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

    citation_files["md"].write_text(boilerplate)

    if not config.execution.md_only_boilerplate and citation_files["md"].exists():
        from shutil import copyfile
        from subprocess import CalledProcessError, TimeoutExpired, check_call

        from xcp_d.data import load as load_data

        # Generate HTML file resolving citations
        cmd = [
            "pandoc",
            "-s",
            "--bibliography",
            str(load_data("boilerplate.bib")),
            "--filter",
            "pandoc-citeproc",
            "--metadata",
            'pagetitle="XCP-D citation boilerplate"',
            str(citation_files["md"]),
            "-o",
            str(citation_files["html"]),
        ]

        config.loggers.cli.info("Generating an HTML version of the citation boilerplate...")
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning("Could not generate CITATION.html file:\n%s", " ".join(cmd))

        # Generate LaTex file resolving citations
        cmd = [
            "pandoc",
            "-s",
            "--bibliography",
            str(load_data("boilerplate.bib")),
            "--natbib",
            str(citation_files["md"]),
            "-o",
            str(citation_files["tex"]),
        ]
        config.loggers.cli.info("Generating a LaTeX version of the citation boilerplate...")
        try:
            check_call(cmd, timeout=10)
        except (FileNotFoundError, CalledProcessError, TimeoutExpired):
            config.loggers.cli.warning("Could not generate CITATION.tex file:\n%s", " ".join(cmd))
        else:
            copyfile(load_data("boilerplate.bib"), citation_files["bib"])
