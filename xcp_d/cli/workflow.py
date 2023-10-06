"""The main XCP-D workflow."""


def build_workflow(config_file, retval):
    """Create the Nipype workflow that supports the whole execution graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows fmriprep to enforce
    a hard-limited memory-scope.
    """
    import os
    from pathlib import Path

    from niworkflows.reports.core import generate_reports
    from niworkflows.utils.bids import check_pipeline_version, collect_participants
    from niworkflows.utils.misc import check_valid_fs_license

    from xcp_d import config
    from xcp_d.utils.misc import check_deps
    from xcp_d.workflows.base import init_xcpd_wf

    config.load(config_file)
    build_log = config.loggers.workflow

    output_dir = config.execution.output_dir
    version = config.environment.version

    retval["return_code"] = 1
    retval["workflow"] = None

    # warn if older results exist: check for dataset_description.json in output folder
    msg = check_pipeline_version(version, output_dir / "xcp_d" / "dataset_description.json")
    if msg is not None:
        build_log.warning(msg)

    # Please note this is the input folder's dataset_description.json
    dset_desc_path = config.execution.fmri_dir / "dataset_description.json"
    if dset_desc_path.exists():
        from hashlib import sha256

        desc_content = dset_desc_path.read_bytes()
        config.execution.bids_description_hash = sha256(desc_content).hexdigest()

    # First check that fmri_dir looks like a BIDS folder
    if config.workflow.input_type in ("dcan", "hcp"):
        if config.workflow.input_type == "dcan":
            from xcp_d.utils.dcan2fmriprep import convert_dcan2bids as convert_to_bids
        elif config.workflow.input_type == "hcp":
            from xcp_d.utils.hcp2fmriprep import convert_hcp2bids as convert_to_bids

        config.loggers.cli.info(f"Converting {config.workflow.input_type} to fmriprep format")
        converted_fmri_dir = os.path.join(
            config.execution.work_dir,
            f"dset_bids/derivatives/{config.workflow.input_type}",
        )
        os.makedirs(converted_fmri_dir, exist_ok=True)

        convert_to_bids(
            config.execution.fmri_dir,
            out_dir=converted_fmri_dir,
            participant_ids=config.execution.participant_label,
        )

        config.execution.fmri_dir = Path(converted_fmri_dir)

    if not os.path.isfile((os.path.join(config.execution.fmri_dir, "dataset_description.json"))):
        build_log.error(
            "No dataset_description.json file found in input directory. "
            "Make sure to point to the specific pipeline's derivatives folder. "
            "For example, use '/dset/derivatives/fmriprep', not /dset/derivatives'."
        )
        retval["return_code"] = 1

    subject_list = collect_participants(
        config.execution.layout,
        participant_label=config.execution.participant_label,
    )

    # Called with reports only
    if config.execution.reports_only:
        from pkg_resources import resource_filename as pkgrf

        build_log.log(25, f"Running --reports-only on participants {', '.join(subject_list)}")
        retval["return_code"] = generate_reports(
            subject_list,
            config.execution.output_dir,
            config.execution.run_uuid,
            config=pkgrf("xcp_d", "data/reports-spec.yml"),
            packagename="xcp_d",
        )
        return retval

    # Build main workflow
    init_msg = f"""
    Running XCP-D version {config.environment.version}:
      * BIDS dataset path: {config.execution.fmri_dir}.
      * Participant list: {subject_list}.
      * Run identifier: {config.execution.run_uuid}.
      * Output spaces: {config.execution.output_spaces}."""

    if config.execution.anat_derivatives:
        init_msg += f"""
      * Anatomical derivatives: {config.execution.anat_derivatives}."""
    build_log.log(25, init_msg)

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
    logs_path = config.execution.output_dir / "xcp_d" / "logs"
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

        from pkg_resources import resource_filename as pkgrf

        # Generate HTML file resolving citations
        cmd = [
            "pandoc",
            "-s",
            "--bibliography",
            pkgrf("xcp_d", "data/boilerplate.bib"),
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
            pkgrf("xcp_d", "data/boilerplate.bib"),
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
            copyfile(pkgrf("xcp_d", "data/boilerplate.bib"), citation_files["bib"])
