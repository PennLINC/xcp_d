"""The main XCP-D workflow."""


def build_workflow(config_file, retval):
    """Create the Nipype workflow that supports the whole execution graph, given the inputs.

    All the checks and the construction of the workflow are done
    inside this function that has pickleable inputs and output
    dictionary (``retval``) to allow isolation using a
    ``multiprocessing.Process`` that allows fmriprep to enforce
    a hard-limited memory-scope.
    """
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

    if opts.clean_workdir:
        from niworkflows.utils.misc import clean_directory

        build_log.info(f"Clearing previous xcp_d working directory: {opts.work_dir}")
        if not clean_directory(opts.work_dir):
            build_log.warning(
                f"Could not clear all contents of working directory: {opts.work_dir}"
            )

    retval["return_code"] = 1
    retval["workflow"] = None
    retval["fmri_dir"] = str(opts.fmri_dir)
    retval["output_dir"] = str(opts.output_dir)
    retval["work_dir"] = str(opts.work_dir)

    # First check that fmriprep_dir looks like a BIDS folder
    if opts.input_type in ("dcan", "hcp"):
        if opts.input_type == "dcan":
            from xcp_d.utils.dcan2fmriprep import convert_dcan2bids as convert_to_bids
        elif opts.input_type == "hcp":
            from xcp_d.utils.hcp2fmriprep import convert_hcp2bids as convert_to_bids

        NIWORKFLOWS_LOG.info(f"Converting {opts.input_type} to fmriprep format")
        converted_fmri_dir = os.path.join(
            opts.work_dir,
            f"dset_bids/derivatives/{opts.input_type}",
        )
        os.makedirs(converted_fmri_dir, exist_ok=True)

        convert_to_bids(
            opts.fmri_dir,
            out_dir=converted_fmri_dir,
            participant_ids=opts.participant_label,
        )

        opts.fmri_dir = Path(converted_fmri_dir)

    if not os.path.isfile((os.path.join(opts.fmri_dir, "dataset_description.json"))):
        build_log.error(
            "No dataset_description.json file found in input directory. "
            "Make sure to point to the specific pipeline's derivatives folder. "
            "For example, use '/dset/derivatives/fmriprep', not /dset/derivatives'."
        )
        retval["return_code"] = 1

    # Set up some instrumental utilities
    run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4()}"
    retval["run_uuid"] = run_uuid

    layout = BIDSLayout(str(opts.fmri_dir), validate=False, derivatives=True)
    subject_list = collect_participants(layout, participant_label=opts.participant_label)
    retval["subject_list"] = subject_list

    # Load base plugin_settings from file if --use-plugin
    if opts.use_plugin is not None:
        from yaml import load as loadyml

        with open(opts.use_plugin) as f:
            plugin_settings = loadyml(f)

        plugin_settings.setdefault("plugin_args", {})

    else:
        # Defaults
        plugin_settings = {
            "plugin": "MultiProc",
            "plugin_args": {
                "raise_insufficient": False,
                "maxtasksperchild": 1,
            },
        }

    # Permit overriding plugin config with specific CLI options
    nthreads = opts.nthreads
    omp_nthreads = opts.omp_nthreads

    if (nthreads == 1) or (omp_nthreads > nthreads):
        omp_nthreads = 1

    plugin_settings["plugin_args"]["n_procs"] = nthreads

    if 1 < nthreads < omp_nthreads:
        build_log.warning(
            f"Per-process threads (--omp-nthreads={omp_nthreads}) exceed total "
            f"threads (--nthreads/--n_cpus={nthreads})"
        )

    if opts.mem_gb:
        plugin_settings["plugin_args"]["memory_gb"] = opts.mem_gb

    retval["plugin_settings"] = plugin_settings

    # Set up directories
    log_dir = opts.output_dir / "xcp_d" / "logs"

    # Check and create output and working directories
    opts.output_dir.mkdir(exist_ok=True, parents=True)
    opts.work_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    # Nipype config (logs and execution)
    ncfg.update_config(
        {
            "logging": {
                "log_directory": str(log_dir),
                "log_to_file": True,
                "workflow_level": log_level,
                "interface_level": log_level,
                "utils_level": log_level,
            },
            "execution": {
                "crashdump_dir": str(log_dir),
                "crashfile_format": "txt",
                "get_linked_libs": False,
            },
            "monitoring": {
                "enabled": opts.resource_monitor,
                "sample_frequency": "0.5",
                "summary_append": True,
            },
        }
    )

    if opts.resource_monitor:
        ncfg.enable_resource_monitor()

    # Build main workflow
    build_log.log(
        25,
        f"""\
Running xcp_d version {config.environment.version}:
    * fMRI directory path: {opts.fmri_dir}.
    * Participant list: {subject_list}.
    * Run identifier: {run_uuid}.

""",
    )

    retval["workflow"] = init_xcpd_wf(
        subject_list=subject_list,
        name="xcpd_wf",
    )

    boilerplate = retval["workflow"].visit_desc()

    if boilerplate:
        citation_files = {ext: log_dir / f"CITATION.{ext}" for ext in ("bib", "tex", "md", "html")}
        # To please git-annex users and also to guarantee consistency among different renderings
        # of the same file, first remove any existing ones
        for citation_file in citation_files.values():
            try:
                citation_file.unlink()
            except FileNotFoundError:
                pass

        citation_files["md"].write_text(boilerplate)

    build_log.log(
        25,
        (
            "Works derived from this xcp_d execution should include the following boilerplate:\n\n"
            f"{boilerplate}"
        ),
    )

    retval["return_code"] = 0

    return retval