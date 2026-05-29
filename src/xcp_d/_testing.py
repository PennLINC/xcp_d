"""Utilities for testing and documentation building."""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkdtemp

from toml import loads

from xcp_d.data import load as load_data
from xcp_d.utils import doc


def _check_arg_specified(argname, arglist):
    for arg in arglist:
        if arg.startswith(argname):
            return True
    return False


def get_cpu_count(max_cpus=4):
    """Figure out how many CPUs are available in the test environment."""
    env_cpus = os.getenv('CIRCLE_CPUS')
    if env_cpus:
        return int(env_cpus)
    return max_cpus


def update_resources(parameters):
    """Add ``--nthreads`` and ``--omp-nthreads`` to a parameter list if absent.

    Reads ``CIRCLE_CPUS`` from the environment (set per-job in CircleCI config)
    and falls back to ``max_cpus`` when the variable is unset.
    """
    nthreads = get_cpu_count()
    if not _check_arg_specified('--nthreads', parameters):
        parameters.append(f'--nthreads={nthreads}')
    if not _check_arg_specified('--omp-nthreads', parameters):
        parameters.append(f'--omp-nthreads={nthreads}')
    return parameters


def get_nodes(wf_results):
    """Return a ``{fullname: node}`` dict from a Nipype workflow result."""
    return {node.fullname: node for node in wf_results.nodes}


@contextmanager
def chdir(path):
    """Context manager: temporarily change the working directory.

    Taken from https://stackoverflow.com/a/37996581/2589328.
    """
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


@contextmanager
def modified_environ(*remove, **update):
    """Context manager: temporarily update ``os.environ`` in-place.

    Parameters
    ----------
    *remove
        Environment variables to remove for the duration of the block.
    **update
        Environment variables to add or update for the duration of the block.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    update_after = {k: env[k] for k in stomped}
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]


@contextmanager
def mock_config():
    """Context manager: set up a mock xcp_d config for tests and doc builds."""
    from xcp_d import config

    _old_fs = os.getenv('FREESURFER_HOME')
    if not _old_fs:
        os.environ['FREESURFER_HOME'] = mkdtemp()

    filename = load_data('tests/config.toml').resolve()
    if not filename.exists():
        base_path = os.path.dirname(filename)
        raise FileNotFoundError(
            f'File not found: {filename}\nFiles in {base_path}:\n{os.listdir(base_path)}'
        )

    settings = loads(filename.read_text())
    for sectionname, configs in settings.items():
        if sectionname != 'environment':
            section = getattr(config, sectionname)
            section.load(configs, init=False)

    config.nipype.omp_nthreads = 1
    config.nipype.init()
    config.loggers.init()

    config.execution.work_dir = Path(mkdtemp())
    config.execution.fmri_dir = Path(doc.download_example_data(out_dir=mkdtemp()))
    config.execution.output_dir = Path(mkdtemp())
    config.execution.bids_database_dir = None
    config.execution._layout = None
    config.execution.init()

    yield

    shutil.rmtree(config.execution.work_dir)
    shutil.rmtree(config.execution.output_dir)

    if not _old_fs:
        del os.environ['FREESURFER_HOME']
