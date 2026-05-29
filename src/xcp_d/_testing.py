"""Utilities for testing and documentation building."""

import os
import shutil
import subprocess
import tarfile
from contextlib import contextmanager
from glob import glob
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from tempfile import mkdtemp

import nibabel as nb
import numpy as np
import requests
from bids.layout import BIDSLayout
from nipype import logging
from toml import loads

from xcp_d.data import load as load_data
from xcp_d.utils import doc

LOGGER = logging.getLogger('nipype.utils')


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


def download_test_data(dset, data_dir):
    """Download test data."""
    URLS = {
        'fmriprepwithoutfreesurfer': (
            'https://upenn.box.com/shared/static/seyp1cu9w5v3ds6iink37hlsa217yge1.tar.gz'
        ),
        'nibabies': 'https://upenn.box.com/shared/static/rsd7vpny5imv3qkd7kpuvdy9scpnfpe2.tar.gz',
        'ds001419': 'https://upenn.box.com/shared/static/yye7ljcdodj9gd6hm2r6yzach1o6xq1d.tar.gz',
        'ds001419-aroma': (
            'https://upenn.box.com/shared/static/dexcmnlj7yujudr3muu05kch66sko4mt.tar.gz'
        ),
        'pnc': 'https://upenn.box.com/shared/static/ui2847ys49d82pgn5ewai1mowcmsv2br.tar.gz',
        'ukbiobank': 'https://upenn.box.com/shared/static/p5h1eg4p5cd2ef9ehhljlyh1uku0xe97.tar.gz',
        'schaefer100': (
            'https://upenn.box.com/shared/static/b9pn9qebr41kteant4ym2q5u4kcbgiy6.tar.gz'
        ),
    }
    if dset == '*':
        for k in URLS:
            download_test_data(k, data_dir=data_dir)
        return

    if dset not in URLS:
        raise ValueError(f'dset ({dset}) must be one of: {", ".join(URLS.keys())}')

    out_dir = os.path.join(data_dir, dset)

    if os.path.isdir(out_dir):
        LOGGER.info(
            f'Dataset {dset} already exists. '
            'If you need to re-download the data, please delete the folder.'
        )
        if dset.startswith('ds001419'):
            out_dir = os.path.join(out_dir, dset)
        return out_dir

    LOGGER.info(f'Downloading {dset} to {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    with requests.get(URLS[dset], stream=True, timeout=10) as req:
        with tarfile.open(fileobj=GzipFile(fileobj=BytesIO(req.content))) as t:
            t.extractall(out_dir)  # noqa: S202

    if dset.startswith('ds001419'):
        out_dir = os.path.join(out_dir, dset)

    return out_dir


def check_generated_files(output_dir, output_list_file):
    """Compare files generated by xcp_d with a list of expected files."""
    found_files = sorted(glob(os.path.join(output_dir, '**/*'), recursive=True))
    found_files = [os.path.relpath(f, output_dir) for f in found_files]
    found_files = [f for f in found_files if 'figures' not in f]
    found_files = [f for f in found_files if 'log' not in f.split(os.path.sep)]

    with open(output_list_file) as fo:
        expected_files = [f.rstrip() for f in fo]

    if sorted(found_files) != sorted(expected_files):
        expected_not_found = sorted(set(expected_files) - set(found_files))
        found_not_expected = sorted(set(found_files) - set(expected_files))

        msg = ''
        if expected_not_found:
            msg += '\nExpected but not found:\n\t'
            msg += '\n\t'.join(expected_not_found)
        if found_not_expected:
            msg += '\nFound but not expected:\n\t'
            msg += '\n\t'.join(found_not_expected)
        raise ValueError(msg)


def check_affines(data_dir, out_dir, input_type):
    """Confirm affines don't change across XCP-D runs."""
    preproc_layout = BIDSLayout(str(data_dir), validate=False)
    xcp_layout = BIDSLayout(str(out_dir), validate=False)
    if input_type == 'cifti':
        denoised_files = xcp_layout.get(
            invalid_filters='allow',
            datatype='func',
            extension='.dtseries.nii',
        )
        space = denoised_files[0].get_entities()['space']
        preproc_files = preproc_layout.get(
            invalid_filters='allow',
            datatype='func',
            space=space,
            extension='.dtseries.nii',
        )
    elif input_type in ('nifti', 'ukb'):
        denoised_files = xcp_layout.get(
            datatype='func',
            suffix='bold',
            extension='.nii.gz',
        )
        space = denoised_files[0].get_entities()['space']
        preproc_files = preproc_layout.get(
            invalid_filters='allow',
            datatype='func',
            space=space,
            suffix='bold',
            extension='.nii.gz',
        )
    else:  # Nibabies
        denoised_files = xcp_layout.get(
            datatype='func',
            space='MNIInfant',
            suffix='bold',
            extension='.nii.gz',
        )
        preproc_files = preproc_layout.get(
            invalid_filters='allow',
            datatype='func',
            space='MNIInfant',
            suffix='bold',
            extension='.nii.gz',
        )

    img1 = nb.load(preproc_files[0].path)
    img2 = nb.load(denoised_files[0].path)

    if input_type == 'cifti':
        if img1._nifti_header.get_intent() != img2._nifti_header.get_intent():
            i1, i2 = img1._nifti_header.get_intent(), img2._nifti_header.get_intent()
            raise ValueError(f'Intent mismatch: {i1} != {i2}')
        np.testing.assert_array_equal(img1.nifti_header.get_zooms(), img2.nifti_header.get_zooms())
    else:
        np.testing.assert_array_equal(img1.affine, img2.affine)
        if input_type != 'ukb':
            # The UK Biobank test dataset has the wrong TR in the header.
            np.testing.assert_array_equal(img1.header.get_zooms(), img2.header.get_zooms())


def run_command(command, env=None):
    """Run a shell command with optional environment overrides."""
    merged_env = os.environ
    if env:
        merged_env.update(env)

    process = subprocess.Popen(
        command.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        env=merged_env,
    )
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() is not None:
            break

    if process.returncode != 0:
        raise RuntimeError(
            f'Non zero return code: {process.returncode}\n{command}\n\n{process.stdout.read()}'
        )


def reorder_expected_outputs(test_data_path):
    """Sort each expected-output file alphabetically.

    Call this manually after modifying expected test outputs.
    Pass ``test_data_path`` as the path to ``test/data/``.
    """
    expected_output_files = sorted(glob(os.path.join(test_data_path, 'test_*_outputs.txt')))
    for expected_output_file in expected_output_files:
        LOGGER.info(f'Sorting {expected_output_file}')
        with open(expected_output_file) as fo:
            file_contents = fo.readlines()
        file_contents = sorted(set(file_contents))
        with open(expected_output_file, 'w') as fo:
            fo.writelines(file_contents)


def list_files(startpath):
    """Return a tree-formatted string of all files under ``startpath``."""
    tree = ''
    for root, _, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        tree += f'{indent}{os.path.basename(root)}/\n'
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            tree += f'{subindent}{f}\n'
    return tree
