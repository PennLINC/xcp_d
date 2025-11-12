"""Integration tests for --skip parameter with real workflows.

These tests verify that the --skip parameter correctly prevents computation and output
of ALFF, ReHo, parcellation, and connectivity in complete workflow runs using real data.

Running these tests:
    # Run all skip integration tests
    pytest -m integration xcp_d/tests/test_skip_integration.py

    # Run specific dataset tests
    pytest -m ds001419_nifti xcp_d/tests/test_skip_integration.py
    pytest -m ds001419_cifti xcp_d/tests/test_skip_integration.py

Note: These tests require test datasets to be downloaded and may take significant time.
They are excluded from the default test run (see pyproject.toml pytest.ini_options).
"""

import os
from contextlib import contextmanager

import pytest

# Ensure HOME environment variable is set for cross-platform compatibility
# This is needed before importing xcp_d modules that reference config.py
if not os.getenv('HOME'):
    os.environ['HOME'] = os.path.expanduser('~')

from xcp_d.tests.utils import download_test_data, get_test_data_path


def _posix_path(path_like):
    """Return a POSIX-style string path, safe for TOML on Windows."""
    return str(path_like).replace('\\', '/')


@contextmanager
def _windows_toml_loads_patch():
    """Temporarily patch toml.loads on Windows to avoid backslash escapes.

    This is a test-local workaround to prevent "Reserved escape sequence" errors
    when computing configuration hashes or re-loading the emitted TOML config. It
    will be applied only on Windows and automatically restored afterward.
    """
    import sys as _sys

    import toml as _toml

    is_windows = _sys.platform.startswith('win') or os.name == 'nt'
    if not is_windows:
        yield
        return

    original_loads = _toml.loads

    def _safe_loads(s, _dict=dict, decoder=None):  # noqa: D401
        if isinstance(s, str):
            s = s.replace('\\', '/')
        return original_loads(s, _dict=_dict, decoder=decoder)

    _toml.loads = _safe_loads
    try:
        yield
    finally:
        _toml.loads = original_loads


def _run_xcp_d_with_skip(
    dataset_dir,
    out_dir,
    work_dir,
    skip_outputs,
    file_format='nifti',
    filter_file=None,
):
    """Helper to run xcp_d with skip parameters."""
    from xcp_d.cli.parser import parse_args
    from xcp_d.cli.workflow import build_workflow

    # Normalize paths to POSIX style (must operate on string values).
    dataset_dir = _posix_path(dataset_dir)
    out_dir = _posix_path(out_dir)
    work_dir = _posix_path(work_dir)
    if filter_file:
        filter_file = _posix_path(filter_file)

    parameters = [
        dataset_dir,
        out_dir,
        'participant',
        '--participant-label=01',
        # Use a mode with fewer mandatory parameters than 'none' to keep tests lightweight.
        # 'none' mode requires many flags (nuisance-regressors, combine-runs, etc.).
        # Selecting 'linc' avoids validation errors while still exercising skip logic.
        '--mode=linc',
        f'-w={work_dir}',
        '--dummy-scans=0',
        '--min-time=0',
        '--despike=n',
        f'--file-format={file_format}',
        '--input-type=fmriprep',
        # Minimal additional flags to satisfy parameter validation for linc mode
        '--nuisance-regressors=acompcor_gsr',
        '--combine-runs=n',
        '--linc-qc=n',
        '--abcc-qc=n',
        '--min-coverage=0.4',
        '--motion-filter-type=lp',
        '--fd-thresh=5',
        '--band-stop-min=6',
        '--smoothing=6',
        '--output-type=censored',
        '--warp-surfaces-native2std=n',
    ]

    # Ensure atlas selection is valid to avoid external AtlasPack dependency.
    # Use a minimal built-in atlas suited to the chosen file format.
    if 'parcellation' not in skip_outputs:
        if file_format == 'nifti':
            parameters.append('--atlases=Tian')
        elif file_format == 'cifti':
            parameters.append('--atlases=Glasser')

    if filter_file:
        parameters.append(f'--bids-filter-file={filter_file}')

    # Add skip parameters as a single argument with space-separated values
    if skip_outputs:
        parameters.append('--skip')
        parameters.extend(skip_outputs)

    # Apply Windows-only TOML patch across parsing and workflow build, then restore.
    with _windows_toml_loads_patch():
        parse_args(parameters)

        # Persist and then build workflow using a real config file path (dict would error)
        from xcp_d import config as _config

        config_file = _config.execution.work_dir / f'config-{_config.execution.run_uuid}.toml'
        _config.to_filename(config_file)
        build_workflow(str(config_file), retval={})


# NIfTI first: alff → reho → alff+reho → parcellation → connectivity
@pytest.mark.integration
@pytest.mark.ds001419_nifti
def test_skip_alff_nifti(data_dir, output_dir, working_dir):
    """Test skipping ALFF with NIfTI workflow."""
    test_name = 'test_skip_alff_nifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_nifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['alff'],
        file_format='nifti',
        filter_file=filter_file,
    )

    # Check that ALFF files were NOT generated
    import glob

    alff_files = glob.glob(os.path.join(out_dir, '**', '*alff*'), recursive=True)
    assert len(alff_files) == 0, f'ALFF files should not be generated: {alff_files}'


@pytest.mark.integration
@pytest.mark.ds001419_nifti
def test_skip_reho_nifti(data_dir, output_dir, working_dir):
    """Test skipping ReHo with NIfTI workflow."""
    test_name = 'test_skip_reho_nifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_nifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['reho'],
        file_format='nifti',
        filter_file=filter_file,
    )

    # Check that ReHo files were NOT generated
    import glob

    reho_files = glob.glob(os.path.join(out_dir, '**', '*reho*'), recursive=True)
    assert len(reho_files) == 0, f'ReHo files should not be generated: {reho_files}'


@pytest.mark.integration
@pytest.mark.ds001419_nifti
def test_skip_multiple_outputs_nifti(data_dir, output_dir, working_dir):
    """Test skipping multiple outputs (ALFF + ReHo) with NIfTI workflow."""
    test_name = 'test_skip_multiple_nifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_nifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['alff', 'reho'],
        file_format='nifti',
        filter_file=filter_file,
    )

    # Check that neither ALFF nor ReHo files were generated
    import glob

    alff_files = glob.glob(os.path.join(out_dir, '**', '*alff*'), recursive=True)
    reho_files = glob.glob(os.path.join(out_dir, '**', '*reho*'), recursive=True)

    assert len(alff_files) == 0, f'ALFF files should not be generated: {alff_files}'
    assert len(reho_files) == 0, f'ReHo files should not be generated: {reho_files}'


@pytest.mark.integration
@pytest.mark.ds001419_nifti
def test_skip_parcellation_nifti(data_dir, output_dir, working_dir):
    """Test skipping parcellation with NIfTI workflow (atlas suppressed)."""
    test_name = 'test_skip_parcellation_nifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_nifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['parcellation'],
        file_format='nifti',
        filter_file=filter_file,
    )

    # Parcellation outputs typically include atlas timeseries (ptseries/tsv) or matrix files.
    import glob

    parcellation_patterns = [
        '*ptseries*',
        '*pscalar*',
        '*timeseries*atlas*',
        '*timeseries*parcellation*',
        '*_connectivity*',
    ]
    found = []
    for pattern in parcellation_patterns:
        found.extend(glob.glob(os.path.join(out_dir, '**', pattern), recursive=True))
    assert len(found) == 0, f'Parcellation files should not be generated: {found}'


@pytest.mark.integration
@pytest.mark.ds001419_nifti
def test_skip_connectivity_nifti(data_dir, output_dir, working_dir):
    """Test skipping connectivity with NIfTI workflow (still computing parcellation)."""
    test_name = 'test_skip_connectivity_nifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_nifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['connectivity'],
        file_format='nifti',
        filter_file=filter_file,
    )

    import glob

    # Connectivity outputs may include correlation matrices, connectivity cifti/tsv.
    connectivity_patterns = [
        '*correlation*',
        '*connectivity*',
        '*_corr*.tsv',
        '*_connectome*',
    ]
    found = []
    for pattern in connectivity_patterns:
        found.extend(glob.glob(os.path.join(out_dir, '**', pattern), recursive=True))
    assert len(found) == 0, f'Connectivity files should not be generated: {found}'


# Then CIFTI: alff → reho → alff+reho → parcellation → connectivity
@pytest.mark.integration
@pytest.mark.ds001419_cifti
def test_skip_alff_cifti(data_dir, output_dir, working_dir):
    """Test skipping ALFF with CIFTI workflow."""
    test_name = 'test_skip_alff_cifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_cifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['alff'],
        file_format='cifti',
        filter_file=filter_file,
    )

    # Check that ALFF files were NOT generated
    import glob

    alff_files = glob.glob(os.path.join(out_dir, '**', '*alff*'), recursive=True)
    assert len(alff_files) == 0, f'ALFF files should not be generated: {alff_files}'


@pytest.mark.integration
@pytest.mark.ds001419_cifti
def test_skip_reho_cifti(data_dir, output_dir, working_dir):
    """Test skipping ReHo with CIFTI workflow."""
    test_name = 'test_skip_reho_cifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_cifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['reho'],
        file_format='cifti',
        filter_file=filter_file,
    )

    # Check that ReHo files were NOT generated
    import glob

    reho_files = glob.glob(os.path.join(out_dir, '**', '*reho*'), recursive=True)
    assert len(reho_files) == 0, f'ReHo files should not be generated: {reho_files}'


@pytest.mark.integration
@pytest.mark.ds001419_cifti
def test_skip_multiple_outputs_cifti(data_dir, output_dir, working_dir):
    """Test skipping multiple outputs (ALFF + ReHo) with CIFTI workflow."""
    test_name = 'test_skip_multiple_cifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_cifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['alff', 'reho'],
        file_format='cifti',
        filter_file=filter_file,
    )

    # Check that neither ALFF nor ReHo files were generated
    import glob

    alff_files = glob.glob(os.path.join(out_dir, '**', '*alff*'), recursive=True)
    reho_files = glob.glob(os.path.join(out_dir, '**', '*reho*'), recursive=True)

    assert len(alff_files) == 0, f'ALFF files should not be generated: {alff_files}'
    assert len(reho_files) == 0, f'ReHo files should not be generated: {reho_files}'


@pytest.mark.integration
@pytest.mark.ds001419_cifti
def test_skip_parcellation_cifti(data_dir, output_dir, working_dir):
    """Test skipping parcellation with CIFTI workflow."""
    test_name = 'test_skip_parcellation_cifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_cifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['parcellation'],
        file_format='cifti',
        filter_file=filter_file,
    )

    import glob

    parcellation_patterns = [
        '*ptseries*',
        '*pscalar*',
        '*timeseries*atlas*',
        '*timeseries*parcellation*',
        '*_connectivity*',
    ]
    found = []
    for pattern in parcellation_patterns:
        found.extend(glob.glob(os.path.join(out_dir, '**', pattern), recursive=True))
    assert len(found) == 0, f'Parcellation files should not be generated: {found}'


@pytest.mark.integration
@pytest.mark.ds001419_cifti
def test_skip_connectivity_cifti(data_dir, output_dir, working_dir):
    """Test skipping connectivity with CIFTI workflow."""
    test_name = 'test_skip_connectivity_cifti'

    dataset_dir = download_test_data('ds001419', data_dir)
    out_dir = os.path.join(output_dir, test_name)
    work_dir = os.path.join(working_dir, test_name)

    test_data_dir = get_test_data_path()
    filter_file = os.path.join(test_data_dir, 'ds001419_cifti_filter.json')

    _run_xcp_d_with_skip(
        dataset_dir=dataset_dir,
        out_dir=out_dir,
        work_dir=work_dir,
        skip_outputs=['connectivity'],
        file_format='cifti',
        filter_file=filter_file,
    )

    import glob

    connectivity_patterns = [
        '*correlation*',
        '*connectivity*',
        '*_corr*.tsv',
        '*_connectome*',
    ]
    found = []
    for pattern in connectivity_patterns:
        found.extend(glob.glob(os.path.join(out_dir, '**', pattern), recursive=True))
    assert len(found) == 0, f'Connectivity files should not be generated: {found}'
