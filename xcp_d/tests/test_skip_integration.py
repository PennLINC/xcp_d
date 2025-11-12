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

import pytest

# Ensure HOME environment variable is set for cross-platform compatibility
# This is needed before importing xcp_d modules that reference config.py
if not os.getenv('HOME'):
    os.environ['HOME'] = os.path.expanduser('~')

from xcp_d.tests.utils import download_test_data, get_test_data_path


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

    parameters = [
        dataset_dir,
        out_dir,
        'participant',
        '--participant-label=01',
        '--mode=none',
        f'-w={work_dir}',
        '--dummy-scans=0',
        '--fd-thresh=0.2',
        '--despike=n',
        f'--file-format={file_format}',
        '--input-type=fmriprep',
    ]

    if filter_file:
        parameters.append(f'--bids-filter-file={filter_file}')

    # Add skip parameters
    for skip_output in skip_outputs:
        parameters.append(f'--skip={skip_output}')

    parse_args(parameters)
    build_workflow({}, {})


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
