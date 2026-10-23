"""Tests for how XCP-D derivatives are written.

The 4D denoised BOLD outputs are written in worker processes rather than on
nipype's main thread, and every derivative niworkflows rewrites or copies is
compressed with ISA-L instead of zlib.
"""

import gzip

import nibabel as nb
import numpy as np
import pytest
from niworkflows.interfaces import bids as niworkflows_bids
from niworkflows.utils.images import unsafe_write_nifti_header_and_data as zlib_writer
from niworkflows.utils.misc import _copy_any as niworkflows_copy_any

from xcp_d import config
from xcp_d.interfaces import bids
from xcp_d.tests.tests import mock_config
from xcp_d.utils import write_save


def _image(shape=(6, 5, 4, 3), dtype='float32'):
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(shape) * 100 + 1000).astype(dtype)
    return nb.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))


def test_niworkflows_sinks_write_with_isal():
    """Test that importing XCP-D's sinks routes niworkflows' writer to ISA-L."""
    assert niworkflows_bids.unsafe_write_nifti_header_and_data is write_save.write_nifti_isal


def test_niworkflows_sinks_copy_with_isal():
    """Test that importing XCP-D's sinks routes niworkflows' file copies to ISA-L."""
    assert niworkflows_bids._copy_any is write_save.copy_any_isal


@pytest.mark.parametrize('src_name', ['in.nii.gz', 'in.nii'])
def test_isal_copy_matches_the_zlib_copy(tmp_path, src_name):
    """Test that the ISA-L copy decompresses to the bytes niworkflows' copy writes."""
    _image().to_filename(tmp_path / src_name)
    niworkflows_copy_any(tmp_path / src_name, tmp_path / 'zlib.nii.gz')
    assert write_save.copy_any_isal(tmp_path / src_name, tmp_path / 'isal.nii.gz')

    with gzip.open(tmp_path / 'zlib.nii.gz') as a, gzip.open(tmp_path / 'isal.nii.gz') as b:
        assert a.read() == b.read()


def test_isal_copy_is_deterministic(tmp_path):
    """Test that the copy holds no file name or time, so two copies are byte-identical."""
    _image().to_filename(tmp_path / 'in.nii.gz')
    write_save.copy_any_isal(tmp_path / 'in.nii.gz', tmp_path / 'a.nii.gz')
    write_save.copy_any_isal(tmp_path / 'in.nii.gz', tmp_path / 'b.nii.gz')
    assert (tmp_path / 'a.nii.gz').read_bytes() == (tmp_path / 'b.nii.gz').read_bytes()
    # FLG byte: no FNAME (0x08) field
    assert not (tmp_path / 'a.nii.gz').read_bytes()[3] & 0x08


def test_isal_copy_leaves_uncompressed_outputs_to_niworkflows(tmp_path):
    """Test that an uncompressed output (e.g., a CIFTI) is copied byte for byte."""
    (tmp_path / 'in.dtseries.nii').write_bytes(b'not compressed')
    write_save.copy_any_isal(tmp_path / 'in.dtseries.nii', tmp_path / 'out.dtseries.nii')
    assert (tmp_path / 'out.dtseries.nii').read_bytes() == b'not compressed'


@pytest.mark.parametrize('dtype', ['float32', 'float64', 'int16'])
def test_isal_writer_matches_the_zlib_writer(tmp_path, dtype):
    """Test that the ISA-L file decompresses to exactly the bytes niworkflows writes."""
    img = _image(dtype=dtype)
    data = np.asanyarray(img.dataobj)
    zlib_writer(tmp_path / 'zlib.nii.gz', img.header, data)
    write_save.write_nifti_isal(tmp_path / 'isal.nii.gz', img.header, data)

    with gzip.open(tmp_path / 'zlib.nii.gz') as a, gzip.open(tmp_path / 'isal.nii.gz') as b:
        assert a.read() == b.read()
    out = nb.load(tmp_path / 'isal.nii.gz')
    assert out.get_data_dtype() == np.dtype(dtype)
    assert np.array_equal(np.asanyarray(out.dataobj), data)


def test_isal_writer_is_deterministic(tmp_path):
    """Test that two writes of one image are byte-identical (no name or time in the header)."""
    img = _image()
    data = np.asanyarray(img.dataobj)
    write_save.write_nifti_isal(tmp_path / 'a.nii.gz', img.header, data)
    write_save.write_nifti_isal(tmp_path / 'b.nii.gz', img.header, data)
    assert (tmp_path / 'a.nii.gz').read_bytes() == (tmp_path / 'b.nii.gz').read_bytes()


def test_isal_writer_leaves_uncompressed_files_uncompressed(tmp_path):
    """Test that a .nii output is written without compression."""
    img = _image()
    data = np.asanyarray(img.dataobj)
    write_save.write_nifti_isal(tmp_path / 'x.nii', img.header, data)
    assert (tmp_path / 'x.nii').read_bytes()[:2] != b'\x1f\x8b'
    assert np.array_equal(np.asanyarray(nb.load(tmp_path / 'x.nii').dataobj), data)


def test_datasink_compresses_a_4d_series(tmp_path):
    """Test a datasink run end to end: an uncompressed input written as .nii.gz."""
    img = _image()
    img.to_filename(tmp_path / 'in.nii')
    result = bids.DerivativesDataSink(
        base_directory=str(tmp_path / 'out'),
        source_file='/data/sub-01/func/sub-01_task-rest_bold.nii.gz',
        space='MNI152NLin2009cAsym',
        desc='denoised',
        suffix='bold',
        extension='.nii.gz',
        compress=True,
        in_file=str(tmp_path / 'in.nii'),
    ).run()
    out = result.outputs.out_file
    out = out[0] if isinstance(out, list) else out
    assert out.endswith('.nii.gz')
    assert np.array_equal(np.asanyarray(nb.load(out).dataobj), np.asanyarray(img.dataobj))


def _sink_denoised_bold(in_file, out_dir):
    out_dir.mkdir()
    return bids.DerivativesDataSink(
        base_directory=str(out_dir),
        source_file='/data/sub-01/func/sub-01_task-rest_space-MNI152NLin2009cAsym_bold.nii.gz',
        desc='denoised',
        extension='.nii.gz',
        in_file=str(in_file),
    ).run(cwd=str(out_dir))


def test_denoised_bold_sink_copies_denoise_nifti_output(tmp_path):
    """Test that the sink copies the denoised BOLD instead of rewriting it.

    If DenoiseNifti's header differs from the one DerivativesDataSink expects
    (transform codes, units), the sink decompresses and recompresses the whole
    series.
    """
    import pandas as pd

    from xcp_d.interfaces.censoring import Censor
    from xcp_d.interfaces.nilearn import DenoiseNifti

    affine = np.array([[2, 0, 0, -96], [0, 2, 0, -132], [0, 0, 2, -78], [0, 0, 0, 1.0]])
    mask = nb.Nifti1Image(np.ones((6, 5, 4), np.uint8), affine)
    # fMRIPrep writes standard-space masks with transform codes of 4.
    mask.header.set_qform(affine, 4)
    mask.header.set_sform(affine, 4)
    mask.to_filename(tmp_path / 'mask.nii.gz')
    rng = np.random.default_rng(0)
    bold = rng.standard_normal((6, 5, 4, 30)).astype(np.float32) * 100 + 1000
    nb.Nifti1Image(bold, affine).to_filename(tmp_path / 'bold.nii.gz')
    temporal_mask = np.zeros(30, int)
    temporal_mask[[5, 6]] = 1
    pd.DataFrame({'denoising': temporal_mask}).to_csv(
        tmp_path / 'outliers.tsv', sep='\t', index=False
    )

    denoised = (
        DenoiseNifti(
            preprocessed_bold=str(tmp_path / 'bold.nii.gz'),
            mask=str(tmp_path / 'mask.nii.gz'),
            temporal_mask=str(tmp_path / 'outliers.tsv'),
            confounds_tsv=None,
            TR=0.8,
            bandpass_filter=True,
            low_pass=0.08,
            high_pass=0.01,
            filter_order=2,
        )
        .run(cwd=str(tmp_path))
        .outputs.denoised_interpolated_bold
    )
    censored = (
        Censor(in_file=denoised, temporal_mask=str(tmp_path / 'outliers.tsv'), column='denoising')
        .run(cwd=str(tmp_path))
        .outputs.out_file
    )

    for label, in_file in (('interpolated', denoised), ('censored', censored)):
        result = _sink_denoised_bold(in_file, tmp_path / label)
        assert result.outputs.fixed_hdr == [False]
        out_file = result.outputs.out_file
        out_file = out_file[0] if isinstance(out_file, list) else out_file
        with gzip.open(in_file) as a, gzip.open(out_file) as b:
            assert a.read() == b.read()

    hdr = nb.load(censored).header
    assert (int(hdr['qform_code']), int(hdr['sform_code'])) == (1, 1)
    assert hdr.get_xyzt_units() == ('mm', 'sec')
    assert hdr.get_zooms()[3] == pytest.approx(0.8)


@pytest.mark.parametrize('file_format', ['nifti', 'cifti'])
def test_4d_bold_sinks_run_in_workers(file_format):
    """Test that the 4D denoised BOLD sinks are submitted to workers.

    A sink with run_without_submitting=True runs in nipype's main process, which
    dispatches no other node until it returns; writing a 4D series there stalls
    the whole run. The small sinks stay on the main thread.
    """
    from xcp_d.workflows.bold.outputs import init_postproc_derivatives_wf

    with mock_config():
        config.workflow.file_format = file_format
        config.workflow.smoothing = 6
        config.workflow.fd_thresh = 0.3
        config.execution.atlases = []

        wf = init_postproc_derivatives_wf(
            name_source='/data/sub-01/func/sub-01_task-rest_bold.nii.gz',
            source_metadata={},
            has_multiple_runs=False,
        )
        nodes = {node.name: node for node in wf._graph.nodes()}
        assert not nodes['ds_denoised_bold'].run_without_submitting
        assert not nodes['ds_smoothed_bold'].run_without_submitting
        assert nodes['ds_motion'].run_without_submitting
        assert nodes['ds_temporal_mask'].run_without_submitting


@pytest.mark.parametrize('file_format', ['nifti', 'cifti'])
def test_concatenated_bold_sinks_run_in_workers(file_format):
    """Test that the concatenated 4D BOLD sinks are submitted to workers."""
    from xcp_d.workflows.bold.concatenation import init_concatenate_data_wf

    with mock_config():
        config.workflow.file_format = file_format
        config.workflow.smoothing = 6
        config.workflow.fd_thresh = 0.3
        config.workflow.combine_runs = True
        config.execution.atlases = []

        wf = init_concatenate_data_wf(
            TR=2.0,
            head_radius=50,
            exact_scans=[],
            mem_gb={'bold': 1.0, 'volume': 0.01},
        )
        nodes = {node.name: node for node in wf._graph.nodes()}
        assert not nodes['ds_denoised_bold'].run_without_submitting
        assert not nodes['ds_smoothed_denoised_bold'].run_without_submitting
        assert nodes['ds_motion_file'].run_without_submitting
