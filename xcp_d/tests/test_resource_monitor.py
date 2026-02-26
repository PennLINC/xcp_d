"""Tests for resource-monitor callback log parsing."""

import json

import pytest

from xcp_d.utils.resource_monitor import summarize_callback_log


def test_summarize_callback_log(tmp_path):
    """Ensure callback log rows are aggregated by node name."""
    callback_log = tmp_path / 'resource_monitor.jsonl'
    rows = [
        {'name': 'resample_node', 'id': 'wf.resample_node', 'estimated_memory_gb': 2},
        {
            'name': 'resample_node',
            'id': 'wf.resample_node',
            'estimated_memory_gb': '2',
            'runtime_memory_gb': '1.25',
        },
        {'name': 'denoise_node', 'id': 'wf.denoise_node', 'estimated_memory_gb': 1},
        {
            'name': 'denoise_node',
            'id': 'wf.denoise_node',
            'estimated_memory_gb': '1',
            'runtime_memory_gb': '2.5',
        },
    ]
    callback_log.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')

    summary = summarize_callback_log(callback_log)

    assert len(summary) == 2
    assert summary[0]['node_name'] == 'denoise_node'
    assert summary[0]['estimated_memory_gb'] == 1.0
    assert summary[0]['runtime_memory_gb'] == 2.5
    assert summary[0]['delta_memory_gb'] == 1.5
    assert summary[0]['runtime_vs_estimated_ratio'] == 2.5
    assert summary[0]['n_samples'] == 1

    assert summary[1]['node_name'] == 'resample_node'
    assert summary[1]['estimated_memory_gb'] == 2.0
    assert summary[1]['runtime_memory_gb'] == 1.25
    assert summary[1]['delta_memory_gb'] == -0.75


def test_summarize_callback_log_missing_file(tmp_path):
    """Missing callback logs should raise a clear error."""
    callback_log = tmp_path / 'missing.jsonl'
    with pytest.raises(FileNotFoundError, match='Callback log does not exist'):
        summarize_callback_log(callback_log)
