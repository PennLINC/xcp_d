"""Tests for resource-monitor callback log parsing."""

import json

import pytest

from xcp_d.utils.resource_monitor import _safe_log_nodes_cb, summarize_callback_log


def test_summarize_callback_log(tmp_path):
    """Ensure callback log rows are aggregated by node ID."""
    callback_log = tmp_path / 'resource_monitor.jsonl'
    rows = [
        {'name': 'resample_node', 'id': 'wf.resample_node', 'estimated_memory_gb': 2},
        {
            'name': 'resample_node',
            'id': 'wf.resample_node',
            'estimated_memory_gb': '2',
            'runtime_memory_gb': '1.25',
            'runtime_pid': 101,
        },
        {'name': 'denoise_node', 'id': 'wf.denoise_node', 'estimated_memory_gb': 1},
        {
            'name': 'denoise_node',
            'id': 'wf.denoise_node',
            'estimated_memory_gb': '1',
            'runtime_memory_gb': '2.5',
            'runtime_pid': 202,
        },
    ]
    callback_log.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')

    summary = summarize_callback_log(callback_log)

    assert len(summary) == 2
    assert summary[0]['node_name'] == 'denoise_node'
    assert summary[0]['node_id'] == 'wf.denoise_node'
    assert summary[0]['estimated_memory_gb'] == 1.0
    assert summary[0]['runtime_memory_gb'] == 2.5
    assert summary[0]['delta_memory_gb'] == 1.5
    assert summary[0]['runtime_vs_estimated_ratio'] == 2.5
    assert summary[0]['runtime_pids'] == '202'
    assert summary[0]['n_samples'] == 1

    assert summary[1]['node_name'] == 'resample_node'
    assert summary[1]['node_id'] == 'wf.resample_node'
    assert summary[1]['estimated_memory_gb'] == 2.0
    assert summary[1]['runtime_memory_gb'] == 1.25
    assert summary[1]['delta_memory_gb'] == -0.75
    assert summary[1]['runtime_pids'] == '101'


def test_summarize_callback_log_groups_by_node_id(tmp_path):
    """Rows with identical names but different IDs should remain separate."""
    callback_log = tmp_path / 'resource_monitor.jsonl'
    rows = [
        {
            'name': 'mapnode',
            'id': '_mapnode0',
            'estimated_memory_gb': '1',
            'runtime_memory_gb': '1.5',
            'runtime_pid': 11,
        },
        {
            'name': 'mapnode',
            'id': '_mapnode1',
            'estimated_memory_gb': '1',
            'runtime_memory_gb': '2.5',
            'runtime_pid': 22,
        },
    ]
    callback_log.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')

    summary = summarize_callback_log(callback_log)

    assert len(summary) == 2
    assert summary[0]['node_name'] == 'mapnode'
    assert summary[0]['node_id'] == '_mapnode1'
    assert summary[0]['runtime_memory_gb'] == 2.5
    assert summary[0]['runtime_pids'] == '22'
    assert summary[1]['node_name'] == 'mapnode'
    assert summary[1]['node_id'] == '_mapnode0'
    assert summary[1]['runtime_memory_gb'] == 1.5
    assert summary[1]['runtime_pids'] == '11'


def test_summarize_callback_log_missing_file(tmp_path):
    """Missing callback logs should raise a clear error."""
    callback_log = tmp_path / 'missing.jsonl'
    with pytest.raises(FileNotFoundError, match='Callback log does not exist'):
        summarize_callback_log(callback_log)


def test_safe_log_nodes_cb_skips_runtime_lists(caplog):
    """MapNode aggregate runtime lists should be ignored in callback logging."""

    class Runtime:
        startTime = '2026-01-01T00:00:00'
        endTime = '2026-01-01T00:00:01'
        duration = 1.0
        cpu_percent = 0.0
        mem_peak_gb = 1.0
        pid = 111

    class Result:
        runtime = [Runtime()]

    class Node:
        name = 'mapnode_parent'
        _id = 'mapnode_parent'
        mem_gb = 1
        n_procs = 1
        result = Result()

    with caplog.at_level('DEBUG', logger='callback'):
        _safe_log_nodes_cb(Node(), 'end')

    assert not caplog.records
