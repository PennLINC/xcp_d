"""Helpers for parsing Nipype callback resource logs."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path


def _as_float(value):
    """Convert numeric-like callback values to float."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_log_nodes_cb(node, status):
    """Robust status callback that logs node run statistics to the callback logger.

    Wraps Nipype's log_nodes_cb to handle edge cases where node or node.result.runtime
    may be a list (e.g. MapNode subnodes), which causes "'list' object has no
    attribute 'startTime'" in the upstream implementation.

    Parameters
    ----------
    node : :obj:`Node` or :obj:`list`
        The node(s) being logged. If a list, logs each item.
    status : :obj:`str`
        'start', 'end', or 'exception'. Only 'end' produces log output.
    """
    if status != 'end':
        return

    # Handle plugin passing a list of nodes (e.g. MapNode expansion edge case)
    if isinstance(node, (list, tuple)):
        for n in node:
            _safe_log_nodes_cb(n, status)
        return

    try:
        result = getattr(node, 'result', None)
        if result is None:
            return

        runtime = getattr(result, 'runtime', None)
        if runtime is None:
            return

        # node.result.runtime can be a list for MapNode aggregated results
        runtimes = runtime if isinstance(runtime, (list, tuple)) else [runtime]

        for rt in runtimes:
            if rt is None:
                continue
            start_time = getattr(rt, 'startTime', None)
            end_time = getattr(rt, 'endTime', None)
            if start_time is None or end_time is None:
                continue

            status_dict = {
                'name': getattr(node, 'name', None),
                'id': getattr(node, '_id', None),
                'start': start_time,
                'finish': end_time,
                'duration': getattr(rt, 'duration', None),
                'runtime_threads': getattr(rt, 'cpu_percent', 'N/A'),
                'runtime_memory_gb': getattr(rt, 'mem_peak_gb', 'N/A'),
                'estimated_memory_gb': getattr(node, 'mem_gb', 'N/A'),
                'num_threads': getattr(node, 'n_procs', 'N/A'),
            }
            logging.getLogger('callback').debug(json.dumps(status_dict))
    except Exception:
        # Avoid crashing the workflow; log and continue
        logging.getLogger('callback').debug(
            json.dumps({'error': True, 'name': getattr(node, 'name', None), 'id': getattr(node, '_id', None)})
        )


def summarize_callback_log(callback_log):
    """Summarize estimated and runtime memory usage from a callback log.

    Parameters
    ----------
    callback_log : :obj:`str` or :obj:`pathlib.Path`
        Path to a JSON-lines callback log written by Nipype's ``log_nodes_cb``.

    Returns
    -------
    list[dict]
        One dictionary per node name with aggregated memory statistics:
        ``estimated_memory_gb``, ``runtime_memory_gb``, ``delta_memory_gb``,
        ``runtime_vs_estimated_ratio`` and ``n_samples``.
    """
    callback_log = Path(callback_log)
    if not callback_log.is_file():
        raise FileNotFoundError(f'Callback log does not exist: {callback_log}')

    grouped = defaultdict(
        lambda: {
            'estimated_memory_gb': 0.0,
            'runtime_memory_gb': 0.0,
            'n_samples': 0,
        }
    )

    with callback_log.open() as fobj:
        for line in fobj:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            runtime_memory = _as_float(row.get('runtime_memory_gb'))
            if runtime_memory is None:
                # Ignore "start" events and malformed rows.
                continue

            node_name = row.get('name') or row.get('id')
            if node_name is None:
                continue

            estimated_memory = _as_float(row.get('estimated_memory_gb')) or 0.0
            stats = grouped[node_name]
            stats['estimated_memory_gb'] = max(stats['estimated_memory_gb'], estimated_memory)
            stats['runtime_memory_gb'] = max(stats['runtime_memory_gb'], runtime_memory)
            stats['n_samples'] += 1

    summary = []
    for node_name, stats in grouped.items():
        estimated_memory = stats['estimated_memory_gb']
        runtime_memory = stats['runtime_memory_gb']
        delta_memory = runtime_memory - estimated_memory
        ratio = None
        if estimated_memory > 0:
            ratio = runtime_memory / estimated_memory

        summary.append(
            {
                'node_name': node_name,
                'estimated_memory_gb': round(estimated_memory, 6),
                'runtime_memory_gb': round(runtime_memory, 6),
                'delta_memory_gb': round(delta_memory, 6),
                'runtime_vs_estimated_ratio': None if ratio is None else round(ratio, 6),
                'n_samples': stats['n_samples'],
            }
        )

    return sorted(summary, key=lambda row: row['delta_memory_gb'], reverse=True)
