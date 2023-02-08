"""Tests for the xcp_d.utils.qcmetrics module."""
import numpy as np

from xcp_d.utils import qcmetrics


def test_compute_dvars():
    """Run a smoke test for xcp_d.utils.qcmetrics.compute_dvars."""
    n_volumes, n_vertices = 100, 10000
    data = np.random.random((n_vertices, n_volumes))

    dvars = qcmetrics.compute_dvars(data)
    assert dvars.shape == (n_volumes,)
