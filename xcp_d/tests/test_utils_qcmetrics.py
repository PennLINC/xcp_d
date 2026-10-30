"""Tests for the xcp_d.utils.qcmetrics module."""

import numpy as np

from xcp_d.utils import qcmetrics


def test_compute_dvars():
    """Run a smoke test for xcp_d.utils.qcmetrics.compute_dvars."""
    n_volumes, n_vertices = 100, 10000
    data = np.random.random((n_vertices, n_volumes))

    dvars, std_dvars = qcmetrics.compute_dvars(datat=data)
    assert dvars.shape == (n_volumes,)
    assert std_dvars.shape == (n_volumes,)


def test_compute_dvars_without_variance():
    """Check DVARS for data in which no vertex varies, such as an exactly fitted denoising."""
    n_volumes, n_vertices = 16, 100
    data = np.full((n_vertices, n_volumes), 1e-12, dtype=np.float32)

    dvars, std_dvars = qcmetrics.compute_dvars(datat=data)
    assert dvars.shape == (n_volumes,)
    assert np.all(dvars == 0)
    assert std_dvars.shape == (n_volumes,)
    assert std_dvars[0] == 0
    assert np.all(np.isnan(std_dvars[1:]))
