"""Tests for the xcp_d.workflows.parcellation module."""

import numpy as np
import pytest

from xcp_d import config
from xcp_d.tests.tests import mock_config
from xcp_d.workflows.parcellation import init_parcellate_cifti_wf


@pytest.mark.parametrize('min_coverage', [0.0, 0.4, 0.5, 0.7, 1.0])
def test_parcellate_cifti_wf_coverage_threshold(min_coverage):
    """Check that CIFTI parcels with coverage equal to min_coverage are kept.

    This matches the NIfTI workflow, which only drops parcels with coverage below the threshold.
    """
    with mock_config():
        config.workflow.min_coverage = min_coverage
        wf = init_parcellate_cifti_wf(mem_gb={'bold': 1, 'volume': 1})

    expression = wf.get_node('threshold_coverage').inputs.expression
    operator, threshold = expression.split()[1:]
    assert operator == '>='
    threshold = float(threshold)

    # Workbench stores parcel coverage as a float32 fraction of covered vertices.
    n_vertices = 1000
    n_covered = round(min_coverage * n_vertices)
    coverage = np.float32(n_covered / n_vertices)
    assert coverage >= threshold

    if n_covered > 0:
        lower_coverage = np.float32((n_covered - 1) / n_vertices)
        assert lower_coverage < threshold
