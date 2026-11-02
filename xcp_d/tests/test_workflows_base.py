"""Tests for helpers in xcp_d.workflows.base."""

from xcp_d.workflows import base


def test_get_concat_exact_scans(monkeypatch):
    """Exact-time scans are only calculated for scans that will be concatenated."""
    warnings = []
    monkeypatch.setattr(base.LOGGER, 'warning', warnings.append)
    kwargs = {'scan_length': 300, 't_r': 0.8, 'bold_file': 'sub-01_task-rest_bold.nii.gz'}

    # Scans with a run or dir entity are concatenated, so exact-time scans are calculated
    exact_scans = base._get_concat_exact_scans(
        correlation_lengths=['all', 300],
        multiscans=True,
        **kwargs,
    )
    assert exact_scans == [375]
    assert not warnings

    # Without exact times, nothing is calculated and there is nothing to warn about
    exact_scans = base._get_concat_exact_scans(
        correlation_lengths=['all'],
        multiscans=False,
        **kwargs,
    )
    assert exact_scans == []
    assert not warnings

    # Exact times for a scan that will not be concatenated are skipped with a warning
    exact_scans = base._get_concat_exact_scans(
        correlation_lengths=[300],
        multiscans=False,
        **kwargs,
    )
    assert exact_scans == []
    assert len(warnings) == 1
    assert '--create-matrices' in warnings[0]
