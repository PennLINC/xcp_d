"""Tests for report generation helpers."""

import stat

from xcp_d.reports import core


def test_run_reports_catches_report_init_errors_and_unlocks_svgs(tmp_path, monkeypatch):
    """Report setup failures should be recorded without crashing XCP-D."""
    dataset_dir = tmp_path / 'xcpd'
    report_dir = dataset_dir / 'sub-01'
    figures_dir = report_dir / 'figures'
    figures_dir.mkdir(parents=True)

    svg_file = figures_dir / 'sub-01_desc-bbregister_bold.svg'
    svg_file.write_text('<svg />')
    svg_file.chmod(stat.S_IREAD)

    def _raise_report_init(*args, **kwargs):
        raise PermissionError('no reportlet writes')

    monkeypatch.setattr(core, 'Report', _raise_report_init)

    result = core.run_reports(
        out_dir=report_dir,
        subject_label='01',
        run_uuid='testuuid',
        dataset_dir=dataset_dir,
        subject='01',
    )

    assert result == '01'
    assert svg_file.stat().st_mode & stat.S_IWUSR

    error_file = report_dir / 'logs' / 'report.err'
    assert error_file.exists()
    assert 'PermissionError: no reportlet writes' in error_file.read_text()
