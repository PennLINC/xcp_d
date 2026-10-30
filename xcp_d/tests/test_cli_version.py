"""Tests for the xcp_d.cli.version module."""

from datetime import datetime, timezone

from packaging.version import Version

from xcp_d.cli import version


class _FakeResponse:
    def __init__(self, tags, next_url=None, status_code=200):
        self._tags = tags
        self._next_url = next_url
        self.status_code = status_code

    def json(self):
        return {
            'next': self._next_url,
            'results': [{'name': tag} for tag in self._tags],
        }


def _patch_get(monkeypatch, pages):
    """Make requests.get return the given responses, keyed by URL."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return pages[url]

    monkeypatch.setattr(version.requests, 'get', fake_get)
    return calls


def test_fetch_latest_tag(monkeypatch):
    """Skip prereleases and non-version tags, and follow pagination."""
    calls = _patch_get(
        monkeypatch,
        {
            version.TAGS_URL: _FakeResponse(
                ['latest', 'unstable', '26.2.0rc1', '0.14.1'],
                next_url='https://example.com/page2',
            ),
            'https://example.com/page2': _FakeResponse(['26.1.1', '0.7.3']),
        },
    )
    assert version._fetch_latest_tag() == Version('26.1.1')
    assert calls == [version.TAGS_URL, 'https://example.com/page2']


def test_fetch_latest_tag_failure(monkeypatch):
    """Return None when Docker Hub can't be reached, refuses, or sends something unexpected."""
    _patch_get(monkeypatch, {version.TAGS_URL: _FakeResponse([], status_code=429)})
    assert version._fetch_latest_tag() is None

    bad_response = _FakeResponse([])
    bad_response.json = lambda: ['not', 'a', 'dict']
    _patch_get(monkeypatch, {version.TAGS_URL: bad_response})
    assert version._fetch_latest_tag() is None

    def raise_error(url, **kwargs):
        raise OSError('no network')

    monkeypatch.setattr(version.requests, 'get', raise_error)
    assert version._fetch_latest_tag() is None


def test_check_latest_cache(monkeypatch, tmp_path):
    """Use a fresh cache, refresh a stale one, and write the result back."""
    monkeypatch.setattr(version.Path, 'home', lambda: tmp_path)
    cachefile = tmp_path / '.cache' / 'xcp_d' / 'latest'
    cachefile.parent.mkdir(parents=True)
    today = datetime.now(tz=timezone.utc).strftime(version.DATE_FMT)

    _patch_get(monkeypatch, {version.TAGS_URL: _FakeResponse(['26.2.0'])})

    cachefile.write_text(f'26.1.0|{today}')
    assert version.check_latest() == Version('26.1.0')

    cachefile.write_text('26.1.0|20000101')
    assert version.check_latest() == Version('26.2.0')
    assert cachefile.read_text() == f'26.2.0|{today}'
