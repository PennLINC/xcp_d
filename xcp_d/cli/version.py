# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Version CLI helpers."""

from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

import requests

from xcp_d import __version__

RELEASE_EXPIRY_DAYS = 14
DATE_FMT = '%Y%m%d'
TAGS_URL = 'https://hub.docker.com/v2/namespaces/pennlinc/repositories/xcp_d/tags?page_size=100'
MAX_TAG_PAGES = 5


def _fetch_latest_tag():
    """Return the highest non-prerelease version among the Docker Hub tags, or None."""
    from packaging.version import InvalidVersion, Version

    versions = []
    url = TAGS_URL
    for _ in range(MAX_TAG_PAGES):
        response = None
        with suppress(Exception):
            response = requests.get(url=url, timeout=1.0)

        if not response or response.status_code != 200:
            break

        try:
            page = response.json()
            tags = [tag['name'] for tag in page['results']]
        except Exception:  # noqa: BLE001
            break

        for tag in tags:
            with suppress(InvalidVersion, TypeError):
                versions.append(Version(tag))

        url = page.get('next')
        if not url:
            break

    versions = [rel for rel in versions if not rel.is_prerelease]
    return max(versions) if versions else None


def check_latest():
    """Determine whether this is the latest version."""
    from packaging.version import InvalidVersion, Version

    latest = None
    date = None
    outdated = None
    now = datetime.now(tz=timezone.utc)
    cachefile = Path.home() / '.cache' / 'xcp_d' / 'latest'
    try:
        cachefile.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        cachefile = None

    if cachefile and cachefile.exists():
        try:
            latest, date = cachefile.read_text().split('|')
        except Exception:  # noqa: S110, BLE001
            pass
        else:
            try:
                latest = Version(latest)
                date = datetime.strptime(date, DATE_FMT).astimezone(timezone.utc)
            except (InvalidVersion, ValueError):
                latest = None
            else:
                if abs((now - date).days) > RELEASE_EXPIRY_DAYS:
                    outdated = True

    if latest is None or outdated is True:
        latest = _fetch_latest_tag()

    if cachefile is not None and latest is not None:
        with suppress(OSError):
            cachefile.write_text(f'{latest}|{now.strftime(DATE_FMT)}')

    return latest


def is_flagged():
    """Check whether current version is flagged."""
    # https://raw.githubusercontent.com/pennlinc/xcp_d/main/.versions.json
    flagged = ()
    response = None
    with suppress(Exception):
        response = requests.get(
            url="""\
https://raw.githubusercontent.com/pennlinc/xcp_d/main/.versions.json""",
            timeout=1.0,
        )

    if response and response.status_code == 200:
        flagged = response.json().get('flagged', {}) or {}

    if __version__ in flagged:
        return True, flagged[__version__]

    return False, None
