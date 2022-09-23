.. include:: links.rst

xcp_d Developer Guide
=====================

This guide provides a more detailed description of the organization and preferred coding style for xcp_d,
for prospective code contributors.

Maintaining xcp_d
-----------------

Making a Release
````````````````

To make an xcp_d release, complete the following steps:

1.  Choose a new version tag, according to the semantic versioning standard.
2.  Modify the CITATION.cff file, updating the version number and release date.
3.  In GitHub's release tool, draft a new release.
    The release title should be the same as the new tag (e.g., ``1.0.0``).

    For pre-releases, use release candidate terminology (e.g., ``0.0.12rc1``) and select the "This is a pre-release" option.

    At the top of the release notes, add some information summarizing the release.

    Once the release notes have been completed, you can publish the release.
    This will make the release on GitHub.
