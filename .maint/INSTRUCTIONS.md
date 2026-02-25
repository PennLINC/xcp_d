# Maintenance instructions for XCP-D

Run all commands from the **repository root** unless noted otherwise.

---

## Updating runtime dependencies in Dockerfile.base

`Dockerfile.base` contains non-Python runtime dependencies that are downloaded as
pre-built binaries: FreeSurfer, AFNI, MSM, atlas resources, and system libraries.
These change rarely and live outside the pixi ecosystem.

1. **Edit `Dockerfile.base`** to change the relevant version, URL, or tag:
   - **FreeSurfer**: Update the version in the `curl` URL and any related copy paths.
   - **AFNI**: The current setup downloads the latest `linux_openmp_64.tgz` tarball
     (no pinned version). To pin, switch to a versioned URL.
   - **Atlas resources**: Update download URLs if the hosted tarballs change.
   - **Ubuntu base**: Update the tag (for example, `ARG BASE_IMAGE=ubuntu:jammy-20250730`).
   - **System packages**: Add/remove `apt-get install` entries.

2. **Bump the date tag** in `Dockerfile` to today's date:

   ```dockerfile
   ARG BASE_IMAGE=pennlinc/xcp_d-base:<YYYYMMDD>
   ```

3. **Commit and push.** The CircleCI `image_prep` job checks whether the new base
   image tag exists. If not, it builds `Dockerfile.base`, pushes the date tag and
   `latest`, then builds the main image on top of it.

4. **Verify** the CI image jobs succeed and the new base image appears on Docker Hub
   at `pennlinc/xcp_d-base:<YYYYMMDD>`.

For local testing:

```bash
docker build -f Dockerfile.base -t pennlinc/xcp_d-base:$(date +%Y%m%d) .
docker build --target xcp_d -t pennlinc/xcp_d:dev .
```

---

## Updating individual Python dependencies in pyproject.toml

There are two sections that define Python-ecosystem dependencies:

| Section | Managed by | Examples |
|---------|-----------|----------|
| `[project.dependencies]` | PyPI (pip) | nipype, niworkflows, nibabel, nilearn |
| `[tool.pixi.dependencies]` | conda (pixi) | python, numpy, FSL tools, ANTs, workbench |

Both are resolved together by pixi when generating `pixi.lock`.

1. **Edit the version specifier** in `pyproject.toml`. For example:
   - PyPI: change `"niworkflows == 1.14.4"` to `"niworkflows >= 1.14.4"`
     in `[project.dependencies]` (or another intended constraint).
   - conda: change `scipy = "1.15.*"` to `scipy = "1.16.*"`
     in `[tool.pixi.dependencies]`.

2. **Commit and open a pull request.**

3. **The `pixi-lock.yml` GitHub Action runs automatically** on the PR. It checks
   whether `pixi.lock` is stale and, if so, regenerates it and pushes the
   updated lockfile back to the PR branch.

4. **Review the lockfile update.** If `pixi.lock` is treated as binary in your
   Git configuration, GitHub may not render a useful diff. Inspect locally with
   `git diff HEAD~1 -- pixi.lock` if needed.

5. **Merge the PR** once CI passes.

> **Note:** `pixi lock` only runs on Linux (the project specifies
> `platforms = ["linux-64"]`). Do not rely on macOS for lockfile refresh;
> use CI or a Linux environment.

---

## Updating all Python dependencies at once with pixi update

`pixi update` resolves every dependency to the latest version compatible with the
specifiers in `pyproject.toml` and rewrites `pixi.lock`. **This must be done on
a Linux machine** because the project only targets `linux-64`.

1. **SSH into a Linux machine** (or use a Linux container, GitHub Codespace, etc.).

2. **Clone and check out a branch:**

   ```bash
   git clone https://github.com/PennLINC/xcp_d.git
   cd xcp_d
   git checkout -b update-deps
   ```

3. **Install pixi** (if not already installed):

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

4. **Run `pixi update`:**

   ```bash
   pixi update
   ```

   To update only a specific package:

   ```bash
   pixi update numpy
   ```

5. **Review, commit, and push:**

   ```bash
   git diff pixi.lock | head -200
   git add pixi.lock
   git commit -m "Update pixi lockfile"
   git push -u origin update-deps
   ```

6. **Open a PR.** The `pixi-lock.yml` action will confirm the lockfile is
   already up to date. CI tests will validate nothing is broken.

> **Tip:** `pixi update` can only update within the version ranges declared in
> `pyproject.toml`. To widen a constraint, edit `pyproject.toml` first, then run
> `pixi update`.

---

## Releasing a new version

### 1. Prepare the release branch

- Ensure the release branch (for example, `main`) is up to date and CI checks pass.
- Confirm that the version you plan to release does not already exist as a tag: `git tag -l`.

### 2. Update CITATION.cff

- In `CITATION.cff`, set:
  - **`version`** to the new release version.
  - **`date-released`** to the release date in `YYYY-MM-DD` format.

### 3. (Optional) Update authorship and Zenodo

- If you use Zenodo and have a `.zenodo.json` file, refresh it from `.maint` tables and git history:

  ```bash
  python .maint/update_authors.py zenodo
  ```

- Ensure `.maint/CONTRIBUTORS.md` (and optionally `MAINTAINERS.md`, `PIs.md`, `FORMER.md`) are up to date before running.

### 4. Base image (if applicable)

- The main Docker image is built **FROM** `pennlinc/xcp_d-base:<YYYYMMDD>` (see `Dockerfile`).
- The base image is built from **Dockerfile.base** (runtime dependencies only; no Python stack) and pushed to Docker Hub with both the date tag and `latest`.
- **CircleCI** (`image_prep`) checks if the base image already exists. If it does, base build is skipped. If not, it builds from `Dockerfile.base` and pushes it.
- To release a new base image:
  1. Update the date tag in **Dockerfile** (`ARG BASE_IMAGE=pennlinc/xcp_d-base:YYYYMMDD`).
  2. Commit and push. The next CI run will detect the missing image and build/push it.

### 5. Commit and push release preparation

- Stage and commit all release-related edits (for example, `CITATION.cff`, and Docker base tag if changed).
- Use a clear message, for example: `Prepare release 0.11.0`.
- Push the release branch to the remote.

### 6. Create and push the version tag

- Create an annotated tag for the new version:

  ```bash
  git tag -a 0.11.0 -m "Release 0.11.0"
  ```

- Push the tag:

  ```bash
  git push origin 0.11.0
  ```

- Pushing the tag triggers CircleCI image/deploy jobs and will publish Docker tags when `DOCKERHUB_TOKEN` is set.

### 7. Create the GitHub Release

- On GitHub, open **Releases** -> **Draft a new release**.
- Choose the tag you just pushed.
- Use **"Generate release notes"** (aligned with `.github/release.yml`) or copy from `CHANGES.md`.
- Publish the release.

### 8. Post-release (optional)

- **Git blame ignores**: If you added new commits to `.git-blame-ignore-revs`, run:

  ```bash
  ./.maint/update_ignore_revs.sh
  ```

  Commit the updated `.git-blame-ignore-revs` if it changed.

- **PyPI**: If publishing to PyPI, run your packaging workflow for the release tag.

### Release checklist

- [ ] `CITATION.cff` version and date-released set
- [ ] (Optional) Zenodo / authorship updated
- [ ] (If needed) Base image date tag bumped in Dockerfile
- [ ] Changes committed and pushed
- [ ] Version tag created and pushed
- [ ] GitHub Release created and published
