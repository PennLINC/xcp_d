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
   whether `pixi.lock` is stale and, if so, regenerates it.

4. **Lockfile behavior depends on PR source:**
   - **Same-repo branch PRs:** `pixi-lock.yml` runs `pixi lock --check` with a
     read-only token, then restores the committed `pixi.lock` (the check rewrites
     the workspace package's own version, which changes with every commit, even
     when it passes). If the check failed, it regenerates `pixi.lock` with
     `datalad run ... -- pixi lock`, uploads the commit as a patch artifact, and
     fails. `pixi-lock-push.yml` then validates the patch and pushes a commit that
     changes only `pixi.lock` to the PR branch. Checks on the bot's commit do not
     start on their own: GitHub may hold them as "action required" until a
     maintainer approves them, or not queue them at all. Push another commit, or
     close and reopen the PR, to run them.
   - **Fork PRs:** the workflow only checks the lockfile (`pixi lock --check`), with
     read-only permissions, and does **not** push. If the check fails, update
     `pixi.lock` manually in a Linux environment and push from the contributor branch.
   - **PRs that delete `pixi.lock`** fail the check with a request to restore it;
     the workflow does not regenerate a deleted lockfile.

5. **Review the lockfile update.** If `pixi.lock` is treated as binary in your
   Git configuration, GitHub may not render a useful diff. Inspect locally with
   `git diff HEAD~1 -- pixi.lock` if needed.

6. **Merge the PR** once CI passes.

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

## CI workflow trigger conditions

This section explains what causes each CI workflow step/job to run.
In this repository, file changes only control a subset of behavior.

### GitHub Actions: `.github/workflows/pixi-lock.yml` and `pixi-lock-push.yml`

Lockfile automation is split across two workflows, adapted from nipreps/fmriprep#3722,
so that the PR's code never runs in a job that can write to the repository.

**`pixi-lock.yml` ("Pixi lock")** is triggered on every `pull_request` event and has
one job, `check-lockfile`, with a read-only token and no persisted credentials. It
runs the PR's code (pixi and DataLad build the workspace package).

- Always runs `Checkout pull request` and `Check pull request for dependency edits`.
  The latter looks for `pyproject.toml` or `pixi.lock` edits anywhere in the PR
  (`git diff base...HEAD`), not only in the latest commit. Editing other files alone
  still starts the workflow, but the pixi steps are skipped and the job passes.
- If the PR touched those files, runs `prefix-dev/setup-pixi` and `Check lockfile`,
  which runs `pixi lock --check`, records `stale=true|false`, and resets `pixi.lock`
  with `git checkout -- pixi.lock`. If the PR deletes `pixi.lock`, the step fails
  with an error asking the author to restore it; a deleted lockfile is never
  regenerated, because the push workflow accepts only a patch that modifies
  `pixi.lock`, not one that creates it.
- If the lockfile is stale and the PR is from a branch of this repository, runs
  `Set git identity`, `Install the latest version of uv`, `Update lockfile`
  (`uvx datalad run -i pixi.lock -i pyproject.toml -o pixi.lock -- pixi lock`),
  `Export lockfile patch` (`git format-patch -1` plus the PR number), and
  `Upload lockfile patch` (artifact `pixi-lock`, kept for one day). The export step
  fails instead if `datalad run` made no commit, which happens when the check failed
  for a reason other than a stale lockfile.
- If the lockfile is stale, `Fail on stale lockfile` fails the job, for fork and
  same-repo PRs alike. The failure keeps a stale lockfile from looking green, and it
  is what starts the push workflow.

**`pixi-lock-push.yml` ("Pixi lock push")** is triggered by `workflow_run` when a
"Pixi lock" run completes. Its `push-lockfile` job runs only when that run came from a
`pull_request` event, failed, and had a head branch in this repository. It has
`contents: write`, `actions: read` and `pull-requests: read`, and never runs the
PR's code. The artifact was made by the PR's code, so it is treated as untrusted:

- `Download lockfile patch` exits quietly if the run uploaded no `pixi-lock` artifact
  (the run failed for another reason, or the PR is from a fork). The zip is kept
  whole, and nothing is extracted by its untrusted entry names.
- `Validate pull request` requires the zip to hold exactly `pixi-lock.patch` and
  `pr-number`, the PR number to be numeric, and the PR to be open with the same head
  SHA, repository and branch as the run, and a valid branch name. A moved head means
  a newer run will handle it.
- `Fetch pull request head` fetches the head commit with plain `git`, without
  credentials (the repository is public), rather than `actions/checkout`.
- `Apply lockfile patch` requires the patch to modify `pixi.lock` and nothing else
  (no other paths, renames, deletions or mode changes), applies only its diff, and
  commits it as `nipreps[bot]` with a DataLad run record written in the workflow.
  The patch's own author and message are discarded: a forged DataLad record would
  run arbitrary commands on `datalad rerun`. It then re-checks that the result is
  one commit on the PR head that changes only `pixi.lock`.
- `Push lockfile commit` pushes non-forced, with the token passed as an
  `http.extraheader` on the `git push` command line.

`workflow_run` always runs the copy of `pixi-lock-push.yml` on `main`, so changes to
it, and to the "Pixi lock" workflow name it matches, take effect only after merge.

Checks on the bot's commit do not start on their own: GitHub may hold them as
"action required" until a maintainer approves them, or not queue them at all. Push
another commit, or close and reopen the PR, to run them.

Do not move `pixi lock --check` back under `datalad run`. The check exits 0 but still
rewrites the workspace package's hatch-vcs version in `pixi.lock`, so running it there
records that rewrite on every run, and each bot push queues another run (seen on
PennLINC/qsirecon#396).

`check-lockfile` is the job to require as a status check. Third-party actions in both
workflows are pinned to commit SHAs with a `# vX.Y.Z` comment.

### CircleCI: `.circleci/config.yml`

CircleCI does **not** use file path filters in this config. Jobs are selected by
workflow filters (branch/tag) and can be halted by commit-message markers.

#### Branch/tag filters that control job execution

- `image_prep`: runs on all branches and all tags.
- `get_data`, integration jobs, `pytests`, `merge_coverage`:
  - Run on all tags.
  - Run on branches except names matching `docs?/.*` or `tests?/.*`.
- `deployable` and `deploy_docker`:
  - Run on `main` and on all tags.

#### Commit-message markers that halt jobs

- Integration jobs stop early if latest commit message contains
  `[skip integration]` or `[skip_integration]` (case-insensitive).
- `pytests` stops early if latest commit message contains
  `[skip pytests]` or `[skip_pytests]` (case-insensitive).

#### File changes that indirectly affect CircleCI behavior

These files are used in the `build-v3` cache key:

- `Dockerfile`
- `pixi.lock`

Changing either of them changes the cache key and invalidates the prior
`imageprep-success.marker`, so `image_prep` sets:

- `BUILD_PRODUCTION_IMAGE=1`
- `BUILD_TEST_IMAGE=1`

That means the production and test image build steps run. Base image rebuild is
controlled separately by whether the `BASE_IMAGE` tag in `Dockerfile` already
exists in Docker Hub.

#### `image_prep` rebuild matrix (by file edit)

- **Edit `Dockerfile` and change `ARG BASE_IMAGE=...` to a new/missing tag**
  - Base image: **rebuilt** (`Dockerfile.base` build runs because manifest is missing)
  - Production image: **rebuilt**
  - Test image: **rebuilt**
- **Edit `Dockerfile` without changing `BASE_IMAGE` tag**
  - Base image: rebuilt **only if** current `BASE_IMAGE` tag is missing in registry
  - Production image: **rebuilt**
  - Test image: **rebuilt**
- **Edit `Dockerfile.base` only**
  - Base image: **not automatically rebuilt** if `BASE_IMAGE` tag already exists
  - Production image: **not rebuilt**
  - Test image: **not rebuilt**
  - Note: to force base rebuild from updated `Dockerfile.base`, also bump
    `ARG BASE_IMAGE=...` in `Dockerfile` to a new date/tag.
  - Caveat: if build cache is unavailable for unrelated reasons, CircleCI may
    still rebuild production/test images.
- **Edit `pixi.lock` only**
  - Base image: **not rebuilt** unless `BASE_IMAGE` tag is missing
  - Production image: **rebuilt**
  - Test image: **rebuilt**
- **Edit none of `Dockerfile` or `pixi.lock`**
  - If cache/marker is restored: production/test builds are skipped by default
  - Base image still goes through existence check and rebuilds only if missing

- `image_prep` reads `BASE_IMAGE` from `Dockerfile`.
- If that base image tag does not exist in Docker Hub, CircleCI builds
  `Dockerfile.base` and pushes the base image.

Practical implication: if your goal is to rebuild the base image, edit
`Dockerfile` to point to a new `BASE_IMAGE` tag; editing `Dockerfile.base` alone
is not sufficient when that tag already exists remotely.

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
