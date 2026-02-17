# AGENTS.md -- XCP-D

This file provides instructions for AI coding agents and human maintainers working on **XCP-D**, a BIDS App for robust postprocessing of fMRI data.

---

## Shared Instructions (All PennLINC BIDS Apps)

The following conventions apply equally to **qsiprep**, **qsirecon**, **xcp_d**, and **aslprep**. All four are PennLINC BIDS Apps built on the NiPreps stack.

### Ecosystem Context

- These projects belong to the [NiPreps](https://www.nipreps.org/) ecosystem and follow its community guidelines.
- Core dependencies include **nipype** (workflow engine), **niworkflows** (reusable workflow components), **nireports** (visual reports), **pybids** (BIDS dataset querying), and **nibabel** (neuroimaging I/O).
- All four apps are containerized via Docker and distributed on Docker Hub under the `pennlinc/` namespace.
- Contributions follow the [NiPreps contributing guidelines](https://www.nipreps.org/community/CONTRIBUTING/).

### Architecture Overview

Every PennLINC BIDS App follows this execution flow:

```
CLI (parser.py / run.py)
  -> config singleton (config.py, serialized as ToML)
    -> workflow graph construction (workflows/*.py)
      -> Nipype interfaces (interfaces/*.py)
        -> BIDS-compliant derivative outputs
```

- **CLI** (`<pkg>/cli/`): `parser.py` defines argparse arguments; `run.py` is the entry point; `workflow.py` builds the execution graph; `version.py` handles `--version`.
- **Config** (`<pkg>/config.py`): A singleton module with class-based sections (`environment`, `execution`, `workflow`, `nipype`, `seeds`). Config is serialized to ToML and passed across processes via the filesystem. Access settings as `config.section.setting`.
- **Workflows** (`<pkg>/workflows/`): Built using `nipype.pipeline.engine` (`pe.Workflow`, `pe.Node`, `pe.MapNode`). Use `LiterateWorkflow` from `niworkflows.engine.workflows` for auto-documentation. Every workflow factory function must be named `init_<descriptive_name>_wf`.
- **Interfaces** (`<pkg>/interfaces/`): Custom Nipype interfaces wrapping external tools or Python logic. Follow standard Nipype patterns: define `_InputSpec` / `_OutputSpec` with `BaseInterfaceInputSpec` / `TraitedSpec`, implement `_run_interface()`.
- **Utilities** (`<pkg>/utils/`): Shared helper functions. BIDS-specific helpers live in `utils/bids.py`.
- **Reports** (`<pkg>/reports/`): HTML report generation using nireports.
- **Data** (`<pkg>/data/`): Static package data (config files, templates, atlases). Accessed via `importlib.resources` or the `acres` package.
- **Tests** (`<pkg>/tests/`): Pytest-based. Unit tests run without external data. Integration tests are gated behind pytest markers and are skipped by default.

### Workflow Authoring Rules

1. Every workflow factory function must be named `init_<name>_wf` and return a `Workflow` object.
2. Use `LiterateWorkflow` (from `niworkflows.engine.workflows`) to enable automatic workflow graph documentation.
3. Define `inputnode` and `outputnode` as `niu.IdentityInterface` nodes to declare the workflow's external API.
4. Connect nodes using `workflow.connect([(source, dest, [('out_field', 'in_field')])])` syntax.
5. Add `# fmt:skip` after multi-line `workflow.connect()` calls to prevent ruff from reformatting them.
6. Include a docstring with `Workflow Graph` and `.. workflow::` Sphinx directive for auto-generated documentation.
7. Use `config` module values (not function parameters) for global settings inside workflow builders.

### Interface Conventions

1. Input/output specs use Nipype traits (`File`, `traits.Bool`, `traits.Int`, etc.).
2. `mandatory = True` for required inputs; provide `desc=` for all traits.
3. Implement `_run_interface(self, runtime)` -- never `run()`.
4. Return `runtime` from `_run_interface`.
5. Set outputs via `self._results['field'] = value`.

### Config Module Usage

```python
from <pkg> import config

# Read a setting
work_dir = config.execution.work_dir

# Serialize to disk
config.to_filename(path)

# Load from disk (in a subprocess)
config.load(path)
```

The config module is the single source of truth for runtime parameters. Never pass global settings as function arguments when they are available via config.

### Testing Conventions

- **Unit tests**: Files named `test_*.py` in `<pkg>/tests/`. Must not require external neuroimaging data or network access.
- **Integration tests**: Decorated with `@pytest.mark.<marker_name>`. Excluded by default via `addopts` in `pyproject.toml`. Require Docker or pre-downloaded test datasets.
- **Fixtures**: Defined in `conftest.py`. Common fixtures include `data_dir`, `working_dir`, `output_dir`, and `datasets`.
- **Coverage**: Configured in `pyproject.toml` under `[tool.coverage.run]` and `[tool.coverage.report]`.

### Documentation

- Built with Sphinx using `sphinx_rtd_theme`.
- Source files in `docs/`.
- Workflow graphs are auto-rendered via `.. workflow::` directives that call `init_*_wf` functions.
- API docs generated via `sphinxcontrib-apidoc`.
- Bibliography managed with `sphinxcontrib-bibtex` and `boilerplate.bib`.

### Docker

- Each app has a custom base image: `pennlinc/<pkg>_build:<version>`.
- The `Dockerfile` installs the app via `pip install` into the base image.
- Entrypoint is the CLI command (e.g., `/usr/local/miniconda/bin/<pkg>`).
- Labels follow the `org.label-schema` convention.

### Release Process

- Versions are derived from git tags via `hatch-vcs` (VCS-based versioning).
- GitHub Releases use auto-generated changelogs configured in `.github/release.yml`.
- Release categories: Breaking Changes, New Features, Deprecations, Bug Fixes, Other.
- Docker images are built and pushed via CI on tagged releases.

### Code Style

- **Formatter**: `ruff format` (target: all four repos).
- **Linter**: `ruff check` with an extended rule set (F, E, W, I, UP, YTT, S, BLE, B, A, C4, DTZ, T10, EXE, FA, ISC, ICN, PT, Q).
- **Import sorting**: Handled by ruff's `I` rule (isort-compatible).
- **Pre-commit**: Uses `ruff-pre-commit` hooks for both linting and formatting.
- **Black is disabled**: `[tool.black] exclude = ".*"` in repos that have migrated to ruff.

### BIDS Compliance

- All outputs must conform to the [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/derivatives/introduction.html) specification.
- Use `pybids.BIDSLayout` for querying input datasets.
- Use `DerivativesDataSink` (from the project's interfaces or niworkflows) for writing BIDS-compliant output files.
- Entity names, suffixes, and extensions must match the BIDS specification.

---

## XCP-D-Specific Instructions

### Project Overview

XCP-D is a BIDS App for postprocessing fMRI data that has been preprocessed by fMRIPrep, nibabies, or similar pipelines. It handles:
- Confound regression and nuisance signal removal
- Temporal filtering (band-pass, motion filtering)
- Framewise displacement-based censoring (scrubbing)
- Despiking
- Smoothing
- Parcellation and connectivity matrix generation
- Surface processing (CIFTI/GIFTI workflows)
- Concatenation of runs within task entity sets
- Executive summary HTML report generation
- Data ingression from non-BIDS formats (ABCD-BIDS, HCP-YA, UK Biobank)

### Repository Details

| Item | Value |
|------|-------|
| Package name | `xcp_d` |
| Default branch | `main` |
| Entry point | `xcp_d.cli.run:main` |
| Python requirement | `>=3.10` |
| Build backend | hatchling + hatch-vcs |
| Linter | ruff ~= 0.15.0 |
| Pre-commit | Yes (ruff v0.6.2) |
| Tox | Yes |
| Docker base | `pennlinc/xcp_d_build:<ver>` |
| Dockerfile | Simple COPY + pip install |

### Key Directories

- `xcp_d/workflows/bold/`: BOLD postprocessing workflows for NIfTI and CIFTI data, plus run concatenation
- `xcp_d/workflows/anatomical/`: Anatomical postprocessing (surfaces, volumes, parcellation)
- `xcp_d/workflows/parcellation.py`: Atlas loading and parcellation workflow
- `xcp_d/workflows/plotting.py`: QC plot generation workflows
- `xcp_d/interfaces/`: Nipype interfaces for censoring, connectivity, concatenation, ANTs, nilearn, workbench, executive summary, plotting
- `xcp_d/ingression/`: Modules for ingressing data from non-BIDS formats (ABCD-BIDS, HCP-YA, UK Biobank)
- `xcp_d/data/atlases/`: Bundled brain atlases (Glasser, Gordon, HCP, Tian, MIDB, MyersLabonte)
- `xcp_d/data/nuisance/`: YAML configs for nuisance regression strategies
- `xcp_d/data/executive_summary_templates/`: Jinja2 HTML templates for the executive summary report

### Version Management

XCP-D uses `__about__.py` for version metadata:
```python
from xcp_d.__about__ import __copyright__, __credits__, __packagename__, __version__
```
This is different from qsiprep/qsirecon which import `__version__` directly from `_version.py`. Both patterns work; harmonization is a roadmap item.

### Linting Status (Reference for Other Repos)

XCP-D has the **cleanest ruff configuration** of all four repos, with only 3 suppressed rules:
- `S105`: Hardcoded password detection (false positives)
- `S311`: Random not for crypto (intentional)
- `S603`: Subprocess with shell=True (trusted commands only)

This minimal ignore set is the target for the other three repos.

### Ingression Modules

`xcp_d/ingression/` contains adapters for non-BIDS input formats:
- `abcdbids.py`: ABCD-BIDS format
- `hcpya.py`: HCP Young Adult format
- `ukbiobank.py`: UK Biobank format
- `utils.py`: Shared ingression utilities

When adding support for new input formats, create a new module in this directory following the existing patterns.

### Executive Summary

XCP-D generates rich HTML executive summary reports using Jinja2 templates in `xcp_d/data/executive_summary_templates/`. These include:
- BrainSprite interactive brain viewers
- Anatomical registration quality plots
- Task-specific static QC plots

The `xcp_d/interfaces/execsummary.py` interface and `xcp_d/utils/execsummary.py` utilities handle generation.

### `fill_doc` Decorator

XCP-D uses a `@fill_doc` decorator (from `xcp_d.utils.doc`) to inject shared parameter documentation into function docstrings. Use it on any workflow init function that accepts standard parameters documented in the shared parameter dictionary.

### Config Hashing

XCP-D implements config hashing (`config.hash_config()`) to create unique identifiers for pipeline configurations. This is used to detect when outputs need to be regenerated due to parameter changes.

---

## Cross-Project Development Roadmap

This roadmap covers harmonization work across all four PennLINC BIDS Apps (qsiprep, qsirecon, xcp_d, aslprep) to reduce maintenance burden.

### Phase 1: Bring qsirecon to parity

1. **Migrate qsirecon from flake8+black+isort to ruff** -- copy the `[tool.ruff]` config from xcp_d's `pyproject.toml` and remove `[tool.black]`, `[tool.isort]`, `[tool.flake8]` sections.
2. **Add `.pre-commit-config.yaml` to qsirecon** -- identical to the config used by qsiprep, xcp_d, and aslprep.
3. **Add `tox.ini` to qsirecon** -- copy from qsiprep or xcp_d (they are identical).
4. **Add `.github/dependabot.yml` to qsirecon**.
5. **Reformat qsirecon codebase** -- run `ruff format` to switch from double quotes to single quotes.

### Phase 2: Standardize across all four repos

6. **Rename qsiprep default branch** from `master` to `main` and update `.github/workflows/lint.yml`.
7. **Rename aslprep test extras** from `test` to `tests` for consistency with the other three repos.
8. **Converge on version management** -- recommend the simpler `_version.py` direct-import pattern (used by qsiprep/qsirecon). Migrate xcp_d and aslprep away from `__about__.py`.
9. **Pin the same ruff version** in all four repos' dev dependencies and `.pre-commit-config.yaml`.
10. **Harmonize ruff ignore lists** -- adopt xcp_d's minimal set (`S105`, `S311`, `S603`) as the target; fix suppressed rules in qsiprep and aslprep incrementally.

### Phase 3: Shared infrastructure

11. **Extract a reusable GitHub Actions workflow** for lint + codespell + build checks, hosted in a shared repo (e.g., `PennLINC/.github`).
12. **Standardize Dockerfile patterns** -- adopt multi-stage wheel builds (as qsiprep does) across all four repos.
13. **Create a shared `pennlinc-style` package or cookiecutter template** providing `pyproject.toml` lint/test config, `.pre-commit-config.yaml`, `tox.ini`, and CI workflows.
14. **Evaluate `nipreps-versions` calver** -- the `raw-options = { version_scheme = "nipreps-calver" }` line is commented out in all four repos. Decide whether to adopt it.

