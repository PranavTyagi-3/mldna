# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [0.2.0] - 2026-10-10

Hardening release from a full issue review (D#37-D#70). **Upgrade notes:** each
producer repo needs a one-time `mldna id` and a commit of `mldna.yaml` before it
can publish again (D#69); `mldna push` no longer stages anything, run
`mldna stage` first (D#68); registry names are now `<namespace>-<model>`, so
consumer pins must use that name (D#58).

### Added

- `mldna stage`: runs `dvc add`, stages the files, and lists them for review
  before `mldna push` commits only what is staged (D#68).
- `mldna id` and `project.model_id`: an immutable per-project id stamped on the
  registry model at first publish; consumers may pin `id:` in `models.lock.yaml`
  and `fetch`/`pull`/`publish` refuse a mismatch (D#69).
- `mldna attach`: recover a run whose final tags were lost to a network failure,
  permanently labeled as attached retroactively (D#61).
  Every run's record is kept in `.mldna/runs/`; `--run-id` attaches an earlier one,
  and the publish refusal now points to `attach` (D#72).
- `mldna workspace new | list | remove`: git worktrees sharing the DVC cache (D#62).
- `project.namespace`, prefixed onto experiment and registry names (D#58).
- `mldna doctor --timeout` (default 10s, no retries) and a `large_tracked_files`
  check; `push` confirms before committing large or model-shaped files (D#65, D#66).
- `mldna pull --run-id` accepts the 8-character id shown by `mldna runs`, or any
  prefix of 4+ characters; `--commit` selects among differing lineages (D#64).
- Publish confirmation shows the run's recorded command line (opt-in via
  `tracking.record_argv`, off by default) and allowlisted environment variables,
  labeled "recorded, not verified"; when no command line was recorded it says
  so (D#70, D#75).
- Per-DVC-out data hashes; manifest_version 2 (D#63).
- `mldna_origin` on the registry model: the producer's git origin (credentials
  stripped), stamped at first publish. Publishing from a different origin warns
  and needs a person's confirmation, even with `--yes`; `mldna id --new`
  replaces a copied id (D#77).

### Changed

- One publish per model file per run (keyed by SHA-256, run tag
  `published__<hash prefix>`), so a run that saved several models can publish
  each. Runs published by older mldna stay refused (D#78).
- The model-id mismatch refusal now first offers restoring the registered id
  (D#77).

- `fetch` re-verifies files already on disk and repairs them; ownership marker
  upgraded to an identity token (D#38, D#46, D#53).
- Two-plane clean gate: `track_run` refuses on unpushed data changes and when
  cleanliness cannot be determined (D#45); the run lock is atomic (D#49).
- Refusal and error texts now show git's real message, name the parent project
  when run from a subfolder, and no longer suggest `allow_dirty` where it would
  not help.

### Removed

- The committed-secret scanner and its `doctor` check (D#67). Scaffolded
  `.gitignore` entries remain; use `mldna stage` to review what gets committed.

### Fixed

- Security: `track_run` previously recorded `sys.argv` unredacted as the
  `invocation_argv` run tag, readable by every user of the tracking server.
  Capture is now off unless `tracking.record_argv: true` (D#75). Review runs
  already on a server for secrets in that tag.

- A failure during `track_run` setup (for example a YAML typo) no longer leaks
  the run lock for the rest of the process.
- A refused connection is no longer reported as a timeout; a dropped upload no
  longer asserts a size limit was the cause.
- The losing publish of a concurrent race now tags its own version
  `abandoned=true` before failing, and `fetch` and `pull` refuse that version;
  previously it stayed untagged and fetchable (D#76).

## [0.1.0] - 2026-07-26

First cut of the complete core loop, developed and verified end-to-end against
real infrastructure (local MLflow file store, a live self-hosted MLflow server,
and a real Azure ML workspace).

### Added

- **Producer CLI**: `init` (interactive scaffold: tracking source, auth mode,
  DVC backend), `push` (dvc add → git commit → dvc push → git push,
  idempotent), `publish` (content-hash run resolution plus the full gate
  chain: run gates, commit reachability, bundle structure, conversion
  provenance, re-publish refusal), `pull`, `runs`, `doctor` (seven independent
  health checks (the committed-secret scan was removed in 0.2.0)).
- **Consumer CLI**: `init --consumer` (lockfile + CI snippet scaffold),
  `fetch` (exact pinned versions only — manifest-first, checksum-verified,
  atomic swap, idempotent, hard-fails on a `recalled` version).
- **Python API**: `mldna.track_run()` (context manager/decorator: run lock
  with stale-PID reclaim, clean-tree gate with labeled `allow_dirty` escape
  hatch, identity resolution, model-file discovery and per-file SHA-256 tags,
  `pip freeze` capture) and `mldna.track_convert()` (provenance-links
  converted formats to their source model; numerical equivalence is
  deliberately not claimed).
- **Three tracking backends**, switched by `mldna.yaml` alone: local file
  store (default, zero config), any plain MLflow server via
  `mlflow.tracking_uri`, or an Azure ML workspace.
- **Azure adapter** (`pip install "mldna-lineage[azure]"`): AAD and service-principal
  shared-secret auth on both the tracking and DVC-blob planes
  (`configure aml-auth` / `configure blob-auth`), workspace and blob-container
  discovery during `init`.
- The base install is local-only with zero Azure dependencies; every
  Azure-touching path refuses cleanly with the exact `pip install` fix when
  the extra is missing.
