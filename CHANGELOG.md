# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once it reaches 1.0.

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
  health checks including committed-secret scanning).
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
